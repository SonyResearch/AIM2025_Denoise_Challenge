import os
import argparse
import torch
import yaml
import numpy as np
from tqdm import tqdm
from glob import glob
import rawpy
from models.ELD_models import UNetSeeInDark
from matplotlib import pyplot as plt


def vis_rggb(img):
    r, g1, g2, b = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
    img = np.stack([r, (g1 + g2) / 2, b], axis=-1)
    img = np.clip(img, 0, 1)
    img = img ** (1 / 2.2)
    img = np.clip(img, 0, 1)
    return img


def infer_and_save_for_submission(model, args):
    with open(args.camera_config_dir, "r") as f:
        all_cam_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cam in args.camera_models:
        ## common settings
        h_start, w_start, h_end, w_end = all_cam_cfg[cam]["valid_roi"]
        suffix = "ARW" if "sony" in cam else "CR2"
        print(f"Processing camera: {cam}")

        ## we have both paired_input for FR-IQA, and in_the_wild for NR-IQA
        for img_type in ["paired", "in_the_wild"]:
            ## make dir for saving
            curr_save_dir = os.path.join(args.save_dir, cam, img_type)
            if not os.path.exists(curr_save_dir):
                os.makedirs(curr_save_dir)

            ## process imgs
            all_noisy_dirs = glob(os.path.join(args.test_data_dir, cam, f"{img_type}/*.{suffix}"))
            for noisy_dir in tqdm(all_noisy_dirs):
                ## ------- get capture_info
                scene_id, iso, dgain = os.path.basename(noisy_dir).split(".")[0].split("_")
                iso, dgain = int(iso[3:]), float(dgain[5:])

                ## ------- STEP 1: load data (load bayer, subtract ds, normalizaton, pack to rggb, scale by dgain, and clip).
                ## ------- PLEASE change to suit your pre-proceesings -------
                rf = rawpy.imread(noisy_dir)
                wb = np.array(rf.camera_whitebalance)
                wb = wb / wb[1]
                wl, bl = float(rf.white_level), np.mean(rf.black_level_per_channel)
                noisy = np.array(rf.raw_image).astype(np.float32)

                dark_shading = np.load(os.path.join(args.dark_shading_dir, cam, f"calib_res/dark_shading_iso{iso}.npy"))
                noisy = noisy - dark_shading  # subtract dark shading
                noisy = noisy[h_start:h_end, w_start:w_end]  # crop valid roi
                noisy = (noisy - bl) / (wl - bl)  # normalize
                noisy = np.stack(
                    [noisy[0::2, 0::2], noisy[0::2, 1::2], noisy[1::2, 0::2], noisy[1::2, 1::2]],
                    axis=-1,
                )  # pack to rggb
                noisy = noisy * dgain  # apply digital gain
                noisy = np.clip(noisy, float("-inf"), 1)  # clip

                ## ------- STEP 2: forward to model -------
                noisy = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).float().to(args.device)  # [1, 4, h, w]
                denoised_rggb = model(noisy)  # [1, 4, h, w]

                ## ------- STEP 3: format to save, make sure the final npy file is saved as rggb
                ## ------- in shape of [h, w, 4] and in 16 bit (Just clip the net's output to [0, 1] and scale by 65535,
                #  ------- no need to apply the white-level and black-level back)
                denoised_rggb = denoised_rggb.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
                denoised_rggb = np.clip(denoised_rggb, 0, 1)
                denoised_rggb = np.uint16(denoised_rggb * 65535)
                np.save(os.path.join(curr_save_dir, os.path.basename(noisy_dir).replace(suffix, "npy")), denoised_rggb)


def main(args):
    model = UNetSeeInDark().to(args.device)
    model.load_state_dict(torch.load(args.checkpoint_dir, map_location="cpu", weights_only=False)["model"], strict=True)
    model.eval()
    infer_and_save_for_submission(model, args)


##--------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## dir
    parser.add_argument("--camera_config_dir", type=str, default="./datasets/camera_config.yaml")
    parser.add_argument("--test_data_dir", type=str, default="/data2/feiran/datasets/aim_challenge/test_phase_release")
    parser.add_argument(
        "--dark_shading_dir", type=str, default="/data2/feiran/datasets/aim_challenge/dev_phase_release"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="/data2/feiran/AIM2025_Denoise_Challenge/checkpoints/epoch_500.bin"
    )
    parser.add_argument("--device", type=str, default="cuda:3")

    ## DO NOT change below setups
    parser.add_argument("--save_dir", type=str, default="./saved_res_for_submission")
    parser.add_argument("--camera_models", type=list, default=["canon70d", "sonya6700", "sonya7r4", "sonyzve10m2"])

    _args = parser.parse_args(args=[])
    main(_args)
