# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from importlib.machinery import SourceFileLoader

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def thr_tag(thr):
    return f"{thr:.2f}".replace(".", "p")


def iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--top_k", type=int, default=8)
    args = parser.parse_args()

    split = args.split
    tag = thr_tag(args.threshold)

    data_dir = cfg.FORECAST_DIR
    unet_dir = cfg.UNET_RUN_DIR
    corr_dir = cfg.UNET_RUN_DIR / "09_unet_plus_figure_verifier"
    out_dir = cfg.UNET_RUN_DIR / "11_correction_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_dir / f"X_{split}.npy", mmap_mode="r")
    y = np.load(data_dir / f"y_{split}.npy", mmap_mode="r")
    dates = np.load(data_dir / f"target_dates_{split}.npy", allow_pickle=True)

    base = np.load(unet_dir / f"pred_mask_{split}.npy", mmap_mode="r")
    corr = np.load(corr_dir / f"corrected_mask_{split}_thr{tag}.npy", mmap_mode="r")

    rows = []
    for i in range(len(y)):
        removed = np.logical_and(base[i] == 1, corr[i] == 0).sum()
        if removed <= 0:
            continue
        base_iou = iou(base[i], y[i])
        corr_iou = iou(corr[i], y[i])
        rows.append((corr_iou - base_iou, removed, i, base_iou, corr_iou))

    rows = sorted(rows, reverse=True)
    selected = rows[:args.top_k]

    print("[SELECTED]")
    for delta, removed, i, biou, ciou in selected:
        print(i, dates[i], "delta_iou=", delta, "removed=", removed, "base_iou=", biou, "corr_iou=", ciou)

        ssta_last = X[i, -1]
        removed_mask = np.logical_and(base[i] == 1, corr[i] == 0)

        fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))

        axes[0].imshow(ssta_last)
        axes[0].set_title("Input SSTA\nlast day")

        axes[1].imshow(y[i], vmin=0, vmax=1)
        axes[1].set_title("Ground truth\nMHW")

        axes[2].imshow(base[i], vmin=0, vmax=1)
        axes[2].set_title(f"U-Net\nIoU={biou:.3f}")

        axes[3].imshow(corr[i], vmin=0, vmax=1)
        axes[3].set_title(f"Corrected\nIoU={ciou:.3f}")

        axes[4].imshow(removed_mask, vmin=0, vmax=1)
        axes[4].set_title("Removed\ncomponents")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"{split} sample={i}, target date={dates[i]}, ΔIoU={delta:.4f}", fontsize=11)
        plt.tight_layout()

        out_file = out_dir / f"{split}_sample{i}_thr{tag}_delta{delta:.4f}.png"
        plt.savefig(out_file, dpi=250)
        plt.close()

    print("[SAVE DIR]", out_dir)


if __name__ == "__main__":
    main()
