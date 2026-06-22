#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize valid / invalid candidate MHW events constructed by 06c.

Input:
  outputs/06c_neurrl_event_dataset_from_multichannel_h10_l5/figure_event_{split}.npz
  outputs/03b_forecast_dataset_multichannel_h10_l5/y_{split}.npy
  outputs/03b_forecast_dataset_multichannel_h10_l5/target_dates_{split}.npy
  outputs/04c_unet_multichannel_h10_l5/pred_mask_{split}.npy

Each figure_event sample:
  X[:, 0:10]   = SSTA history
  X[:, 10:20]  = historical MHW mask
  X[:, 20:30]  = historical exceed90 mask
  X[:, 30:40]  = threshold_gap history
  X[:, 40]     = predicted component mask
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage


def crop_2d(arr: np.ndarray, center_r: float, center_c: float, crop_size: int, fill_value=0):
    """Crop a square patch with zero padding."""
    h, w = arr.shape
    half = crop_size // 2

    cr = int(round(center_r))
    cc = int(round(center_c))

    r0 = cr - half
    r1 = r0 + crop_size
    c0 = cc - half
    c1 = c0 + crop_size

    out = np.full((crop_size, crop_size), fill_value, dtype=arr.dtype)

    sr0 = max(r0, 0)
    sr1 = min(r1, h)
    sc0 = max(c0, 0)
    sc1 = min(c1, w)

    dr0 = sr0 - r0
    dr1 = dr0 + (sr1 - sr0)
    dc0 = sc0 - c0
    dc1 = dc0 + (sc1 - sc0)

    if sr1 > sr0 and sc1 > sc0:
        out[dr0:dr1, dc0:dc1] = arr[sr0:sr1, sc0:sc1]

    return out


def get_component_mask(pred_mask: np.ndarray, component_id: int, expected_area: float):
    """
    Reconstruct connected component from full pred_mask.

    Try 8-connectivity and 4-connectivity; if component_id does not match,
    fallback to the component with closest area.
    """
    pred_bool = pred_mask.astype(bool)

    structures = [
        np.ones((3, 3), dtype=int),  # 8-connectivity
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),  # 4-connectivity
    ]

    best = None
    best_area_diff = float("inf")

    for structure in structures:
        lab, nlab = ndimage.label(pred_bool, structure=structure)

        # First try direct component_id
        if 1 <= component_id <= nlab:
            comp = lab == component_id
            area = int(comp.sum())
            if abs(area - expected_area) <= max(2, 0.02 * expected_area):
                return comp

        # Fallback: closest area
        if nlab > 0:
            areas = ndimage.sum(pred_bool, lab, index=np.arange(1, nlab + 1))
            areas = np.asarray(areas)
            idx = int(np.argmin(np.abs(areas - expected_area))) + 1
            area_diff = abs(float(areas[idx - 1]) - float(expected_area))
            if area_diff < best_area_diff:
                best_area_diff = area_diff
                best = lab == idx

    return best


def candidate_centers(comp: np.ndarray):
    """Return candidate crop centers from a full-size component mask."""
    rr, cc = np.where(comp)
    if len(rr) == 0:
        return [(comp.shape[0] / 2, comp.shape[1] / 2)]

    centroid = (float(rr.mean()), float(cc.mean()))
    bbox_center = ((float(rr.min()) + float(rr.max())) / 2, (float(cc.min()) + float(cc.max())) / 2)

    centers = [centroid, bbox_center]

    # Small local search around bbox center and centroid to better match 06c crop logic.
    for base in [centroid, bbox_center]:
        br, bc = base
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                centers.append((br + dr, bc + dc))

    return centers


def find_best_crop_center(full_comp: np.ndarray, comp_patch: np.ndarray, crop_size: int):
    """
    Find a crop center whose component crop best matches X[channel 40].
    This makes target crop aligned with the stored predicted component patch.
    """
    comp_patch_bool = comp_patch > 0.5

    best_center = None
    best_mismatch = float("inf")

    for center_r, center_c in candidate_centers(full_comp):
        crop = crop_2d(full_comp.astype(np.uint8), center_r, center_c, crop_size, fill_value=0)
        crop_bool = crop > 0
        mismatch = np.mean(crop_bool != comp_patch_bool)

        if mismatch < best_mismatch:
            best_mismatch = float(mismatch)
            best_center = (center_r, center_c)

    return best_center, best_mismatch


def make_overlay(pred_comp: np.ndarray, target: np.ndarray):
    """
    0 background
    1 TP
    2 FP
    3 FN
    """
    p = pred_comp > 0.5
    t = target > 0.5

    overlay = np.zeros_like(pred_comp, dtype=np.uint8)
    overlay[p & t] = 1
    overlay[p & (~t)] = 2
    overlay[(~p) & t] = 3
    return overlay


def plot_event(
    x_patch: np.ndarray,
    target_crop: np.ndarray | None,
    row: pd.Series,
    target_date,
    out_file: Path,
    crop_mismatch: float | None = None,
):
    """Plot one candidate event patch."""
    ssta = x_patch[9]
    mhw_hist = x_patch[19]
    exceed90 = x_patch[29]
    gap = x_patch[39]
    pred_comp = x_patch[40]

    if target_crop is None:
        target_crop = np.zeros_like(pred_comp)

    overlay = make_overlay(pred_comp, target_crop)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))

    # Continuous fields
    im0 = axes[0, 0].imshow(ssta, origin="lower", cmap="coolwarm")
    axes[0, 0].set_title("Recent SSTA")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(gap, origin="lower", cmap="coolwarm")
    axes[0, 1].set_title("Recent threshold_gap")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Binary maps
    bin_cmap = ListedColormap(["#2b2b2b", "#ffd84d"])

    axes[0, 2].imshow(mhw_hist > 0.5, origin="lower", cmap=bin_cmap, vmin=0, vmax=1)
    axes[0, 2].set_title("Recent historical MHW")

    axes[0, 3].imshow(exceed90 > 0.5, origin="lower", cmap=bin_cmap, vmin=0, vmax=1)
    axes[0, 3].set_title("Recent exceed90")

    axes[1, 0].imshow(pred_comp > 0.5, origin="lower", cmap=bin_cmap, vmin=0, vmax=1)
    axes[1, 0].set_title("Predicted component")

    axes[1, 1].imshow(target_crop > 0.5, origin="lower", cmap=bin_cmap, vmin=0, vmax=1)
    axes[1, 1].set_title("Target MHW crop")

    overlay_cmap = ListedColormap([
        "#2b2b2b",  # background
        "#4daf4a",  # TP
        "#e41a1c",  # FP
        "#377eb8",  # FN
    ])
    axes[1, 2].imshow(overlay, origin="lower", cmap=overlay_cmap, vmin=0, vmax=3)
    axes[1, 2].set_title("Overlay: TP green / FP red / FN blue")

    axes[1, 3].axis("off")
    text = (
        f"split event example\n"
        f"target_date: {target_date}\n"
        f"label y_valid: {int(row['y_valid'])}\n"
        f"sample_index: {int(row['sample_index'])}\n"
        f"component_id: {int(row['component_id'])}\n"
        f"area_px: {row['area_px']:.0f}\n"
        f"best_iou: {row['best_iou']:.4f}\n"
        f"overlap_ratio: {row['overlap_ratio']:.4f}\n"
    )
    if crop_mismatch is not None:
        text += f"crop_match_error: {crop_mismatch:.4f}\n"

    axes[1, 3].text(0.02, 0.95, text, va="top", ha="left", fontsize=11)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    label_name = "VALID" if int(row["y_valid"]) == 1 else "INVALID"
    fig.suptitle(
        f"{label_name} candidate MHW event | "
        f"date={target_date} | "
        f"IoU={row['best_iou']:.3f} | "
        f"overlap={row['overlap_ratio']:.3f}",
        fontsize=14,
    )

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def select_examples(df: pd.DataFrame, num_valid: int, num_invalid: int, num_small_invalid: int):
    """Select representative valid / invalid examples."""
    valid = (
        df[(df["y_valid"] == 1) & (df["area_px"] >= 300)]
        .sort_values(["best_iou", "overlap_ratio", "area_px"], ascending=False)
        .head(num_valid)
    )

    large_invalid = (
        df[(df["y_valid"] == 0) & (df["area_px"] >= 300)]
        .sort_values(["area_px", "best_iou"], ascending=[False, True])
        .head(num_invalid)
    )

    small_invalid = (
        df[(df["y_valid"] == 0) & (df["area_px"] <= 20)]
        .sort_values(["best_iou", "overlap_ratio", "area_px"], ascending=True)
        .head(num_small_invalid)
    )

    out = pd.concat(
        [
            valid.assign(example_group="large_high_iou_valid"),
            large_invalid.assign(example_group="large_false_positive_invalid"),
            small_invalid.assign(example_group="small_noise_invalid"),
        ],
        axis=0,
    )

    return out.reset_index().rename(columns={"index": "event_index"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_valid", type=int, default=8)
    parser.add_argument("--num_invalid", type=int, default=8)
    parser.add_argument("--num_small_invalid", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument(
        "--out_dir",
        default="outputs/19_event_positive_negative_visualization",
    )
    args = parser.parse_args()

    root = Path(".")
    event_dir = root / "outputs/06c_neurrl_event_dataset_from_multichannel_h10_l5"
    data_dir = root / "outputs/03b_forecast_dataset_multichannel_h10_l5"
    pred_dir = root / "outputs/04c_unet_multichannel_h10_l5"
    out_dir = root / args.out_dir

    out_valid = out_dir / args.split / "valid"
    out_invalid = out_dir / args.split / "invalid"
    out_valid.mkdir(parents=True, exist_ok=True)
    out_invalid.mkdir(parents=True, exist_ok=True)

    print("[LOAD]", event_dir / f"figure_event_{args.split}.npz")
    z = np.load(event_dir / f"figure_event_{args.split}.npz", allow_pickle=True)
    X = z["X"]
    y = z["y_valid"]
    meta = z["meta"]
    cols = [str(c) for c in z["meta_columns"]]

    target = np.load(data_dir / f"y_{args.split}.npy", mmap_mode="r")
    dates = np.load(data_dir / f"target_dates_{args.split}.npy", allow_pickle=True)
    pred_mask = np.load(pred_dir / f"pred_mask_{args.split}.npy", mmap_mode="r")

    df = pd.DataFrame(meta, columns=cols)
    df["y_valid"] = y.astype(int)
    df["sample_index"] = df["sample_index"].astype(int)
    df["component_id"] = df["component_id"].astype(int)
    df["target_date"] = [dates[i] for i in df["sample_index"]]

    selected = select_examples(
        df,
        num_valid=args.num_valid,
        num_invalid=args.num_invalid,
        num_small_invalid=args.num_small_invalid,
    )

    records = []

    for _, row in selected.iterrows():
        event_index = int(row["event_index"])
        sample_index = int(row["sample_index"])
        component_id = int(row["component_id"])
        expected_area = float(row["area_px"])
        target_date = row["target_date"]

        x_patch = X[event_index]
        pred_comp_patch = x_patch[40]

        full_comp = get_component_mask(
            pred_mask[sample_index],
            component_id=component_id,
            expected_area=expected_area,
        )

        crop_mismatch = None
        target_crop = None

        if full_comp is not None:
            center, crop_mismatch = find_best_crop_center(
                full_comp,
                pred_comp_patch,
                crop_size=args.crop_size,
            )
            target_crop = crop_2d(
                target[sample_index].astype(np.uint8),
                center[0],
                center[1],
                args.crop_size,
                fill_value=0,
            )

        label_dir = out_valid if int(row["y_valid"]) == 1 else out_invalid
        fname = (
            f"{row['example_group']}"
            f"_event{event_index}"
            f"_sample{sample_index}"
            f"_comp{component_id}"
            f"_date{str(target_date)}"
            f"_iou{row['best_iou']:.3f}"
            f"_ov{row['overlap_ratio']:.3f}.png"
        )
        out_file = label_dir / fname

        plot_event(
            x_patch=x_patch,
            target_crop=target_crop,
            row=row,
            target_date=target_date,
            out_file=out_file,
            crop_mismatch=crop_mismatch,
        )

        rec = row.to_dict()
        rec["figure_path"] = str(out_file)
        rec["crop_match_error"] = crop_mismatch
        records.append(rec)

        print("[SAVE]", out_file)

    summary = pd.DataFrame(records)
    summary_file = out_dir / args.split / "selected_event_examples.csv"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_file, index=False)
    print("[SAVE]", summary_file)
    print("[DONE]", out_dir)


if __name__ == "__main__":
    main()
