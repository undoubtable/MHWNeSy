#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize candidate MHW events removed by symbolic rule verifier.

This script visualizes events with remove_by_rule = 1 from:
  outputs/20_event_rule_learning/event_rule_predictions_{split}.csv

It separates:
  - good_removed_invalid: y_valid = 0, correctly removed false-positive events
  - bad_removed_valid: y_valid = 1, mistakenly removed valid events

Each figure shows:
  recent SSTA
  recent threshold_gap
  recent historical MHW
  recent exceed90
  predicted component
  target MHW crop
  TP / FP / FN overlay
  event metadata and rule hits
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
    pred_bool = pred_mask.astype(bool)

    structures = [
        np.ones((3, 3), dtype=int),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    ]

    best = None
    best_area_diff = float("inf")

    for structure in structures:
        lab, nlab = ndimage.label(pred_bool, structure=structure)

        if 1 <= component_id <= nlab:
            comp = lab == component_id
            area = int(comp.sum())
            if abs(area - expected_area) <= max(2, 0.02 * expected_area):
                return comp

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
    rr, cc = np.where(comp)
    if len(rr) == 0:
        return [(comp.shape[0] / 2, comp.shape[1] / 2)]

    centroid = (float(rr.mean()), float(cc.mean()))
    bbox_center = ((float(rr.min()) + float(rr.max())) / 2, (float(cc.min()) + float(cc.max())) / 2)

    centers = [centroid, bbox_center]
    for base in [centroid, bbox_center]:
        br, bc = base
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                centers.append((br + dr, bc + dc))

    return centers


def find_best_crop_center(full_comp: np.ndarray, comp_patch: np.ndarray, crop_size: int):
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
    p = pred_comp > 0.5
    t = target > 0.5

    overlay = np.zeros_like(pred_comp, dtype=np.uint8)
    overlay[p & t] = 1   # TP
    overlay[p & (~t)] = 2  # FP
    overlay[(~p) & t] = 3  # FN
    return overlay


def plot_removed_event(
    x_patch: np.ndarray,
    target_crop: np.ndarray,
    row: pd.Series,
    target_date,
    out_file: Path,
    crop_mismatch: float | None,
):
    ssta = x_patch[9]
    mhw_hist = x_patch[19]
    exceed90 = x_patch[29]
    gap = x_patch[39]
    pred_comp = x_patch[40]

    overlay = make_overlay(pred_comp, target_crop)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))

    im0 = axes[0, 0].imshow(ssta, origin="lower", cmap="coolwarm")
    axes[0, 0].set_title("Recent SSTA")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(gap, origin="lower", cmap="coolwarm")
    axes[0, 1].set_title("Recent threshold_gap")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    bin_cmap = ListedColormap(["#2b2b2b", "#ffd84d"])

    axes[0, 2].imshow(mhw_hist > 0.5, origin="lower", cmap=bin_cmap, vmin=0, vmax=1)
    axes[0, 2].set_title("Recent historical MHW")

    axes[0, 3].imshow(exceed90 > 0.5, origin="lower", cmap=bin_cmap, vmin=0, vmax=1)
    axes[0, 3].set_title("Recent exceed90")

    axes[1, 0].imshow(pred_comp > 0.5, origin="lower", cmap=bin_cmap, vmin=0, vmax=1)
    axes[1, 0].set_title("Removed predicted component")

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

    if int(row["y_valid"]) == 0:
        removal_type = "GOOD removal: true invalid"
    else:
        removal_type = "BAD removal: true valid"

    info = (
        f"{removal_type}\n"
        f"target_date: {target_date}\n"
        f"event_index: {int(row['event_index'])}\n"
        f"sample_index: {int(row['sample_index'])}\n"
        f"component_id: {int(row['component_id'])}\n"
        f"y_valid: {int(row['y_valid'])}\n"
        f"area_px: {row['area_px']:.0f}\n"
        f"best_iou: {row['best_iou']:.4f}\n"
        f"overlap_ratio: {row['overlap_ratio']:.4f}\n"
        f"invalid_rule_hits: {int(row.get('invalid_rule_hits', 0))}\n"
        f"valid_rule_hits: {int(row.get('valid_rule_hits', 0))}\n"
    )

    if crop_mismatch is not None:
        info += f"crop_match_error: {crop_mismatch:.4f}\n"

    axes[1, 3].text(0.02, 0.95, info, va="top", ha="left", fontsize=11)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    title = (
        f"Removed by symbolic rule | "
        f"{'true invalid' if int(row['y_valid']) == 0 else 'true valid'} | "
        f"date={target_date} | IoU={row['best_iou']:.3f} | "
        f"overlap={row['overlap_ratio']:.3f}"
    )
    fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--top_invalid", type=int, default=20)
    parser.add_argument("--top_valid", type=int, default=20)
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--out_dir", default="outputs/24_removed_rule_event_visualization")
    args = parser.parse_args()

    root = Path(".")
    rule_dir = root / "outputs/20_event_rule_learning"
    event_dir = root / "outputs/06c_neurrl_event_dataset_from_multichannel_h10_l5"
    data_dir = root / "outputs/03b_forecast_dataset_multichannel_h10_l5"
    pred_dir = root / "outputs/04c_unet_multichannel_h10_l5"
    out_dir = root / args.out_dir / args.split

    good_dir = out_dir / "good_removed_invalid"
    bad_dir = out_dir / "bad_removed_valid"
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    pred_csv = rule_dir / f"event_rule_predictions_{args.split}.csv"
    print("[LOAD]", pred_csv)
    pred_df = pd.read_csv(pred_csv)

    removed = pred_df[pred_df["remove_by_rule"] == 1].copy()
    print("[INFO] removed events:", len(removed))
    print("[INFO] y_valid counts:")
    print(removed["y_valid"].value_counts())

    removed_invalid = (
        removed[removed["y_valid"] == 0]
        .sort_values(["area_px", "best_iou"], ascending=[False, True])
        .head(args.top_invalid)
    )

    removed_valid = (
        removed[removed["y_valid"] == 1]
        .sort_values(["area_px", "best_iou"], ascending=[False, True])
        .head(args.top_valid)
    )

    selected = pd.concat(
        [
            removed_invalid.assign(removal_group="good_removed_invalid"),
            removed_valid.assign(removal_group="bad_removed_valid"),
        ],
        axis=0,
    ).reset_index(drop=True)

    z = np.load(event_dir / f"figure_event_{args.split}.npz", allow_pickle=True)
    X = z["X"]

    y_true = np.load(data_dir / f"y_{args.split}.npy", mmap_mode="r")
    dates = np.load(data_dir / f"target_dates_{args.split}.npy", allow_pickle=True)
    pred_mask = np.load(pred_dir / f"pred_mask_{args.split}.npy", mmap_mode="r")

    records = []

    for _, row in selected.iterrows():
        event_index = int(row["event_index"])
        sample_index = int(row["sample_index"])
        component_id = int(row["component_id"])
        expected_area = float(row["area_px"])
        target_date = dates[sample_index]

        x_patch = X[event_index]
        pred_comp_patch = x_patch[40]

        full_comp = get_component_mask(
            pred_mask[sample_index],
            component_id=component_id,
            expected_area=expected_area,
        )

        if full_comp is None:
            target_crop = np.zeros((args.crop_size, args.crop_size), dtype=np.uint8)
            crop_mismatch = None
        else:
            center, crop_mismatch = find_best_crop_center(
                full_comp,
                pred_comp_patch,
                crop_size=args.crop_size,
            )
            target_crop = crop_2d(
                y_true[sample_index].astype(np.uint8),
                center[0],
                center[1],
                args.crop_size,
                fill_value=0,
            )

        save_dir = good_dir if int(row["y_valid"]) == 0 else bad_dir

        fname = (
            f"{row['removal_group']}"
            f"_event{event_index}"
            f"_sample{sample_index}"
            f"_comp{component_id}"
            f"_date{str(target_date)}"
            f"_area{expected_area:.0f}"
            f"_iou{row['best_iou']:.3f}"
            f"_ov{row['overlap_ratio']:.3f}.png"
        )
        out_file = save_dir / fname

        plot_removed_event(
            x_patch=x_patch,
            target_crop=target_crop,
            row=row,
            target_date=target_date,
            out_file=out_file,
            crop_mismatch=crop_mismatch,
        )

        rec = row.to_dict()
        rec["target_date"] = target_date
        rec["figure_path"] = str(out_file)
        rec["crop_match_error"] = crop_mismatch
        records.append(rec)

        print("[SAVE]", out_file)

    summary = pd.DataFrame(records)
    summary_file = out_dir / "selected_removed_rule_events.csv"
    summary.to_csv(summary_file, index=False)
    print("[SAVE]", summary_file)

    count_file = out_dir / "removed_rule_event_summary.csv"
    summary_rows = []
    for name, sub in [
        ("all_removed", removed),
        ("good_removed_invalid", removed[removed["y_valid"] == 0]),
        ("bad_removed_valid", removed[removed["y_valid"] == 1]),
    ]:
        summary_rows.append({
            "group": name,
            "n_events": len(sub),
            "mean_area_px": sub["area_px"].mean() if len(sub) else 0,
            "median_area_px": sub["area_px"].median() if len(sub) else 0,
            "mean_best_iou": sub["best_iou"].mean() if len(sub) else 0,
            "mean_overlap_ratio": sub["overlap_ratio"].mean() if len(sub) else 0,
        })
    pd.DataFrame(summary_rows).to_csv(count_file, index=False)
    print("[SAVE]", count_file)

    print("[DONE]", out_dir)


if __name__ == "__main__":
    main()
