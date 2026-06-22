#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compact visualization for candidate MHW events removed by symbolic rules.

This keeps the original 24_visualize_removed_rule_events.py untouched and
creates presentation-friendly 2x3 figures.

Outputs:
    outputs/24b_removed_rule_event_compact_visualization/{split}/good_removed_invalid/
    outputs/24b_removed_rule_event_compact_visualization/{split}/bad_removed_valid/
    outputs/24b_removed_rule_event_compact_visualization/{split}/selected_removed_rule_events_compact.csv
    outputs/24b_removed_rule_event_compact_visualization/{split}/removed_rule_event_summary_compact.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


BINARY_CMAP = ListedColormap(["#f7f7f7", "#2166ac"])
BINARY_NORM = BoundaryNorm([-0.5, 0.5, 1.5], BINARY_CMAP.N)
OVERLAY_CMAP = ListedColormap([
    "#2b2b2b",  # background / TN
    "#2ca25f",  # TP
    "#de2d26",  # FP
    "#3182bd",  # FN
])
OVERLAY_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], OVERLAY_CMAP.N)


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
    from scipy import ndimage

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
        mismatch = float(np.mean((crop > 0) != comp_patch_bool))
        if mismatch < best_mismatch:
            best_mismatch = mismatch
            best_center = (center_r, center_c)

    return best_center, best_mismatch


def make_overlay(pred: np.ndarray, target: np.ndarray):
    p = pred > 0.5
    t = target > 0.5
    overlay = np.zeros(pred.shape, dtype=np.uint8)
    overlay[p & t] = 1
    overlay[p & (~t)] = 2
    overlay[(~p) & t] = 3
    return overlay


def plot_binary(ax, arr, title):
    ax.imshow(arr > 0.5, origin="lower", cmap=BINARY_CMAP, norm=BINARY_NORM)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_compact_event(
    out_file: Path,
    gap_patch: np.ndarray,
    target_crop: np.ndarray,
    original_crop: np.ndarray,
    removed_crop: np.ndarray,
    corrected_crop: np.ndarray,
    row: pd.Series,
    target_date,
    overlay_source: str,
):
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.2), constrained_layout=True)

    vmax = float(np.nanpercentile(np.abs(gap_patch), 98))
    vmax = max(vmax, 1e-6)
    im = axes[0, 0].imshow(gap_patch, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title("Recent threshold_gap", fontsize=11)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    cbar = fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.03)
    cbar.set_label("threshold_gap", fontsize=10)

    plot_binary(axes[0, 1], target_crop, "Target MHW")
    plot_binary(axes[0, 2], original_crop, "Original prediction")
    plot_binary(axes[1, 0], removed_crop, "Removed by rule")
    plot_binary(axes[1, 1], corrected_crop, "Corrected prediction")

    overlay_pred = original_crop if overlay_source == "original" else corrected_crop
    axes[1, 2].imshow(make_overlay(overlay_pred, target_crop), origin="lower", cmap=OVERLAY_CMAP, norm=OVERLAY_NORM)
    axes[1, 2].set_title("Overlay: TP / FP / FN", fontsize=11)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    status = "GOOD removal" if int(row["y_valid"]) == 0 else "BAD removal"
    title = (
        f"{status} | date={target_date} | area={float(row['area_px']):.0f} | "
        f"IoU={float(row['best_iou']):.3f} | overlap={float(row['overlap_ratio']):.3f} | "
        f"invalid_hits={int(row.get('invalid_rule_hits', 0))} | "
        f"valid_hits={int(row.get('valid_rule_hits', 0))}"
    )
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def select_removed_events(pred_df, top_invalid, top_valid):
    removed = pred_df[pred_df["remove_by_rule"].astype(int) == 1].copy()

    good = (
        removed[removed["y_valid"].astype(int) == 0]
        .sort_values(["area_px", "best_iou", "overlap_ratio"], ascending=[False, True, True])
        .head(top_invalid)
        .assign(removal_group="good_removed_invalid")
    )
    bad = (
        removed[removed["y_valid"].astype(int) == 1]
        .sort_values(["area_px", "overlap_ratio", "best_iou"], ascending=[False, False, False])
        .head(top_valid)
        .assign(removal_group="bad_removed_valid")
    )
    return removed, pd.concat([good, bad], axis=0).reset_index(drop=True)


def write_summary(out_csv, pred_df, removed, selected):
    rows = []
    for group_name, sub in [
        ("all_events", pred_df),
        ("all_removed", removed),
        ("good_removed_invalid", removed[removed["y_valid"].astype(int) == 0]),
        ("bad_removed_valid", removed[removed["y_valid"].astype(int) == 1]),
        ("selected_good_removed_invalid", selected[selected["removal_group"] == "good_removed_invalid"]),
        ("selected_bad_removed_valid", selected[selected["removal_group"] == "bad_removed_valid"]),
    ]:
        rows.append({
            "group": group_name,
            "n_events": int(len(sub)),
            "mean_area_px": float(sub["area_px"].mean()) if len(sub) else 0.0,
            "mean_best_iou": float(sub["best_iou"].mean()) if len(sub) else 0.0,
            "mean_overlap_ratio": float(sub["overlap_ratio"].mean()) if len(sub) else 0.0,
            "mean_invalid_rule_hits": float(sub["invalid_rule_hits"].mean()) if len(sub) and "invalid_rule_hits" in sub else 0.0,
            "mean_valid_rule_hits": float(sub["valid_rule_hits"].mean()) if len(sub) and "valid_rule_hits" in sub else 0.0,
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("[SAVE]", out_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--top_invalid", type=int, default=20)
    parser.add_argument("--top_valid", type=int, default=10)
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--overlay_source", type=str, default="corrected", choices=["corrected", "original"])
    parser.add_argument("--rule_dir", type=str, default=str(cfg.OUTPUT_DIR / "20_event_rule_learning"))
    parser.add_argument(
        "--event_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "06c_neurrl_event_dataset_from_multichannel_h10_l5"),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "03b_forecast_dataset_multichannel_h10_l5"),
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5"),
    )
    parser.add_argument(
        "--corrected_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "20_event_rule_learning" / "rule_verifier_correction"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "24b_removed_rule_event_compact_visualization"),
    )
    args = parser.parse_args()

    rule_dir = Path(args.rule_dir)
    event_dir = Path(args.event_dir)
    data_dir = Path(args.data_dir)
    pred_dir = Path(args.pred_dir)
    corrected_dir = Path(args.corrected_dir)
    out_root = Path(args.out_dir) / args.split
    good_dir = out_root / "good_removed_invalid"
    bad_dir = out_root / "bad_removed_valid"
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    corrected_path = corrected_dir / f"rule_corrected_mask_{args.split}.npy"
    if not corrected_path.exists():
        raise FileNotFoundError(
            f"Missing corrected mask: {corrected_path}\n"
            "Run first:\n"
            "python code/22_apply_event_rule_verifier.py --splits train,val,test"
        )

    pred_csv = rule_dir / f"event_rule_predictions_{args.split}.csv"
    pred_df = pd.read_csv(pred_csv)
    removed, selected = select_removed_events(pred_df, args.top_invalid, args.top_valid)

    z = np.load(event_dir / f"figure_event_{args.split}.npz", allow_pickle=True)
    X_event = z["X"]

    y_true = np.load(data_dir / f"y_{args.split}.npy", mmap_mode="r")
    dates = np.load(data_dir / f"target_dates_{args.split}.npy", allow_pickle=True)
    pred_mask = np.load(pred_dir / f"pred_mask_{args.split}.npy", mmap_mode="r")
    corrected_mask = np.load(corrected_path, mmap_mode="r")

    records = []
    for _, row in selected.iterrows():
        event_index = int(row["event_index"])
        sample_index = int(row["sample_index"])
        component_id = int(row["component_id"])
        expected_area = float(row["area_px"])
        target_date = str(dates[sample_index])

        x_patch = X_event[event_index]
        comp_patch = x_patch[40]
        full_comp = get_component_mask(
            pred_mask[sample_index],
            component_id=component_id,
            expected_area=expected_area,
        )
        if full_comp is None:
            print("[WARN] Component not found; falling back to event patch only:", event_index)
            center = (pred_mask.shape[1] / 2, pred_mask.shape[2] / 2)
            crop_mismatch = np.nan
            removed_crop = comp_patch > 0.5
        else:
            center, crop_mismatch = find_best_crop_center(full_comp, comp_patch, args.crop_size)
            removed_crop = crop_2d(full_comp.astype(np.uint8), center[0], center[1], args.crop_size, fill_value=0)

        target_crop = crop_2d(y_true[sample_index].astype(np.uint8), center[0], center[1], args.crop_size, fill_value=0)
        original_crop = crop_2d(pred_mask[sample_index].astype(np.uint8), center[0], center[1], args.crop_size, fill_value=0)
        corrected_crop = crop_2d(corrected_mask[sample_index].astype(np.uint8), center[0], center[1], args.crop_size, fill_value=0)

        group = row["removal_group"]
        save_dir = good_dir if group == "good_removed_invalid" else bad_dir
        prefix = "good" if group == "good_removed_invalid" else "bad"
        out_file = save_dir / (
            f"{prefix}_event{event_index:05d}_sample{sample_index:04d}_"
            f"comp{component_id}_date{target_date}_area{expected_area:.0f}.png"
        )

        plot_compact_event(
            out_file=out_file,
            gap_patch=np.array(x_patch[39], dtype=np.float32),
            target_crop=target_crop,
            original_crop=original_crop,
            removed_crop=removed_crop,
            corrected_crop=corrected_crop,
            row=row,
            target_date=target_date,
            overlay_source=args.overlay_source,
        )

        rec = row.to_dict()
        rec.update({
            "target_date": target_date,
            "figure_path": str(out_file),
            "crop_match_error": crop_mismatch,
        })
        records.append(rec)
        print("[SAVE]", out_file)

    selected_csv = out_root / "selected_removed_rule_events_compact.csv"
    pd.DataFrame(records).to_csv(selected_csv, index=False)
    print("[SAVE]", selected_csv)

    summary_csv = out_root / "removed_rule_event_summary_compact.csv"
    write_summary(summary_csv, pred_df, removed, selected)


if __name__ == "__main__":
    main()
