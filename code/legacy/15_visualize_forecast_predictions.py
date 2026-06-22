# -*- coding: utf-8 -*-
"""
Visualize forecast prediction examples on the test split.

Outputs:
    outputs/15_forecast_visualization/
        *.png
        selected_cases.csv
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from tqdm import tqdm

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def load_threshold(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(obj["selected_threshold"])


def binary_metrics_one(pred, true):
    pred = pred.astype(bool)
    true = true.astype(bool)
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return float(f1), float(iou)


def compute_case_scores(y_true, persistence, mc_mask):
    rows = []
    for i in tqdm(range(y_true.shape[0]), desc="Score cases"):
        p_f1, p_iou = binary_metrics_one(persistence[i], y_true[i])
        m_f1, m_iou = binary_metrics_one(mc_mask[i], y_true[i])
        rows.append({
            "sample_index": i,
            "persistence_f1": p_f1,
            "persistence_iou": p_iou,
            "multichannel_f1": m_f1,
            "multichannel_iou": m_iou,
            "iou_delta_mc_minus_persistence": m_iou - p_iou,
            "f1_delta_mc_minus_persistence": m_f1 - p_f1,
        })
    return rows


def unique_indices(*groups):
    seen = set()
    out = []
    for group in groups:
        kept = []
        for idx in group:
            idx = int(idx)
            if idx in seen:
                continue
            seen.add(idx)
            kept.append(idx)
        out.append(kept)
    return out


def select_cases(score_rows, top_k, random_k, seed):
    order_mc = sorted(score_rows, key=lambda r: r["iou_delta_mc_minus_persistence"], reverse=True)
    order_p = sorted(score_rows, key=lambda r: r["iou_delta_mc_minus_persistence"])

    top_mc = [r["sample_index"] for r in order_mc[:top_k]]
    top_p = [r["sample_index"] for r in order_p[:top_k]]

    rng = np.random.default_rng(seed)
    random_idx = rng.choice(
        np.arange(len(score_rows)),
        size=min(random_k, len(score_rows)),
        replace=False,
    ).tolist()

    top_mc, top_p, random_idx = unique_indices(top_mc, top_p, random_idx)
    return {
        "mc_better_than_persistence": top_mc,
        "persistence_better_than_mc": top_p,
        "random": random_idx,
    }


def error_map(pred, true):
    pred = pred.astype(bool)
    true = true.astype(bool)
    out = np.zeros(true.shape, dtype=np.uint8)
    out[np.logical_and(pred, true)] = 1
    out[np.logical_and(pred, ~true)] = 2
    out[np.logical_and(~pred, true)] = 3
    return out


def add_imshow(ax, arr, title, cmap="viridis", vmin=None, vmax=None, colorbar=False):
    im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return im


def plot_case(
    out_path, idx, date, context_name, context_arr, y_true, persistence,
    unet_mask, mc_prob, mc_mask, score_row, unet_thr, mc_thr,
):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes = axes.ravel()

    add_imshow(axes[0], context_arr, f"Recent {context_name}", cmap="coolwarm", colorbar=True)
    add_imshow(axes[1], y_true, "Ground truth", cmap="gray_r", vmin=0, vmax=1)
    add_imshow(axes[2], persistence, "Persistence", cmap="gray_r", vmin=0, vmax=1)
    add_imshow(axes[3], unet_mask, f"SSTA-only U-Net mask\nthr={unet_thr:.2f}", cmap="gray_r", vmin=0, vmax=1)
    add_imshow(axes[4], mc_prob, "Multichannel U-Net probability", cmap="magma", vmin=0, vmax=1, colorbar=True)
    add_imshow(axes[5], mc_mask, f"Multichannel U-Net mask\nthr={mc_thr:.2f}", cmap="gray_r", vmin=0, vmax=1)

    cmap = ListedColormap(["white", "#2ca25f", "#de2d26", "#3182bd"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im = axes[6].imshow(error_map(mc_mask, y_true), origin="lower", cmap=cmap, norm=norm)
    axes[6].set_title("Multichannel error map\nTP green / FP red / FN blue", fontsize=10)
    axes[6].set_xticks([])
    axes[6].set_yticks([])
    cbar = plt.colorbar(im, ax=axes[6], fraction=0.046, pad=0.03, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["TN", "TP", "FP", "FN"])

    axes[7].axis("off")
    axes[7].text(
        0.02,
        0.98,
        "\n".join([
            f"sample_index: {idx}",
            f"target_date: {date}",
            f"persistence IoU: {score_row['persistence_iou']:.4f}",
            f"multichannel IoU: {score_row['multichannel_iou']:.4f}",
            f"IoU delta: {score_row['iou_delta_mc_minus_persistence']:.4f}",
            f"persistence F1: {score_row['persistence_f1']:.4f}",
            f"multichannel F1: {score_row['multichannel_f1']:.4f}",
        ]),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle(f"Forecast comparison for {date}", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead", type=int, default=5)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--random_k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context", type=str, default="threshold_gap", choices=["ssta", "threshold_gap"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--orig_data_dir", type=str, default=str(cfg.FORECAST_DIR))
    parser.add_argument(
        "--mc_data_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "03b_forecast_dataset_multichannel_h10_l5"),
    )
    parser.add_argument("--unet_dir", type=str, default=str(cfg.UNET_RUN_DIR))
    parser.add_argument(
        "--mc_unet_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5"),
    )
    parser.add_argument(
        "--unet_threshold_json",
        type=str,
        default=str(cfg.UNET_RUN_DIR / "12_baseline_diagnostics" / "selected_unet_threshold.json"),
    )
    parser.add_argument(
        "--mc_threshold_json",
        type=str,
        default=str(cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5" / "selected_threshold.json"),
    )
    parser.add_argument("--out_dir", type=str, default=str(cfg.OUTPUT_DIR / "15_forecast_visualization"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = args.split
    y_orig = np.load(Path(args.orig_data_dir) / f"y_{split}.npy", mmap_mode="r")
    X_mc = np.load(Path(args.mc_data_dir) / f"X_{split}.npy", mmap_mode="r")
    y_mc = np.load(Path(args.mc_data_dir) / f"y_{split}.npy", mmap_mode="r")
    target_dates = np.load(Path(args.mc_data_dir) / f"target_dates_{split}.npy")
    unet_prob = np.load(Path(args.unet_dir) / f"pred_prob_{split}.npy", mmap_mode="r")
    mc_prob = np.load(Path(args.mc_unet_dir) / f"pred_prob_{split}.npy", mmap_mode="r")

    if y_orig.shape != y_mc.shape:
        raise ValueError(f"Original and multichannel y shape mismatch: {y_orig.shape} vs {y_mc.shape}")

    unet_thr = load_threshold(args.unet_threshold_json)
    mc_thr = load_threshold(args.mc_threshold_json)

    history = args.history
    persistence = X_mc[:, history + history - 1] >= 0.5
    y_true = y_mc.astype(bool)
    unet_mask = unet_prob >= unet_thr
    mc_mask = mc_prob >= mc_thr

    score_rows = compute_case_scores(y_true, persistence, mc_mask)
    by_idx = {r["sample_index"]: r for r in score_rows}
    selected = select_cases(score_rows, args.top_k, args.random_k, args.seed)

    selected_rows = []
    context_offset = 0 if args.context == "ssta" else history * 3
    context_ch = context_offset + history - 1

    for category, indices in selected.items():
        cat_dir = out_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        for rank, idx in enumerate(indices, start=1):
            date = str(target_dates[idx])
            score_row = by_idx[idx]
            out_png = cat_dir / f"{rank:02d}_sample{idx:04d}_{date}.png"

            plot_case(
                out_path=out_png,
                idx=idx,
                date=date,
                context_name=args.context,
                context_arr=np.array(X_mc[idx, context_ch], dtype=np.float32),
                y_true=np.array(y_true[idx], dtype=np.uint8),
                persistence=np.array(persistence[idx], dtype=np.uint8),
                unet_mask=np.array(unet_mask[idx], dtype=np.uint8),
                mc_prob=np.array(mc_prob[idx], dtype=np.float32),
                mc_mask=np.array(mc_mask[idx], dtype=np.uint8),
                score_row=score_row,
                unet_thr=unet_thr,
                mc_thr=mc_thr,
            )

            row = {
                "category": category,
                "rank": rank,
                "sample_index": idx,
                "target_date": date,
                "figure": str(out_png),
            }
            row.update(score_row)
            selected_rows.append(row)
            print("[SAVE]", out_png)

    out_csv = out_dir / "selected_cases.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(selected_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    print("[SAVE]", out_csv)
    print("[DONE]", out_dir)


if __name__ == "__main__":
    main()
