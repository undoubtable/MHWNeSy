# -*- coding: utf-8 -*-
"""
Plot test-period daily MHW area ratio for forecast baselines.

Outputs:
    outputs/15_forecast_visualization/
        area_timeseries_test.png
        area_timeseries_test.csv
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def load_threshold(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(obj["selected_threshold"])


def area_ratio(mask):
    return mask.reshape(mask.shape[0], -1).mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, default=10)
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
    y = np.load(Path(args.orig_data_dir) / f"y_{split}.npy", mmap_mode="r")
    X_mc = np.load(Path(args.mc_data_dir) / f"X_{split}.npy", mmap_mode="r")
    dates = np.load(Path(args.mc_data_dir) / f"target_dates_{split}.npy")
    unet_prob = np.load(Path(args.unet_dir) / f"pred_prob_{split}.npy", mmap_mode="r")
    mc_prob = np.load(Path(args.mc_unet_dir) / f"pred_prob_{split}.npy", mmap_mode="r")

    if y.shape != unet_prob.shape or y.shape != mc_prob.shape:
        raise ValueError(f"Shape mismatch: y={y.shape}, unet={unet_prob.shape}, multichannel={mc_prob.shape}")
    if X_mc.shape[0] != y.shape[0] or len(dates) != y.shape[0]:
        raise ValueError(f"Sample count mismatch: X={X_mc.shape}, y={y.shape}, dates={len(dates)}")

    unet_thr = load_threshold(args.unet_threshold_json)
    mc_thr = load_threshold(args.mc_threshold_json)

    history = args.history
    persistence = X_mc[:, history + history - 1] >= 0.5
    unet_mask = unet_prob >= unet_thr
    mc_mask = mc_prob >= mc_thr

    rows = []
    observed = area_ratio(y.astype(bool))
    persistence_area = area_ratio(persistence)
    unet_area = area_ratio(unet_mask)
    mc_area = area_ratio(mc_mask)

    for i, date in enumerate(dates):
        rows.append({
            "target_date": str(date),
            "observed_area_ratio": float(observed[i]),
            "persistence_area_ratio": float(persistence_area[i]),
            "ssta_unet_area_ratio": float(unet_area[i]),
            "multichannel_unet_area_ratio": float(mc_area[i]),
        })

    out_csv = out_dir / "area_timeseries_test.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(x, observed, label="Observed", linewidth=1.5, color="black")
    ax.plot(x, persistence_area, label="Persistence", linewidth=1.0, alpha=0.85)
    ax.plot(x, unet_area, label="SSTA-only U-Net", linewidth=1.0, alpha=0.85)
    ax.plot(x, mc_area, label="Multichannel U-Net", linewidth=1.0, alpha=0.9)

    tick_idx = np.linspace(0, len(rows) - 1, num=min(10, len(rows)), dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(dates[i]) for i in tick_idx], rotation=30, ha="right")
    ax.set_ylabel("MHW area ratio")
    ax.set_xlabel("Target date")
    ax.set_title("Test-period MHW area ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2)

    out_png = out_dir / "area_timeseries_test.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print("[SAVE]", out_csv)
    print("[SAVE]", out_png)


if __name__ == "__main__":
    main()
