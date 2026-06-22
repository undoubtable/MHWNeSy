# -*- coding: utf-8 -*-
"""
Compare forecast baselines:
    1. lead-day persistence
    2. original U-Net with validation-selected probability threshold
    3. persistence-aware multichannel U-Net with validation-selected threshold

Output:
    outputs/14_forecast_baseline_comparison/final_forecast_baseline_comparison.csv
"""

import argparse
import csv
from pathlib import Path

import pandas as pd

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


METRIC_COLUMNS = [
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pixel_iou",
    "pixel_acc",
    "pred_pos_ratio",
    "true_pos_ratio",
]


def parse_splits(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def require_columns(df, path, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")


def row_for_split(df, split, path):
    sub = df[df["split"] == split]
    if len(sub) != 1:
        raise ValueError(f"{path}: expected one row for split={split}, found {len(sub)}")
    return sub.iloc[0].to_dict()


def comparison_row(split, method, source_row, threshold=""):
    row = {
        "split": split,
        "method": method,
        "threshold": threshold,
    }
    for col in METRIC_COLUMNS:
        row[col] = source_row[col] if col in source_row else ""
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument(
        "--persistence_csv",
        type=str,
        default=str(cfg.UNET_RUN_DIR / "12_baseline_diagnostics" / "persistence_metrics.csv"),
    )
    parser.add_argument(
        "--unet_selected_csv",
        type=str,
        default=str(cfg.UNET_RUN_DIR / "12_baseline_diagnostics" / "unet_best_threshold_comparison.csv"),
    )
    parser.add_argument(
        "--multichannel_csv",
        type=str,
        default=str(cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5" / "eval_metrics.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "14_forecast_baseline_comparison"),
    )
    args = parser.parse_args()

    splits = parse_splits(args.splits)
    persistence_path = Path(args.persistence_csv)
    unet_path = Path(args.unet_selected_csv)
    multichannel_path = Path(args.multichannel_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    persistence = pd.read_csv(persistence_path)
    unet = pd.read_csv(unet_path)
    multichannel = pd.read_csv(multichannel_path)

    require_columns(persistence, persistence_path, ["split"] + METRIC_COLUMNS)
    require_columns(unet, unet_path, ["split", "method", "threshold"] + METRIC_COLUMNS)
    require_columns(multichannel, multichannel_path, ["split", "threshold"] + METRIC_COLUMNS)

    rows = []
    for split in splits:
        p = row_for_split(persistence, split, persistence_path)

        u_sub = unet[
            (unet["split"] == split)
            & (unet["method"] == "U-Net selected threshold")
        ]
        if len(u_sub) != 1:
            raise ValueError(f"{unet_path}: expected one selected U-Net row for split={split}, found {len(u_sub)}")
        u = u_sub.iloc[0].to_dict()

        m = row_for_split(multichannel, split, multichannel_path)

        rows.append(comparison_row(split, "Persistence baseline", p, threshold=""))
        rows.append(comparison_row(split, "U-Net selected threshold", u, threshold=u["threshold"]))
        rows.append(comparison_row(split, "Multichannel U-Net selected threshold", m, threshold=m["threshold"]))

    out_csv = out_dir / "final_forecast_baseline_comparison.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[FINAL COMPARISON]")
    print(pd.DataFrame(rows).to_string(index=False))
    print("[SAVE]", out_csv)


if __name__ == "__main__":
    main()
