# -*- coding: utf-8 -*-
"""
Build tabular event-level features for symbolic rule learning.

Input:
    outputs/06c_neurrl_event_dataset_from_multichannel_h10_l5/
        multi_event_train/val/test.npz

Outputs:
    outputs/20_event_rule_learning/
        event_rule_table_train.csv
        event_rule_table_val.csv
        event_rule_table_test.csv
"""

import argparse
import csv
from pathlib import Path
import re

import numpy as np

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def parse_splits(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def safe_name(name):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_")


def trend(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size <= 1:
        return 0.0
    x = np.arange(values.size, dtype=np.float32)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return 0.0
    return float((x * (values - values.mean())).sum() / denom)


def add_variable_stats(row, name, values):
    values = np.asarray(values, dtype=np.float32)
    row[f"{name}_mean"] = float(np.mean(values))
    row[f"{name}_max"] = float(np.max(values))
    row[f"{name}_min"] = float(np.min(values))
    row[f"{name}_std"] = float(np.std(values))
    row[f"{name}_trend"] = trend(values)
    row[f"{name}_last"] = float(values[-1])
    row[f"{name}_positive_days"] = int((values > 0.0).sum())
    row[f"{name}_active_days"] = int((values > 0.5).sum())


def build_split(event_dir, out_dir, split):
    in_file = event_dir / f"multi_event_{split}.npz"
    z = np.load(in_file, allow_pickle=True)
    X_seq = z["X_seq"].astype(np.float32)
    y = z["y"].astype(np.uint8)
    variable_names = [safe_name(x) for x in z["variable_names"]]
    meta = z["meta"].astype(np.float32) if "meta" in z else None

    rows = []
    for i in range(X_seq.shape[0]):
        row = {
            "split": split,
            "event_index": int(i),
            "y_valid": int(y[i]),
        }
        if meta is not None:
            row.update({
                "sample_index": int(meta[i, 0]),
                "component_id": int(meta[i, 1]),
                "area_px": float(meta[i, 2]),
                "best_iou": float(meta[i, 3]),
                "overlap_ratio": float(meta[i, 4]),
            })

        for v, name in enumerate(variable_names):
            add_variable_stats(row, name, X_seq[i, v])

        # Semantic shortcuts used by the rule learner.
        if "mean_threshold_gap_inside_candidate" in variable_names:
            k = variable_names.index("mean_threshold_gap_inside_candidate")
            row["threshold_gap_positive_days"] = int((X_seq[i, k] > 0.0).sum())
            row["threshold_gap_last_positive"] = int(X_seq[i, k, -1] > 0.0)
        if "max_threshold_gap_inside_candidate" in variable_names:
            k = variable_names.index("max_threshold_gap_inside_candidate")
            row["max_threshold_gap_positive_days"] = int((X_seq[i, k] > 0.0).sum())
        if "mhw_fraction_inside_candidate" in variable_names:
            k = variable_names.index("mhw_fraction_inside_candidate")
            row["historical_mhw_days"] = int((X_seq[i, k] > 0.5).sum())
            row["historical_mhw_any_days"] = int((X_seq[i, k] > 0.0).sum())
            row["historical_mhw_last"] = int(X_seq[i, k, -1] > 0.5)
        if "exceed90_fraction_inside_candidate" in variable_names:
            k = variable_names.index("exceed90_fraction_inside_candidate")
            row["historical_exceed90_days"] = int((X_seq[i, k] > 0.5).sum())
            row["historical_exceed90_any_days"] = int((X_seq[i, k] > 0.0).sum())
            row["historical_exceed90_last"] = int(X_seq[i, k, -1] > 0.5)

        rows.append(row)

    out_csv = out_dir / f"event_rule_table_{split}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[SAVE]", out_csv, "rows=", len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "06c_neurrl_event_dataset_from_multichannel_h10_l5"),
    )
    parser.add_argument("--out_dir", type=str, default=str(cfg.OUTPUT_DIR / "20_event_rule_learning"))
    parser.add_argument("--splits", type=str, default="train,val,test")
    args = parser.parse_args()

    event_dir = Path(args.event_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in parse_splits(args.splits):
        build_split(event_dir, out_dir, split)


if __name__ == "__main__":
    main()
