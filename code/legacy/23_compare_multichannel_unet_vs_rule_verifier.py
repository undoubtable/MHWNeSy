# -*- coding: utf-8 -*-
"""
Compare multichannel U-Net baseline against symbolic rule verifier correction.

Output:
    outputs/23_rule_verifier_comparison/final_comparison.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


METRICS = [
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


def one_row(df, split, path):
    sub = df[df["split"] == split]
    if len(sub) != 1:
        raise ValueError(f"{path}: expected one row for split={split}, found {len(sub)}")
    return sub.iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument(
        "--baseline_csv",
        type=str,
        default=str(cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5" / "eval_metrics.csv"),
    )
    parser.add_argument(
        "--rule_correction_csv",
        type=str,
        default=str(cfg.OUTPUT_DIR / "20_event_rule_learning" / "rule_verifier_correction" / "rule_correction_metrics.csv"),
    )
    parser.add_argument("--out_dir", type=str, default=str(cfg.OUTPUT_DIR / "23_rule_verifier_comparison"))
    args = parser.parse_args()

    baseline_path = Path(args.baseline_csv)
    correction_path = Path(args.rule_correction_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(baseline_path)
    corr = pd.read_csv(correction_path)

    rows = []
    for split in parse_splits(args.splits):
        b = one_row(base, split, baseline_path)
        c = one_row(corr, split, correction_path)

        row_base = {
            "split": split,
            "method": "Multichannel U-Net baseline",
            "removed_event_ratio": "",
        }
        row_rule = {
            "split": split,
            "method": "Multichannel U-Net + symbolic rule verifier",
            "removed_event_ratio": c.get("removed_event_ratio", ""),
        }
        row_delta = {
            "split": split,
            "method": "Delta rule - baseline",
            "removed_event_ratio": "",
        }
        for metric in METRICS:
            row_base[metric] = b.get(metric, "")
            row_rule[metric] = c.get(metric, "")
            row_delta[metric] = c.get(metric, 0.0) - b.get(metric, 0.0)

        rows.extend([row_base, row_rule, row_delta])

    out = pd.DataFrame(rows)
    out_csv = out_dir / "final_comparison.csv"
    out.to_csv(out_csv, index=False)

    print("[FINAL COMPARISON]")
    print(out.to_string(index=False))
    print("[SAVE]", out_csv)


if __name__ == "__main__":
    main()
