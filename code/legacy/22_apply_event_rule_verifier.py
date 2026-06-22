# -*- coding: utf-8 -*-
"""
Apply learned symbolic invalid-event rules to multichannel U-Net masks.

Outputs:
    outputs/20_event_rule_learning/rule_verifier_correction/
        rule_corrected_mask_{split}.npy
        rule_correction_metrics.csv
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def parse_splits(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def compute_metrics(pred_mask, y):
    pred = pred_mask.astype(bool)
    yy = y.astype(bool)
    tp = np.logical_and(pred, yy).sum()
    fp = np.logical_and(pred, ~yy).sum()
    fn = np.logical_and(~pred, yy).sum()
    tn = np.logical_and(~pred, ~yy).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "pixel_f1": float(f1),
        "pixel_iou": float(iou),
        "pixel_acc": float(acc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "pred_pos_ratio": float(pred.mean()),
        "true_pos_ratio": float(yy.mean()),
    }


def build_remove_dict(pred_df):
    remove = defaultdict(list)
    removed = pred_df[pred_df["remove_by_rule"].astype(int) == 1]
    for _, row in removed.iterrows():
        remove[int(row["sample_index"])].append(int(row["component_id"]))
    return remove, int(len(removed))


def correct_split(split, data_dir, pred_dir, rule_dir, out_dir):
    y_path = data_dir / f"y_{split}.npy"
    pred_path = pred_dir / f"pred_mask_{split}.npy"
    pred_csv = rule_dir / f"event_rule_predictions_{split}.csv"

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Missing {pred_path}. Run 06c_make_figure_event_dataset_multichannel.py first."
        )

    y = np.load(y_path, mmap_mode="r")
    pred = np.load(pred_path, mmap_mode="r")
    rule_pred = pd.read_csv(pred_csv)
    remove_dict, removed_events = build_remove_dict(rule_pred)

    N, H, W = pred.shape
    out_mask = out_dir / f"rule_corrected_mask_{split}.npy"
    corrected = np.lib.format.open_memmap(out_mask, mode="w+", dtype=np.uint8, shape=(N, H, W))

    for i in tqdm(range(N), desc=f"Rule correct {split}"):
        m = np.array(pred[i], dtype=np.uint8)
        if i in remove_dict:
            lab, _ = ndimage.label(m.astype(bool))
            for cid in remove_dict[i]:
                m[lab == cid] = 0
        corrected[i] = m

    corrected.flush()
    corrected_eval = np.load(out_mask, mmap_mode="r")
    metrics = compute_metrics(corrected_eval, y)
    n_events = int(len(rule_pred))
    metrics.update({
        "split": split,
        "method": "Multichannel U-Net + symbolic rule verifier",
        "n_samples": int(N),
        "n_events": n_events,
        "removed_events": int(removed_events),
        "kept_events": int(n_events - removed_events),
        "removed_event_ratio": float(removed_events / (n_events + 1e-8)),
        "mask_file": str(out_mask),
    })
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=str, default="train,val,test")
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
    parser.add_argument("--rule_dir", type=str, default=str(cfg.OUTPUT_DIR / "20_event_rule_learning"))
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "20_event_rule_learning" / "rule_verifier_correction"),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pred_dir = Path(args.pred_dir)
    rule_dir = Path(args.rule_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in parse_splits(args.splits):
        row = correct_split(split, data_dir, pred_dir, rule_dir, out_dir)
        rows.append(row)
        print("[METRICS]", split, row)

    out_csv = out_dir / "rule_correction_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("[SAVE]", out_csv)


if __name__ == "__main__":
    main()
