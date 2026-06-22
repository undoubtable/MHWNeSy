# -*- coding: utf-8 -*-
"""
Evaluate a lead-day persistence baseline for MHW mask forecasting.

For each forecast sample, use the strict Hobday MHW mask at
target_date - lead as the prediction for target_date.

Outputs:
    outputs/04_unet_baseline_h10_l5/12_baseline_diagnostics/
        persistence_metrics.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def parse_splits(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def update_counts(pred_mask, y, counts):
    pred = pred_mask.astype(bool)
    yy = y.astype(bool)

    counts["tp"] += int(np.logical_and(pred, yy).sum())
    counts["fp"] += int(np.logical_and(pred, ~yy).sum())
    counts["fn"] += int(np.logical_and(~pred, yy).sum())
    counts["tn"] += int(np.logical_and(~pred, ~yy).sum())


def counts_to_metrics(counts):
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    total = tp + fp + fn + tn

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    acc = (tp + tn) / (total + 1e-8)

    return {
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "pixel_f1": float(f1),
        "pixel_iou": float(iou),
        "pixel_acc": float(acc),
        "pred_pos_ratio": float((tp + fp) / (total + 1e-8)),
        "true_pos_ratio": float((tp + fn) / (total + 1e-8)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def build_time_index(ds):
    dates = pd.DatetimeIndex(ds["time"].values).normalize()
    return {d.date().isoformat(): i for i, d in enumerate(dates)}


def source_indices_for_targets(target_dates, lead, date_to_index):
    source_indices = []
    source_dates = []

    for target_date in target_dates:
        target_ts = pd.Timestamp(str(target_date))
        source_date = (target_ts - pd.Timedelta(days=lead)).date().isoformat()
        if source_date not in date_to_index:
            raise KeyError(f"source date not found in label file: {source_date}")
        source_dates.append(source_date)
        source_indices.append(date_to_index[source_date])

    return np.array(source_indices, dtype=np.int64), source_dates


def evaluate_split(split, data_dir, mhw, date_to_index, lead, chunk_size):
    y_path = data_dir / f"y_{split}.npy"
    dates_path = data_dir / f"target_dates_{split}.npy"

    y = np.load(y_path, mmap_mode="r")
    target_dates = np.load(dates_path)
    source_indices, source_dates = source_indices_for_targets(target_dates, lead, date_to_index)

    if len(source_indices) != y.shape[0]:
        raise ValueError(f"{split}: source date count does not match y samples")

    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for start in tqdm(range(0, y.shape[0], chunk_size), desc=f"Persistence {split}"):
        end = min(start + chunk_size, y.shape[0])
        pred_chunk = mhw[source_indices[start:end]].astype(np.uint8)
        y_chunk = np.array(y[start:end], dtype=np.uint8)

        if pred_chunk.shape != y_chunk.shape:
            raise ValueError(f"{split}: prediction shape {pred_chunk.shape} != target shape {y_chunk.shape}")

        update_counts(pred_chunk, y_chunk, counts)

    metrics = counts_to_metrics(counts)
    metrics.update({
        "split": split,
        "lead": int(lead),
        "n_samples": int(y.shape[0]),
        "first_target_date": str(target_dates[0]) if len(target_dates) else "",
        "last_target_date": str(target_dates[-1]) if len(target_dates) else "",
        "first_source_date": source_dates[0] if source_dates else "",
        "last_source_date": source_dates[-1] if source_dates else "",
    })
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_nc", type=str, default=str(cfg.LABEL_NC))
    parser.add_argument("--data_dir", type=str, default=str(cfg.FORECAST_DIR))
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(cfg.UNET_RUN_DIR / "12_baseline_diagnostics"),
    )
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--lead", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=256)
    args = parser.parse_args()

    label_nc = Path(args.label_nc)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[LABEL]", label_nc)
    print("[DATA]", data_dir)
    print("[OUT]", out_dir)
    print("[LEAD]", args.lead)

    ds = xr.open_dataset(label_nc)
    date_to_index = build_time_index(ds)
    mhw = ds["mhw"].astype("uint8").values
    ds.close()

    rows = []
    for split in parse_splits(args.splits):
        metrics = evaluate_split(
            split=split,
            data_dir=data_dir,
            mhw=mhw,
            date_to_index=date_to_index,
            lead=args.lead,
            chunk_size=args.chunk_size,
        )
        rows.append(metrics)
        print("[METRICS]", split, metrics)

    out_csv = out_dir / "persistence_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[SAVE]", out_csv)


if __name__ == "__main__":
    main()
