# -*- coding: utf-8 -*-
"""
Compute extended pixel-level forecast metrics.

Outputs:
    outputs/17_extended_metrics/extended_forecast_metrics.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def parse_splits(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def load_threshold(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(obj["selected_threshold"])


def empty_counts():
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0}


def update_counts(counts, pred, true):
    pred = pred.astype(bool)
    true = true.astype(bool)
    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    tn = int(np.logical_and(~pred, ~true).sum())
    counts["tp"] += tp
    counts["fp"] += fp
    counts["fn"] += fn
    counts["tn"] += tn
    counts["total"] += tp + fp + fn + tn


def metrics_from_counts(counts):
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    total = counts["total"]

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    acc = (tp + tn) / (total + 1e-8)

    tpr = recall
    tnr = tn / (tn + fp + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    fnr = fn / (fn + tp + 1e-8)
    mcc_num = tp * tn - fp * fn
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-8
    mcc = mcc_num / mcc_den
    csi = iou

    hits_random = ((tp + fp) * (tp + fn)) / (total + 1e-8)
    ets = (tp - hits_random) / (tp + fp + fn - hits_random + 1e-8)
    balanced_accuracy = 0.5 * (tpr + tnr)

    return {
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "pixel_f1": float(f1),
        "pixel_iou": float(iou),
        "pixel_acc": float(acc),
        "TPR": float(tpr),
        "TNR": float(tnr),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "MCC": float(mcc),
        "CSI": float(csi),
        "ETS": float(ets),
        "balanced_accuracy": float(balanced_accuracy),
        "pred_pos_ratio": float((tp + fp) / (total + 1e-8)),
        "true_pos_ratio": float((tp + fn) / (total + 1e-8)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def evaluate_binary_method(name, split, y, pred, threshold, chunk_size):
    counts = empty_counts()
    for start in tqdm(range(0, y.shape[0], chunk_size), desc=f"{name} {split}"):
        end = min(start + chunk_size, y.shape[0])
        update_counts(counts, np.array(pred[start:end]), np.array(y[start:end]))

    row = metrics_from_counts(counts)
    row.update({
        "split": split,
        "method": name,
        "threshold": threshold,
        "brier_score": "",
        "n_samples": int(y.shape[0]),
    })
    return row


def evaluate_probability_method(name, split, y, prob, threshold, chunk_size):
    counts = empty_counts()
    brier_sum = 0.0
    n_pix = 0

    for start in tqdm(range(0, y.shape[0], chunk_size), desc=f"{name} {split}"):
        end = min(start + chunk_size, y.shape[0])
        yy = np.array(y[start:end], dtype=np.float32)
        pp = np.array(prob[start:end], dtype=np.float32)
        pred = pp >= threshold

        update_counts(counts, pred, yy.astype(bool))
        brier_sum += float(np.square(pp - yy).sum())
        n_pix += int(np.prod(yy.shape))

    row = metrics_from_counts(counts)
    row.update({
        "split": split,
        "method": name,
        "threshold": float(threshold),
        "brier_score": float(brier_sum / (n_pix + 1e-8)),
        "n_samples": int(y.shape[0]),
    })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=256)
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
    parser.add_argument("--out_dir", type=str, default=str(cfg.OUTPUT_DIR / "17_extended_metrics"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unet_thr = load_threshold(args.unet_threshold_json)
    mc_thr = load_threshold(args.mc_threshold_json)

    rows = []
    for split in parse_splits(args.splits):
        y = np.load(Path(args.orig_data_dir) / f"y_{split}.npy", mmap_mode="r")
        X_mc = np.load(Path(args.mc_data_dir) / f"X_{split}.npy", mmap_mode="r")
        unet_prob = np.load(Path(args.unet_dir) / f"pred_prob_{split}.npy", mmap_mode="r")
        mc_prob = np.load(Path(args.mc_unet_dir) / f"pred_prob_{split}.npy", mmap_mode="r")

        if y.shape != unet_prob.shape or y.shape != mc_prob.shape:
            raise ValueError(f"{split}: shape mismatch y={y.shape}, unet={unet_prob.shape}, multichannel={mc_prob.shape}")
        if X_mc.shape[0] != y.shape[0]:
            raise ValueError(f"{split}: sample count mismatch X={X_mc.shape}, y={y.shape}")

        persistence = X_mc[:, args.history + args.history - 1] >= 0.5

        rows.append(
            evaluate_binary_method(
                name="Persistence",
                split=split,
                y=y,
                pred=persistence,
                threshold="",
                chunk_size=args.chunk_size,
            )
        )
        rows.append(
            evaluate_probability_method(
                name="SSTA-only U-Net",
                split=split,
                y=y,
                prob=unet_prob,
                threshold=unet_thr,
                chunk_size=args.chunk_size,
            )
        )
        rows.append(
            evaluate_probability_method(
                name="Multichannel U-Net",
                split=split,
                y=y,
                prob=mc_prob,
                threshold=mc_thr,
                chunk_size=args.chunk_size,
            )
        )

    out_csv = out_dir / "extended_forecast_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[SAVE]", out_csv)


if __name__ == "__main__":
    main()
