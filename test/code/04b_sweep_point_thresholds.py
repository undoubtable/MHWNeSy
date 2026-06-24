#!/usr/bin/env python
"""Sweep point-wise LSTM probability thresholds on the test split."""

from __future__ import annotations

import csv
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


cfg = SourceFileLoader("test_cfg", str(Path(__file__).with_name("00_test_config.py"))).load_module()


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute basic binary metrics without external dependencies."""

    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = float(np.logical_and(y_true, y_pred).sum())
    fp = float(np.logical_and(~y_true, y_pred).sum())
    fn = float(np.logical_and(y_true, ~y_pred).sum())
    tn = float(np.logical_and(~y_true, ~y_pred).sum())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    cfg.ensure_dirs()
    if not cfg.POINT_LSTM_PRED_FILE.exists():
        raise SystemExit(
            f"[MISSING] {cfg.POINT_LSTM_PRED_FILE}\n"
            "Run first: python test/code/03_eval_point_lstm.py"
        )

    pred = np.load(cfg.POINT_LSTM_PRED_FILE, allow_pickle=True)
    y_true = pred["y_true"].astype(np.uint8)
    y_prob = pred["y_prob"].astype(np.float32)

    rows = []
    for step in range(5, 100, 5):
        threshold = step / 100.0
        y_pred = (y_prob >= threshold).astype(np.uint8)
        row = {"threshold": threshold}
        row.update(binary_metrics(y_true, y_pred))
        rows.append(row)

    with cfg.POINT_THRESHOLD_SWEEP_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threshold", "accuracy", "precision", "recall", "f1"],
        )
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    print(f"[SAVED] {cfg.POINT_THRESHOLD_SWEEP_FILE}")
    print(
        "[BEST F1] "
        f"threshold={best['threshold']:.2f} "
        f"accuracy={best['accuracy']:.6f} "
        f"precision={best['precision']:.6f} "
        f"recall={best['recall']:.6f} "
        f"f1={best['f1']:.6f}"
    )


if __name__ == "__main__":
    main()
