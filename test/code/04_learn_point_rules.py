#!/usr/bin/env python
"""Learn simple interpretable point-level rule baselines.

This is not the final NeurRL learner. It creates a transparent baseline table
that later region-level and intensity-level experiments can build on.
"""

from __future__ import annotations

import csv
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


cfg = SourceFileLoader("test_cfg", str(Path(__file__).with_name("00_test_config.py"))).load_module()


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = float(np.logical_and(y_true, y_pred).sum())
    fp = float(np.logical_and(~y_true, y_pred).sum())
    fn = float(np.logical_and(y_true, ~y_pred).sum())

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "support": int(y_pred.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_best_threshold(default: float = 0.5) -> tuple[float, str]:
    """Read the F1-best point threshold if the sweep CSV exists."""

    if not cfg.POINT_THRESHOLD_SWEEP_FILE.exists():
        return default, "default"

    best_threshold = default
    best_f1 = -1.0
    best_precision = -1.0
    with cfg.POINT_THRESHOLD_SWEEP_FILE.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            threshold = float(row["threshold"])
            f1 = float(row["f1"])
            precision = float(row["precision"])
            if (f1, precision, threshold) > (best_f1, best_precision, best_threshold):
                best_threshold = threshold
                best_f1 = f1
                best_precision = precision

    return best_threshold, str(cfg.POINT_THRESHOLD_SWEEP_FILE)


def main() -> None:
    cfg.ensure_dirs()
    if not cfg.POINT_DATASET_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.POINT_DATASET_FILE}")
    if not cfg.POINT_LSTM_PRED_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.POINT_LSTM_PRED_FILE}")

    data = np.load(cfg.POINT_DATASET_FILE, allow_pickle=True)
    pred = np.load(cfg.POINT_LSTM_PRED_FILE, allow_pickle=True)
    x_test = data["X_test"].astype(np.float32)
    y_true = pred["y_true"].astype(np.uint8)
    y_prob = pred["y_prob"].astype(np.float32)
    feature_names = [str(x) for x in data["feature_names"].tolist()]
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}

    required = {"ssta", "exceed90", "mhw"}
    missing = required.difference(feature_to_idx)
    if missing:
        raise SystemExit(f"[ERROR] Missing features in point dataset: {sorted(missing)}")

    ssta = x_test[:, :, feature_to_idx["ssta"]]
    exceed90 = x_test[:, :, feature_to_idx["exceed90"]]
    mhw = x_test[:, :, feature_to_idx["mhw"]]
    best_threshold, threshold_source = load_best_threshold()
    print(f"[THRESHOLD] best_threshold={best_threshold:.2f} source={threshold_source}")

    recent_mhw_ge_2 = (mhw > 0.5).sum(axis=1) >= 2
    recent_exceed90_ge_3 = (exceed90 > 0.5).sum(axis=1) >= 3
    latest_ssta_gt_0 = ssta[:, -1] > 0.0
    lstm_prob_gt_best = y_prob > best_threshold

    # Candidate point-level rules. These intentionally mirror simple physical
    # conditions and the learned neural score, so later experiments can compare
    # symbolic-only, neural-only, and hybrid rules.
    rules = [
        ("recent_mhw_days_ge_2", "recent_mhw_days >= 2", recent_mhw_ge_2),
        (
            "recent_exceed90_days_ge_3",
            "recent_exceed90_days >= 3",
            recent_exceed90_ge_3,
        ),
        ("latest_ssta_gt_0", "latest_ssta > 0", latest_ssta_gt_0),
        ("lstm_prob_gt_0_5", "lstm_prob > 0.5", y_prob > 0.5),
        (
            "lstm_prob_gt_best_threshold",
            f"lstm_prob > {best_threshold:.2f}",
            lstm_prob_gt_best,
        ),
        (
            "mhw_or_lstm",
            "recent_mhw_days >= 2 OR lstm_prob > 0.5",
            recent_mhw_ge_2 | (y_prob > 0.5),
        ),
        (
            "exceed90_and_lstm",
            "recent_exceed90_days >= 3 AND lstm_prob > 0.5",
            recent_exceed90_ge_3 & (y_prob > 0.5),
        ),
        (
            "mhw_and_lstm_best_threshold",
            f"recent_mhw_days >= 2 AND lstm_prob > {best_threshold:.2f}",
            recent_mhw_ge_2 & lstm_prob_gt_best,
        ),
        (
            "exceed90_and_lstm_best_threshold",
            f"recent_exceed90_days >= 3 AND lstm_prob > {best_threshold:.2f}",
            recent_exceed90_ge_3 & lstm_prob_gt_best,
        ),
        (
            "ssta_and_lstm_best_threshold",
            f"latest_ssta > 0 AND lstm_prob > {best_threshold:.2f}",
            latest_ssta_gt_0 & lstm_prob_gt_best,
        ),
    ]

    rows = []
    for name, expression, mask in rules:
        metrics = binary_metrics(y_true, mask.astype(np.uint8))
        row = {"rule_name": name, "rule": expression}
        row.update(metrics)
        rows.append(row)

    fieldnames = ["rule_name", "rule", "support", "tp", "fp", "fn", "precision", "recall", "f1"]
    with cfg.POINT_RULES_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[SAVED] {cfg.POINT_RULES_FILE}")
    top_rows = sorted(rows, key=lambda row: (row["f1"], row["precision"], row["support"]), reverse=True)[:10]
    print("[TOP 10 RULES BY F1]")
    for rank, row in enumerate(top_rows, start=1):
        print(
            f"{rank:02d}. {row['rule_name']}: support={row['support']} "
            f"precision={row['precision']:.6f} recall={row['recall']:.6f} f1={row['f1']:.6f}"
        )


if __name__ == "__main__":
    main()
