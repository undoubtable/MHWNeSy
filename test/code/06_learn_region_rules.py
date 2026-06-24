#!/usr/bin/env python
"""Learn simple threshold-based region-level symbolic rules.

This script intentionally excludes ``max_iou`` and ``overlap_ratio`` from rule
features because they are diagnostic fields used to create ``region_label``.
Using them would leak label information into the symbolic rules.
"""

from __future__ import annotations

import itertools
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd


cfg = SourceFileLoader("test_cfg", str(Path(__file__).with_name("00_test_config.py"))).load_module()


AREA_THRESHOLDS = [2, 3, 5, 8, 10, 20, 30, 50, 100, 200]
PROB_THRESHOLDS = [0.85, 0.88, 0.90, 0.92, 0.95]
RECENT_DAY_THRESHOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
SSTA_THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0]

SINGLE_FEATURE_THRESHOLDS = {
    "area": AREA_THRESHOLDS,
    "mean_lstm_prob": PROB_THRESHOLDS,
    "max_lstm_prob": PROB_THRESHOLDS,
    "mean_recent_mhw_days": RECENT_DAY_THRESHOLDS,
    "mean_recent_exceed90_days": RECENT_DAY_THRESHOLDS,
    "mean_latest_ssta": SSTA_THRESHOLDS,
    "mean_intensity_ssta": SSTA_THRESHOLDS,
}

LEAKAGE_COLUMNS = {"max_iou", "overlap_ratio"}


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Compute support, confusion counts, precision, recall, and F1."""

    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)
    tp = int(np.logical_and(y_true_bool, y_pred_bool).sum())
    fp = int(np.logical_and(~y_true_bool, y_pred_bool).sum())
    fn = int(np.logical_and(y_true_bool, ~y_pred_bool).sum())
    support = int(y_pred_bool.sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "support": support,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def add_rule(rows: list[dict[str, object]], y_true: np.ndarray, name: str, expression: str, mask: np.ndarray) -> None:
    metrics = binary_metrics(y_true, mask)
    row: dict[str, object] = {
        "rule_name": name,
        "rule": expression,
    }
    row.update(metrics)
    rows.append(row)


def load_region_dataset() -> pd.DataFrame:
    if not cfg.REGION_DATASET_FULL_GRID_FILE.exists():
        raise SystemExit(
            f"[MISSING] {cfg.REGION_DATASET_FULL_GRID_FILE}\n"
            "Run first: python test/code/05b_build_region_dataset_full_grid.py"
        )

    df = pd.read_csv(cfg.REGION_DATASET_FULL_GRID_FILE)
    required = {"region_label", *SINGLE_FEATURE_THRESHOLDS.keys()}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"[ERROR] Missing required columns in region dataset: {missing}")

    present_leakage = sorted(LEAKAGE_COLUMNS.intersection(df.columns))
    if present_leakage:
        print(f"[INFO] Excluding diagnostic label columns from rules: {present_leakage}")

    df["region_label"] = df["region_label"].astype(int)
    return df


def print_top(title: str, rows: list[dict[str, object]], sort_keys: tuple[str, ...], limit: int = 20) -> None:
    print(title)
    ranked = sorted(
        rows,
        key=lambda row: tuple(float(row[key]) for key in sort_keys),
        reverse=True,
    )
    for rank, row in enumerate(ranked[:limit], start=1):
        print(
            f"{rank:02d}. {row['rule_name']}: "
            f"support={row['support']} tp={row['tp']} fp={row['fp']} fn={row['fn']} "
            f"precision={float(row['precision']):.6f} "
            f"recall={float(row['recall']):.6f} "
            f"f1={float(row['f1']):.6f} | {row['rule']}"
        )


def main() -> None:
    cfg.ensure_dirs()
    df = load_region_dataset()
    y_true = df["region_label"].to_numpy(dtype=np.uint8)

    print(f"[LOAD] {cfg.REGION_DATASET_FULL_GRID_FILE}")
    print(
        f"[DATA] regions={len(df)} positives={int(y_true.sum())} "
        f"negatives={int(len(y_true) - y_true.sum())} "
        f"positive_ratio={float(y_true.mean()) if len(y_true) else 0.0:.6f}"
    )

    rows: list[dict[str, object]] = []

    for feature, thresholds in SINGLE_FEATURE_THRESHOLDS.items():
        values = df[feature].to_numpy(dtype=np.float32)
        for threshold in thresholds:
            add_rule(
                rows,
                y_true,
                f"{feature}_ge_{threshold:g}",
                f"{feature} >= {threshold:g}",
                values >= threshold,
            )

    area = df["area"].to_numpy(dtype=np.float32)
    combo_specs = [
        ("mean_lstm_prob", PROB_THRESHOLDS),
        ("max_lstm_prob", PROB_THRESHOLDS),
        ("mean_latest_ssta", SSTA_THRESHOLDS),
        ("mean_recent_mhw_days", RECENT_DAY_THRESHOLDS),
        ("mean_recent_exceed90_days", RECENT_DAY_THRESHOLDS),
    ]

    for area_threshold, (feature, thresholds) in itertools.product(AREA_THRESHOLDS, combo_specs):
        feature_values = df[feature].to_numpy(dtype=np.float32)
        for threshold in thresholds:
            add_rule(
                rows,
                y_true,
                f"area_ge_{area_threshold:g}_and_{feature}_ge_{threshold:g}",
                f"area >= {area_threshold:g} AND {feature} >= {threshold:g}",
                (area >= area_threshold) & (feature_values >= threshold),
            )

    out = pd.DataFrame(rows)
    ordered_cols = ["rule_name", "rule", "support", "tp", "fp", "fn", "precision", "recall", "f1"]
    out = out[ordered_cols].sort_values(
        ["f1", "precision", "support"],
        ascending=[False, False, False],
    )
    out.to_csv(cfg.REGION_RULES_FILE, index=False)
    print(f"[SAVED] {cfg.REGION_RULES_FILE} rules={len(out)}")

    row_dicts = out.to_dict("records")
    print_top("[TOP 20 RULES BY F1]", row_dicts, ("f1", "precision", "support"), limit=20)

    high_precision = [row for row in row_dicts if int(row["support"]) >= 100]
    print_top(
        "[TOP 20 RULES BY PRECISION WITH support >= 100]",
        high_precision,
        ("precision", "f1", "support"),
        limit=20,
    )


if __name__ == "__main__":
    main()
