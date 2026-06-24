#!/usr/bin/env python
"""Learn conservative rules for removing false-positive MHW regions.

Here ``region_label == 0`` is the positive class for rule evaluation: a rule
fires when it proposes deleting a region. ``max_iou`` and ``overlap_ratio`` are
diagnostic label-construction fields and are never used as rule features.
"""

from __future__ import annotations

import itertools
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd


cfg = SourceFileLoader("test_cfg", str(Path(__file__).with_name("00_test_config.py"))).load_module()


AREA_THRESHOLDS = [2, 3, 5, 8, 10, 20, 30, 50, 100, 200]
PROB_THRESHOLDS = [0.86, 0.88, 0.90, 0.92, 0.95]
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


def removal_metrics(
    invalid_mask: np.ndarray,
    valid_mask: np.ndarray,
    remove_mask: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate a removal rule with invalid regions as the positive class."""

    remove_mask = remove_mask.astype(bool)
    correctly_removed = int(np.logical_and(remove_mask, invalid_mask).sum())
    wrongly_removed = int(np.logical_and(remove_mask, valid_mask).sum())
    missed_invalid = int(np.logical_and(~remove_mask, invalid_mask).sum())
    support = int(remove_mask.sum())

    total_invalid = int(invalid_mask.sum())
    total_valid = int(valid_mask.sum())

    removal_precision = correctly_removed / max(support, 1)
    removal_recall = correctly_removed / max(total_invalid, 1)
    valid_loss_ratio = wrongly_removed / max(total_valid, 1)
    removal_f1 = (
        2.0 * removal_precision * removal_recall
        / max(removal_precision + removal_recall, 1e-12)
    )

    return {
        "support": support,
        "correctly_removed": correctly_removed,
        "wrongly_removed": wrongly_removed,
        "missed_invalid": missed_invalid,
        "removal_precision": removal_precision,
        "removal_recall": removal_recall,
        "valid_loss_ratio": valid_loss_ratio,
        "removal_f1": removal_f1,
    }


def add_rule(
    rows: list[dict[str, object]],
    invalid_mask: np.ndarray,
    valid_mask: np.ndarray,
    name: str,
    expression: str,
    remove_mask: np.ndarray,
) -> None:
    row: dict[str, object] = {
        "rule_name": name,
        "rule": expression,
    }
    row.update(removal_metrics(invalid_mask, valid_mask, remove_mask))
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
        print(f"[INFO] Excluding diagnostic label columns from removal rules: {present_leakage}")

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
            f"support={row['support']} "
            f"correctly_removed={row['correctly_removed']} "
            f"wrongly_removed={row['wrongly_removed']} "
            f"precision={float(row['removal_precision']):.6f} "
            f"recall={float(row['removal_recall']):.6f} "
            f"valid_loss={float(row['valid_loss_ratio']):.6f} "
            f"f1={float(row['removal_f1']):.6f} | {row['rule']}"
        )


def main() -> None:
    cfg.ensure_dirs()
    df = load_region_dataset()

    valid_mask = df["region_label"].to_numpy(dtype=np.uint8).astype(bool)
    invalid_mask = ~valid_mask
    total_valid = int(valid_mask.sum())
    total_invalid = int(invalid_mask.sum())
    print(f"[LOAD] {cfg.REGION_DATASET_FULL_GRID_FILE}")
    print(
        f"[DATA] regions={len(df)} invalid={total_invalid} valid={total_valid} "
        f"invalid_ratio={total_invalid / max(len(df), 1):.6f}"
    )

    rows: list[dict[str, object]] = []

    single_specs = [
        ("area", AREA_THRESHOLDS),
        ("mean_lstm_prob", PROB_THRESHOLDS),
        ("max_lstm_prob", PROB_THRESHOLDS),
        ("mean_recent_mhw_days", RECENT_DAY_THRESHOLDS),
        ("mean_recent_exceed90_days", RECENT_DAY_THRESHOLDS),
        ("mean_latest_ssta", SSTA_THRESHOLDS),
        ("mean_intensity_ssta", SSTA_THRESHOLDS),
    ]

    for feature, thresholds in single_specs:
        values = df[feature].to_numpy(dtype=np.float32)
        for threshold in thresholds:
            add_rule(
                rows,
                invalid_mask,
                valid_mask,
                f"{feature}_lt_{threshold:g}",
                f"{feature} < {threshold:g}",
                values < threshold,
            )

    area = df["area"].to_numpy(dtype=np.float32)
    combo_specs = [
        ("mean_latest_ssta", SSTA_THRESHOLDS),
        ("mean_recent_mhw_days", RECENT_DAY_THRESHOLDS),
        ("mean_recent_exceed90_days", RECENT_DAY_THRESHOLDS),
        ("mean_lstm_prob", PROB_THRESHOLDS),
        ("max_lstm_prob", PROB_THRESHOLDS),
    ]
    for area_threshold, (feature, thresholds) in itertools.product(AREA_THRESHOLDS, combo_specs):
        feature_values = df[feature].to_numpy(dtype=np.float32)
        for threshold in thresholds:
            add_rule(
                rows,
                invalid_mask,
                valid_mask,
                f"area_lt_{area_threshold:g}_and_{feature}_lt_{threshold:g}",
                f"area < {area_threshold:g} AND {feature} < {threshold:g}",
                (area < area_threshold) & (feature_values < threshold),
            )

    out = pd.DataFrame(rows)
    ordered_cols = [
        "rule_name",
        "rule",
        "support",
        "correctly_removed",
        "wrongly_removed",
        "missed_invalid",
        "removal_precision",
        "removal_recall",
        "valid_loss_ratio",
        "removal_f1",
    ]
    out = out[ordered_cols].sort_values(
        ["removal_f1", "removal_precision", "support"],
        ascending=[False, False, False],
    )
    out.to_csv(cfg.REGION_REMOVAL_RULES_FILE, index=False)
    print(f"[SAVED] {cfg.REGION_REMOVAL_RULES_FILE} rules={len(out)}")

    row_dicts = out.to_dict("records")
    print_top("[TOP 20 REMOVAL RULES BY removal_f1]", row_dicts, ("removal_f1", "removal_precision", "support"))

    high_precision = [row for row in row_dicts if int(row["support"]) >= 100]
    print_top(
        "[TOP 20 REMOVAL RULES BY removal_precision WITH support >= 100]",
        high_precision,
        ("removal_precision", "removal_f1", "support"),
    )

    conservative = [
        row
        for row in row_dicts
        if float(row["valid_loss_ratio"]) <= 0.05
    ]
    print_top(
        "[TOP 20 REMOVAL RULES WITH valid_loss_ratio <= 0.05 BY correctly_removed]",
        conservative,
        ("correctly_removed", "removal_precision", "removal_f1"),
    )


if __name__ == "__main__":
    main()
