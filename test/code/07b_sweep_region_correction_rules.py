#!/usr/bin/env python
"""Sweep full-grid region removal rules and evaluate pixel-level correction.

The expensive part is extracting predicted-region features from the full-grid
prediction masks. This script does that once, stores per-region deployable
features plus diagnostic true-overlap counts in memory, then evaluates all
candidate removal rules by vectorized aggregation. True MHW masks are used only
for evaluation and removed-region diagnostics, never for rule decisions.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd


cfg = SourceFileLoader("test_cfg", str(Path(__file__).with_name("00_test_config.py"))).load_module()

try:
    from scipy import ndimage
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    ndimage = None


AREA_THRESHOLDS = [2, 3, 5, 8, 10, 20, 30, 50]
RECENT_DAY_THRESHOLDS = [1, 2, 3, 4]
PROB_THRESHOLDS = [0.86, 0.88, 0.90, 0.92, 0.95]
SSTA_THRESHOLDS = [0.0, 0.5, 1.0, 1.5]


@dataclass(frozen=True)
class CandidateRule:
    rule_name: str
    rule: str
    kind: str
    area_threshold: float | None = None
    feature: str | None = None
    feature_threshold: float | None = None


def require_xarray():
    try:
        import xarray as xr  # noqa: WPS433 - optional runtime dependency
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[MISSING DEPENDENCY] xarray is required to read LABEL_FILE.\n"
            "Install project dependencies first: pip install -r requirements.txt"
        ) from exc
    return xr


def validate_dataset(ds) -> tuple[str, str]:
    missing = [name for name in ("ssta", "mhw", "exceed90") if name not in ds.data_vars]
    if missing:
        print("[DATA_VARS]", list(ds.data_vars))
        print("[DIMS]", dict(ds.sizes))
        raise SystemExit(f"[ERROR] LABEL_FILE is missing required variables: {missing}")

    dims = ds["mhw"].dims
    spatial_dims = [dim for dim in dims if dim != "time"]
    if len(spatial_dims) != 2:
        raise SystemExit(f"[ERROR] Expected time + 2 spatial dims, got {dims}")
    return spatial_dims[0], spatial_dims[1]


def label_components_python(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Pure-Python fallback for 4-neighbor connected components."""

    mask = mask.astype(bool, copy=False)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    components: list[tuple[np.ndarray, np.ndarray]] = []
    current_label = 0

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or labels[r0, c0] != 0:
                continue

            current_label += 1
            rows = []
            cols = []
            queue: deque[tuple[int, int]] = deque([(r0, c0)])
            labels[r0, c0] = current_label

            while queue:
                r, c = queue.popleft()
                rows.append(r)
                cols.append(c)
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = current_label
                        queue.append((nr, nc))

            components.append((np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)))

    return labels, components


def label_components(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Label 4-neighbor connected components in a 2D boolean mask."""

    if ndimage is None:
        return label_components_python(mask)

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, n_labels = ndimage.label(mask.astype(bool, copy=False), structure=structure)
    labels = labels.astype(np.int32, copy=False)
    objects = ndimage.find_objects(labels)

    components: list[tuple[np.ndarray, np.ndarray]] = []
    for label_id in range(1, n_labels + 1):
        obj = objects[label_id - 1]
        if obj is None:
            continue
        local = labels[obj] == label_id
        local_rows, local_cols = np.where(local)
        row_offset = obj[0].start
        col_offset = obj[1].start
        components.append(
            (
                (local_rows + row_offset).astype(np.int32),
                (local_cols + col_offset).astype(np.int32),
            )
        )
    return labels, components


def component_match_metrics(
    rows: np.ndarray,
    cols: np.ndarray,
    true_labels: np.ndarray,
    true_areas: np.ndarray,
) -> tuple[float, float]:
    """Compute diagnostic max IoU and overlap ratio for one predicted region."""

    area = len(rows)
    if area == 0 or len(true_areas) == 0:
        return 0.0, 0.0

    labels_inside = true_labels[rows, cols]
    labels_inside = labels_inside[labels_inside > 0]
    if labels_inside.size == 0:
        return 0.0, 0.0

    counts = np.bincount(labels_inside, minlength=len(true_areas) + 1)
    max_iou = 0.0
    max_intersection = 0
    for label_id in np.flatnonzero(counts):
        if label_id == 0:
            continue
        intersection = int(counts[label_id])
        true_area = int(true_areas[label_id - 1])
        union = area + true_area - intersection
        if union > 0:
            max_iou = max(max_iou, intersection / union)
        max_intersection = max(max_intersection, intersection)

    return float(max_iou), float(max_intersection / area)


def update_confusion(counts: dict[str, int], truth: np.ndarray, pred: np.ndarray) -> None:
    truth_bool = truth.astype(bool)
    pred_bool = pred.astype(bool)
    counts["tp"] += int(np.logical_and(truth_bool, pred_bool).sum())
    counts["fp"] += int(np.logical_and(~truth_bool, pred_bool).sum())
    counts["fn"] += int(np.logical_and(truth_bool, ~pred_bool).sum())
    counts["tn"] += int(np.logical_and(~truth_bool, ~pred_bool).sum())


def pixel_metrics(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_csi": iou,
    }


def make_candidate_rules(max_rules: int | None = None) -> list[CandidateRule]:
    rules: list[CandidateRule] = []

    for area_threshold in AREA_THRESHOLDS:
        rules.append(
            CandidateRule(
                rule_name=f"area_lt_{area_threshold:g}",
                rule=f"area < {area_threshold:g}",
                kind="area",
                area_threshold=area_threshold,
            )
        )

    for area_threshold in AREA_THRESHOLDS:
        for d in RECENT_DAY_THRESHOLDS:
            rules.append(
                CandidateRule(
                    rule_name=f"area_lt_{area_threshold:g}_and_mean_recent_mhw_days_lt_{d:g}",
                    rule=f"area < {area_threshold:g} AND mean_recent_mhw_days < {d:g}",
                    kind="area_and_feature",
                    area_threshold=area_threshold,
                    feature="mean_recent_mhw_days",
                    feature_threshold=d,
                )
            )

    for area_threshold in AREA_THRESHOLDS:
        for d in RECENT_DAY_THRESHOLDS:
            rules.append(
                CandidateRule(
                    rule_name=f"area_lt_{area_threshold:g}_and_mean_recent_exceed90_days_lt_{d:g}",
                    rule=f"area < {area_threshold:g} AND mean_recent_exceed90_days < {d:g}",
                    kind="area_and_feature",
                    area_threshold=area_threshold,
                    feature="mean_recent_exceed90_days",
                    feature_threshold=d,
                )
            )

    for area_threshold in AREA_THRESHOLDS:
        for p in PROB_THRESHOLDS:
            rules.append(
                CandidateRule(
                    rule_name=f"area_lt_{area_threshold:g}_and_mean_lstm_prob_lt_{p:g}",
                    rule=f"area < {area_threshold:g} AND mean_lstm_prob < {p:g}",
                    kind="area_and_feature",
                    area_threshold=area_threshold,
                    feature="mean_lstm_prob",
                    feature_threshold=p,
                )
            )

    for feature in ("mean_latest_ssta", "mean_intensity_ssta"):
        for area_threshold in AREA_THRESHOLDS:
            for s in SSTA_THRESHOLDS:
                rules.append(
                    CandidateRule(
                        rule_name=f"area_lt_{area_threshold:g}_and_{feature}_lt_{s:g}",
                        rule=f"area < {area_threshold:g} AND {feature} < {s:g}",
                        kind="area_and_feature",
                        area_threshold=area_threshold,
                        feature=feature,
                        feature_threshold=s,
                    )
                )

    if max_rules is not None:
        return rules[:max_rules]
    return rules


def rule_mask(rule: CandidateRule, features: dict[str, np.ndarray]) -> np.ndarray:
    area = features["area"]
    if rule.kind == "area":
        return area < float(rule.area_threshold)
    if rule.kind == "area_and_feature":
        assert rule.feature is not None
        assert rule.feature_threshold is not None
        return (area < float(rule.area_threshold)) & (
            features[rule.feature] < float(rule.feature_threshold)
        )
    raise ValueError(f"Unknown rule kind: {rule.kind}")


def collect_region_features(max_days: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Load cached region features and compute diagnostic pixel counts once.

    ``region_dataset_full_grid.csv`` already stores deployable per-region
    features from the full-grid prediction masks. This sweep reuses those
    features, then reads only target-day true masks to compute pixel-level
    correction deltas. ``max_iou`` and ``overlap_ratio`` are not used as rule
    features.
    """

    required_files = [
        cfg.FULL_GRID_TEST_PRED_FILE,
        cfg.FULL_GRID_TEST_PROB_FILE,
        cfg.FULL_GRID_TEST_META_FILE,
        cfg.LABEL_FILE,
        cfg.REGION_DATASET_FULL_GRID_FILE,
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise SystemExit("[MISSING]\n" + "\n".join(missing))

    xr = require_xarray()
    pred = np.load(cfg.FULL_GRID_TEST_PRED_FILE, mmap_mode="r")
    meta = np.load(cfg.FULL_GRID_TEST_META_FILE, allow_pickle=True)
    target_time_indices = meta["target_time_indices"].astype(np.int64)
    target_times = meta["target_times"] if "target_times" in meta.files else target_time_indices
    ocean_mask = (
        meta["ocean_mask"].astype(bool)
        if "ocean_mask" in meta.files
        else np.ones(pred.shape[1:], dtype=bool)
    )

    if max_days is not None:
        target_time_indices = target_time_indices[:max_days]
        target_times = target_times[:max_days]

    if len(target_time_indices) == 0:
        raise SystemExit("[ERROR] No target days selected.")

    region_df = pd.read_csv(cfg.REGION_DATASET_FULL_GRID_FILE)
    selected_targets = set(int(x) for x in target_time_indices.tolist())
    region_df = region_df[region_df["target_time_index"].isin(selected_targets)].copy()
    required_region_cols = {
        "target_time_index",
        "area",
        "mean_lstm_prob",
        "mean_recent_mhw_days",
        "mean_recent_exceed90_days",
        "mean_latest_ssta",
        "mean_intensity_ssta",
        "region_label",
    }
    missing_region_cols = sorted(required_region_cols.difference(region_df.columns))
    if missing_region_cols:
        raise SystemExit(f"[ERROR] Missing columns in region feature cache: {missing_region_cols}")

    before_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    true_pixels_accum: list[int] = []
    false_pixels_accum: list[int] = []

    print(f"[INPUT] pred_shape={pred.shape} selected_days={len(target_time_indices)}")
    print(f"[CACHE] {cfg.REGION_DATASET_FULL_GRID_FILE} rows={len(region_df)}")
    ds = xr.open_dataset(cfg.LABEL_FILE)
    try:
        lat_dim, lon_dim = validate_dataset(ds)
        region_count = 0
        for day_i, target_t in enumerate(target_time_indices):
            pred_day = pred[day_i].astype(bool)
            true_mask = (
                ds["mhw"]
                .isel(time=int(target_t))
                .transpose(lat_dim, lon_dim)
                .astype("uint8")
                .values
                > 0
            )

            update_confusion(before_counts, true_mask[ocean_mask], pred_day[ocean_mask])
            _, pred_components = label_components(pred_day)
            expected = int((region_df["target_time_index"] == int(target_t)).sum())
            if expected != len(pred_components):
                raise SystemExit(
                    "[ERROR] Region cache and mask components do not match "
                    f"for target_t={int(target_t)}: cache={expected}, mask={len(pred_components)}"
                )
            for rows, cols in pred_components:
                area = int(len(rows))
                true_pixels = int(true_mask[rows, cols].sum())
                false_pixels = area - true_pixels
                true_pixels_accum.append(true_pixels)
                false_pixels_accum.append(false_pixels)
                region_count += 1

            if (day_i + 1) % 100 == 0 or day_i == 0 or day_i + 1 == len(target_time_indices):
                print(
                    f"[FEATURES] {day_i + 1}/{len(target_time_indices)} "
                    f"target_t={int(target_t)} date={target_times[day_i]} "
                    f"regions={region_count}"
                )
    finally:
        ds.close()

    if region_count != len(region_df):
        raise SystemExit(f"[ERROR] Extracted regions={region_count} but cache rows={len(region_df)}")

    features: dict[str, np.ndarray] = {
        "area": region_df["area"].to_numpy(dtype=np.float32),
        "mean_lstm_prob": region_df["mean_lstm_prob"].to_numpy(dtype=np.float32),
        "mean_recent_mhw_days": region_df["mean_recent_mhw_days"].to_numpy(dtype=np.float32),
        "mean_recent_exceed90_days": region_df["mean_recent_exceed90_days"].to_numpy(dtype=np.float32),
        "mean_latest_ssta": region_df["mean_latest_ssta"].to_numpy(dtype=np.float32),
        "mean_intensity_ssta": region_df["mean_intensity_ssta"].to_numpy(dtype=np.float32),
        "true_pixels": np.array(true_pixels_accum, dtype=np.float32),
        "false_pixels": np.array(false_pixels_accum, dtype=np.float32),
        "is_valid_region": region_df["region_label"].to_numpy(dtype=np.uint8),
    }
    print(f"[FEATURES] total_pred_regions={len(features['area'])}")
    return features, before_counts


def evaluate_rule(rule: CandidateRule, features: dict[str, np.ndarray], before_counts: dict[str, int]) -> dict[str, object]:
    remove = rule_mask(rule, features)
    true_removed = int(features["true_pixels"][remove].sum())
    false_removed = int(features["false_pixels"][remove].sum())

    after_counts = {
        "tp": before_counts["tp"] - true_removed,
        "fp": before_counts["fp"] - false_removed,
        "fn": before_counts["fn"] + true_removed,
        "tn": before_counts["tn"] + false_removed,
    }
    before = pixel_metrics(before_counts)
    after = pixel_metrics(after_counts)
    delta = {key: after[key] - before[key] for key in before}

    is_valid = features["is_valid_region"].astype(bool)
    removed_regions = int(remove.sum())
    wrongly_removed = int(np.logical_and(remove, is_valid).sum())
    correctly_removed = int(np.logical_and(remove, ~is_valid).sum())
    total_valid_regions = int(is_valid.sum())
    removal_precision = correctly_removed / max(removed_regions, 1)
    valid_loss_ratio = wrongly_removed / max(total_valid_regions, 1)

    row: dict[str, object] = {
        "rule_name": rule.rule_name,
        "rule": rule.rule,
        "removed_regions": removed_regions,
        "correctly_removed_invalid_regions": correctly_removed,
        "wrongly_removed_valid_regions": wrongly_removed,
        "removal_precision": removal_precision,
        "valid_loss_ratio": valid_loss_ratio,
    }
    for metric in ("accuracy", "precision", "recall", "f1", "iou_csi"):
        row[f"before_{metric}"] = before[metric]
        row[f"after_{metric}"] = after[metric]
        row[f"delta_{metric}"] = delta[metric]
    return row


def write_rows(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "rule_name",
        "rule",
        "removed_regions",
        "correctly_removed_invalid_regions",
        "wrongly_removed_valid_regions",
        "removal_precision",
        "valid_loss_ratio",
        "before_accuracy",
        "after_accuracy",
        "delta_accuracy",
        "before_precision",
        "after_precision",
        "delta_precision",
        "before_recall",
        "after_recall",
        "delta_recall",
        "before_f1",
        "after_f1",
        "delta_f1",
        "before_iou_csi",
        "after_iou_csi",
        "delta_iou_csi",
    ]
    with cfg.REGION_CORRECTION_SWEEP_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
            f"removed={row['removed_regions']} "
            f"correct={row['correctly_removed_invalid_regions']} "
            f"wrong={row['wrongly_removed_valid_regions']} "
            f"delta_f1={float(row['delta_f1']):+.6f} "
            f"delta_iou={float(row['delta_iou_csi']):+.6f} "
            f"delta_precision={float(row['delta_precision']):+.6f} "
            f"delta_recall={float(row['delta_recall']):+.6f} "
            f"removal_precision={float(row['removal_precision']):.6f} "
            f"valid_loss={float(row['valid_loss_ratio']):.6f} | {row['rule']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep full-grid region correction rules.")
    parser.add_argument("--max_rules", type=int, default=None, help="Debug limit for candidate rules.")
    parser.add_argument("--max_days", type=int, default=None, help="Debug limit for target days.")
    args = parser.parse_args()

    cfg.ensure_dirs()
    rules = make_candidate_rules(max_rules=args.max_rules)
    features, before_counts = collect_region_features(max_days=args.max_days)
    rows = [evaluate_rule(rule, features, before_counts) for rule in rules]
    write_rows(rows)
    print(f"[SAVED] {cfg.REGION_CORRECTION_SWEEP_FILE} rules={len(rows)}")

    print_top("[TOP 20 BY delta_f1]", rows, ("delta_f1", "delta_iou_csi", "delta_precision"))
    print_top("[TOP 20 BY delta_iou_csi]", rows, ("delta_iou_csi", "delta_f1", "delta_precision"))

    precision_balanced = [row for row in rows if float(row["delta_recall"]) >= -0.001]
    print_top(
        "[TOP 20 BY delta_precision WITH delta_recall >= -0.001]",
        precision_balanced,
        ("delta_precision", "delta_f1", "delta_iou_csi"),
    )

    high_precision = [row for row in rows if int(row["removed_regions"]) >= 100]
    print_top(
        "[TOP 20 BY removal_precision WITH removed_regions >= 100]",
        high_precision,
        ("removal_precision", "delta_f1", "removed_regions"),
    )

    conservative = [row for row in rows if float(row["valid_loss_ratio"]) <= 0.01]
    print_top(
        "[TOP 20 valid_loss_ratio <= 0.01 BY correctly_removed_invalid_regions]",
        conservative,
        ("correctly_removed_invalid_regions", "removal_precision", "delta_f1"),
    )


if __name__ == "__main__":
    main()
