#!/usr/bin/env python
"""Build full-grid region candidates from full-grid point LSTM predictions."""

from __future__ import annotations

import csv
import argparse
from collections import deque
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


cfg = SourceFileLoader("test_cfg", str(Path(__file__).with_name("00_test_config.py"))).load_module()


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
    missing = [name for name in cfg.INPUT_VARIABLES if name not in ds.data_vars]
    if missing:
        print("[DATA_VARS]", list(ds.data_vars))
        print("[DIMS]", dict(ds.sizes))
        raise SystemExit(f"[ERROR] LABEL_FILE is missing required variables: {missing}")

    dims = ds[cfg.INPUT_VARIABLES[0]].dims
    spatial_dims = [dim for dim in dims if dim != "time"]
    if len(spatial_dims) != 2:
        raise SystemExit(f"[ERROR] Expected time + 2 spatial dims, got {dims}")
    return spatial_dims[0], spatial_dims[1]


def label_components(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Label 4-neighbor connected components in a 2D boolean mask."""

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


def component_match_metrics(
    rows: np.ndarray,
    cols: np.ndarray,
    true_labels: np.ndarray,
    true_areas: np.ndarray,
) -> tuple[float, float]:
    """Compute max IoU and true-overlap ratio for one predicted component."""

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

    overlap_ratio = max_intersection / area
    return float(max_iou), float(overlap_ratio)


def add_feature_stats(stats: dict[int, dict[str, float]], label: int, row: dict[str, float], feature_cols: list[str]) -> None:
    if label not in stats:
        stats[label] = {"count": 0.0, **{name: 0.0 for name in feature_cols}}
    stats[label]["count"] += 1.0
    for name in feature_cols:
        stats[label][name] += float(row[name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-grid region dataset from point LSTM fields.")
    parser.add_argument("--iou_threshold", type=float, default=0.1)
    parser.add_argument("--overlap_threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg.ensure_dirs()
    required_files = [
        cfg.FULL_GRID_TEST_PROB_FILE,
        cfg.FULL_GRID_TEST_PRED_FILE,
        cfg.FULL_GRID_TEST_META_FILE,
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise SystemExit(
            "[MISSING] Full-grid point prediction files are required:\n"
            + "\n".join(missing)
            + "\nRun first: python test/code/05a_predict_point_lstm_full_grid.py"
        )
    if not cfg.LABEL_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.LABEL_FILE}")

    xr = require_xarray()
    prob = np.load(cfg.FULL_GRID_TEST_PROB_FILE, mmap_mode="r")
    pred = np.load(cfg.FULL_GRID_TEST_PRED_FILE, mmap_mode="r")
    meta = np.load(cfg.FULL_GRID_TEST_META_FILE, allow_pickle=True)
    target_time_indices = meta["target_time_indices"].astype(np.int64)
    target_times = meta["target_times"] if "target_times" in meta.files else target_time_indices
    threshold = float(meta["threshold"])

    if prob.shape != pred.shape:
        raise SystemExit(f"[ERROR] prob shape {prob.shape} != pred shape {pred.shape}")
    if prob.shape[0] != len(target_time_indices):
        raise SystemExit("[ERROR] full-grid arrays and target_time_indices have inconsistent day counts.")

    feature_cols = [
        "area",
        "mean_lstm_prob",
        "max_lstm_prob",
        "mean_recent_mhw_days",
        "mean_recent_exceed90_days",
        "mean_latest_ssta",
        "mean_intensity_ssta",
        "point_rule_support_ratio",
    ]
    fieldnames = [
        "region_id",
        "target_time_index",
        "target_time",
        *feature_cols,
        "max_iou",
        "overlap_ratio",
        "region_label",
    ]

    label_counts = {0: 0, 1: 0}
    feature_stats: dict[int, dict[str, float]] = {}
    region_id = 0

    ds = xr.open_dataset(cfg.LABEL_FILE)
    try:
        lat_dim, lon_dim = validate_dataset(ds)
        print(
            f"[INPUT] prob_shape={prob.shape} pred_shape={pred.shape} "
            f"days={len(target_time_indices)} threshold={threshold:.2f}"
        )

        with cfg.REGION_DATASET_FULL_GRID_FILE.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for day_i, target_t in enumerate(target_time_indices):
                source_end = int(target_t) - cfg.LEAD_DAYS
                source_start = source_end - cfg.HISTORY_DAYS + 1

                pred_mask = pred[day_i].astype(bool)
                true_mask = (
                    ds["mhw"]
                    .isel(time=int(target_t))
                    .transpose(lat_dim, lon_dim)
                    .astype("uint8")
                    .values
                    > 0
                )

                _, pred_components = label_components(pred_mask)
                true_labels, true_components = label_components(true_mask)
                true_areas = np.array([len(rows) for rows, _ in true_components], dtype=np.int32)

                mhw_hist = (
                    ds["mhw"]
                    .isel(time=slice(source_start, source_end + 1))
                    .transpose("time", lat_dim, lon_dim)
                    .astype("float32")
                    .values
                )
                exceed_hist = (
                    ds["exceed90"]
                    .isel(time=slice(source_start, source_end + 1))
                    .transpose("time", lat_dim, lon_dim)
                    .astype("float32")
                    .values
                )
                latest_ssta = (
                    ds["ssta"]
                    .isel(time=source_end)
                    .transpose(lat_dim, lon_dim)
                    .astype("float32")
                    .values
                )

                recent_mhw_days = np.nan_to_num((mhw_hist > 0.5).sum(axis=0), nan=0.0).astype(np.float32)
                recent_exceed90_days = np.nan_to_num((exceed_hist > 0.5).sum(axis=0), nan=0.0).astype(np.float32)
                latest_ssta = np.nan_to_num(latest_ssta, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                intensity_ssta = np.maximum(latest_ssta, 0.0)

                # Physical point-rule support only. LSTM probability is kept as
                # separate region features, so this ratio remains informative
                # inside predicted regions.
                point_support = (
                    (recent_mhw_days >= 2)
                    | (recent_exceed90_days >= 3)
                    | (latest_ssta > 0.0)
                )

                for rows, cols in pred_components:
                    area = int(len(rows))
                    prob_values = np.nan_to_num(prob[day_i, rows, cols], nan=0.0)
                    max_iou, overlap_ratio = component_match_metrics(
                        rows,
                        cols,
                        true_labels,
                        true_areas,
                    )
                    region_label = int(
                        max_iou >= args.iou_threshold
                        or overlap_ratio >= args.overlap_threshold
                    )

                    row = {
                        "region_id": region_id,
                        "target_time_index": int(target_t),
                        "target_time": str(target_times[day_i]),
                        "area": area,
                        "mean_lstm_prob": float(prob_values.mean()),
                        "max_lstm_prob": float(prob_values.max()),
                        "mean_recent_mhw_days": float(recent_mhw_days[rows, cols].mean()),
                        "mean_recent_exceed90_days": float(recent_exceed90_days[rows, cols].mean()),
                        "mean_latest_ssta": float(latest_ssta[rows, cols].mean()),
                        "mean_intensity_ssta": float(intensity_ssta[rows, cols].mean()),
                        "point_rule_support_ratio": float(point_support[rows, cols].mean()),
                        "max_iou": max_iou,
                        "overlap_ratio": overlap_ratio,
                        "region_label": region_label,
                    }
                    writer.writerow(row)
                    label_counts[region_label] += 1
                    add_feature_stats(feature_stats, region_label, row, feature_cols)
                    region_id += 1

                if (day_i + 1) % 100 == 0 or day_i == 0 or day_i + 1 == len(target_time_indices):
                    print(f"[REGION] {day_i + 1}/{len(target_time_indices)} total_regions={region_id}")

        total = label_counts[0] + label_counts[1]
        positive_ratio = label_counts[1] / total if total else 0.0
        print(f"[SAVED] {cfg.REGION_DATASET_FULL_GRID_FILE}")
        print(
            f"[SUMMARY] total_regions={total} valid={label_counts[1]} "
            f"invalid={label_counts[0]} positive_ratio={positive_ratio:.6f}"
        )
        print("[FEATURE MEANS BY region_label]")
        for label in (0, 1):
            count = feature_stats.get(label, {}).get("count", 0.0)
            if count == 0:
                print(f"region_label={label}: count=0")
                continue
            means = {
                name: feature_stats[label][name] / count
                for name in feature_cols
            }
            mean_text = ", ".join(f"{name}={value:.6f}" for name, value in means.items())
            print(f"region_label={label}: count={int(count)}, {mean_text}")
        print("[NOTE] max_iou and overlap_ratio are diagnostics used for labeling only, not rule features.")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
