#!/usr/bin/env python
"""Build a region-level candidate table from point-wise predictions.

This first scaffold operates on the sampled test points saved by
``01_build_point_dataset.py``. For each target day, it reconstructs a sparse
prediction mask from sampled points, extracts 4-connected components, computes
simple region features, and assigns a region label by IoU against sampled true
MHW components from the same day.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


cfg = SourceFileLoader("test_cfg", str(Path(__file__).with_name("00_test_config.py"))).load_module()


def load_best_threshold(default: float | None = None) -> float | None:
    """Return the F1-best threshold from the sweep CSV if available."""

    if not cfg.POINT_THRESHOLD_SWEEP_FILE.exists():
        return default

    best_threshold = default
    best_f1 = -1.0
    best_precision = -1.0
    with cfg.POINT_THRESHOLD_SWEEP_FILE.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            threshold = float(row["threshold"])
            f1 = float(row["f1"])
            precision = float(row["precision"])
            if (f1, precision, threshold) > (best_f1, best_precision, best_threshold or -1.0):
                best_threshold = threshold
                best_f1 = f1
                best_precision = precision
    return best_threshold


def connected_components(coords: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Extract 4-connected components from a sparse coordinate set."""

    remaining = set(coords)
    components: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        queue: deque[tuple[int, int]] = deque([seed])
        while queue:
            row, col = queue.popleft()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def component_iou(
    predicted_component: set[tuple[int, int]],
    true_components: list[set[tuple[int, int]]],
) -> float:
    """Return max IoU against true components from the same target day."""

    if not true_components:
        return 0.0

    best = 0.0
    for true_component in true_components:
        intersection = len(predicted_component & true_component)
        union = len(predicted_component | true_component)
        if union:
            best = max(best, intersection / union)
    return best


def aggregate_day_coords(
    day_indices: np.ndarray,
    points: np.ndarray,
    mask: np.ndarray,
) -> set[tuple[int, int]]:
    """Convert point-level labels for one day into sparse coordinates."""

    coords: set[tuple[int, int]] = set()
    for idx in day_indices:
        if mask[idx]:
            _, row, col = points[idx]
            coords.add((int(row), int(col)))
    return coords


def main() -> None:
    parser = argparse.ArgumentParser(description="Build region candidates from point predictions.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold. If omitted, use F1-best sweep threshold when present, otherwise saved y_pred.",
    )
    parser.add_argument("--iou_threshold", type=float, default=0.1)
    args = parser.parse_args()

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
    saved_y_pred = pred["y_pred"].astype(np.uint8)
    test_points = data["test_points"].astype(np.int32)
    feature_names = [str(name) for name in data["feature_names"].tolist()]
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}

    required = {"ssta", "exceed90", "mhw"}
    missing = required.difference(feature_to_idx)
    if missing:
        raise SystemExit(f"[ERROR] Missing features in point dataset: {sorted(missing)}")

    threshold = args.threshold
    threshold_source = "command line"
    if threshold is None:
        threshold = load_best_threshold(default=None)
        threshold_source = "point_threshold_sweep.csv"

    if threshold is None:
        y_point_pred = saved_y_pred
        threshold_text = "saved_y_pred"
    else:
        y_point_pred = (y_prob >= threshold).astype(np.uint8)
        threshold_text = f"{threshold:.2f} ({threshold_source})"

    print(f"[PREDICTION LABELS] using threshold/source: {threshold_text}")

    mhw_idx = feature_to_idx["mhw"]
    exceed_idx = feature_to_idx["exceed90"]
    ssta_idx = feature_to_idx["ssta"]
    recent_mhw_days = (x_test[:, :, mhw_idx] > 0.5).sum(axis=1).astype(np.float32)
    recent_exceed90_days = (x_test[:, :, exceed_idx] > 0.5).sum(axis=1).astype(np.float32)
    latest_ssta = x_test[:, -1, ssta_idx].astype(np.float32)

    # A simple point-rule support signal for region aggregation. This is not the
    # final NeurRL rule set; it just gives a clear, interpretable region feature.
    prob_cut = float(threshold) if threshold is not None else 0.5
    point_rule_support = (
        (recent_mhw_days >= 2)
        | (recent_exceed90_days >= 3)
        | (latest_ssta > 0.0)
        | (y_prob >= prob_cut)
    )

    day_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, (target_t, _, _) in enumerate(test_points):
        day_to_indices[int(target_t)].append(idx)

    rows = []
    component_id = 0
    for target_t in sorted(day_to_indices):
        day_indices = np.array(day_to_indices[target_t], dtype=np.int64)
        pred_coords = aggregate_day_coords(day_indices, test_points, y_point_pred)
        true_coords = aggregate_day_coords(day_indices, test_points, y_true)

        pred_components = connected_components(pred_coords)
        true_components = connected_components(true_coords)

        coord_to_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx in day_indices:
            _, row, col = test_points[idx]
            coord_to_indices[(int(row), int(col))].append(int(idx))

        for component in pred_components:
            member_indices = [
                sample_idx
                for coord in component
                for sample_idx in coord_to_indices.get(coord, [])
            ]
            if not member_indices:
                continue

            member_indices_arr = np.array(member_indices, dtype=np.int64)
            max_iou = component_iou(component, true_components)
            region_label = int(max_iou >= args.iou_threshold)

            rows.append(
                {
                    "region_id": component_id,
                    "target_time_index": target_t,
                    "area": len(component),
                    "mean_lstm_prob": float(y_prob[member_indices_arr].mean()),
                    "mean_recent_mhw_days": float(recent_mhw_days[member_indices_arr].mean()),
                    "mean_recent_exceed90_days": float(recent_exceed90_days[member_indices_arr].mean()),
                    "mean_latest_ssta": float(latest_ssta[member_indices_arr].mean()),
                    "point_rule_support_ratio": float(point_rule_support[member_indices_arr].mean()),
                    "max_iou": float(max_iou),
                    "region_label": region_label,
                }
            )
            component_id += 1

    fieldnames = [
        "region_id",
        "target_time_index",
        "area",
        "mean_lstm_prob",
        "mean_recent_mhw_days",
        "mean_recent_exceed90_days",
        "mean_latest_ssta",
        "point_rule_support_ratio",
        "max_iou",
        "region_label",
    ]
    with cfg.REGION_DATASET_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    valid = sum(row["region_label"] for row in rows)
    invalid = total - valid
    positive_ratio = valid / total if total else 0.0
    print(f"[SAVED] {cfg.REGION_DATASET_FILE}")
    print(f"[REGIONS] total={total} valid={valid} invalid={invalid} positive_ratio={positive_ratio:.6f}")
    print(
        "[NOTE] Region masks and IoU are reconstructed from sampled test points; "
        "use dense point predictions later for full-grid region experiments."
    )


if __name__ == "__main__":
    main()
