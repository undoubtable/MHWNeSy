#!/usr/bin/env python
"""Apply region removal rules to full-grid point LSTM predictions.

Removal rules use only deployable region features. IoU and overlap ratio are
computed only after the rule fires, to diagnose whether removed regions were
actually invalid.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


cfg = SourceFileLoader("test_cfg", str(Path(__file__).with_name("00_test_config.py"))).load_module()


RULE_DESCRIPTIONS = {
    "rule_a": "area < 30",
    "rule_b": "area < 2 AND mean_recent_mhw_days < 1",
    "rule_c": "area < 50 AND mean_lstm_prob < 0.88",
}


def require_xarray():
    try:
        import xarray as xr  # noqa: WPS433 - optional runtime dependency
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[MISSING DEPENDENCY] xarray is required to read LABEL_FILE.\n"
            "Install project dependencies first: pip install -r requirements.txt"
        ) from exc
    return xr


def corrected_pred_path(rule_name: str) -> Path:
    return cfg.POINT_LSTM_DIR / f"full_grid_test_pred_corrected_{rule_name}.npy"


def summary_paths(rule_name: str) -> tuple[Path, Path]:
    """Return per-rule summary paths to avoid overwriting other rule runs."""

    csv_path = cfg.TEST_OUTPUT_DIR / f"region_rule_correction_summary_{rule_name}.csv"
    json_path = cfg.TEST_OUTPUT_DIR / f"region_rule_correction_summary_{rule_name}.json"
    return csv_path, json_path


def validate_dataset(ds) -> tuple[str, str]:
    missing = [name for name in ("ssta", "mhw") if name not in ds.data_vars]
    if missing:
        print("[DATA_VARS]", list(ds.data_vars))
        print("[DIMS]", dict(ds.sizes))
        raise SystemExit(f"[ERROR] LABEL_FILE is missing required variables: {missing}")

    dims = ds["mhw"].dims
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


def should_remove(rule_name: str, area: int, mean_prob: float, mean_recent_mhw_days: float) -> bool:
    if rule_name == "rule_a":
        return area < 30
    if rule_name == "rule_b":
        return area < 2 and mean_recent_mhw_days < 1.0
    if rule_name == "rule_c":
        return area < 50 and mean_prob < 0.88
    raise ValueError(f"Unknown rule_name: {rule_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply full-grid region removal rules.")
    parser.add_argument("--rule", choices=sorted(RULE_DESCRIPTIONS), default="rule_c")
    parser.add_argument("--iou_threshold", type=float, default=0.1)
    parser.add_argument("--overlap_threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg.ensure_dirs()
    required_files = [
        cfg.FULL_GRID_TEST_PRED_FILE,
        cfg.FULL_GRID_TEST_PROB_FILE,
        cfg.FULL_GRID_TEST_META_FILE,
        cfg.LABEL_FILE,
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise SystemExit("[MISSING]\n" + "\n".join(missing))

    xr = require_xarray()
    pred = np.load(cfg.FULL_GRID_TEST_PRED_FILE, mmap_mode="r")
    prob = np.load(cfg.FULL_GRID_TEST_PROB_FILE, mmap_mode="r")
    meta = np.load(cfg.FULL_GRID_TEST_META_FILE, allow_pickle=True)
    target_time_indices = meta["target_time_indices"].astype(np.int64)
    target_times = meta["target_times"] if "target_times" in meta.files else target_time_indices
    ocean_mask = meta["ocean_mask"].astype(bool) if "ocean_mask" in meta.files else np.ones(pred.shape[1:], dtype=bool)

    if pred.shape != prob.shape:
        raise SystemExit(f"[ERROR] pred shape {pred.shape} != prob shape {prob.shape}")
    if pred.shape[0] != len(target_time_indices):
        raise SystemExit("[ERROR] full-grid arrays and target_time_indices have inconsistent day counts.")

    corrected_path = corrected_pred_path(args.rule)
    corrected = np.lib.format.open_memmap(
        corrected_path,
        mode="w+",
        dtype="uint8",
        shape=pred.shape,
    )

    before_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    after_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    total_pred_regions = 0
    valid_pred_regions = 0
    invalid_pred_regions = 0
    removed_regions = 0
    correctly_removed_invalid_regions = 0
    wrongly_removed_valid_regions = 0

    print(f"[RULE] {args.rule}: {RULE_DESCRIPTIONS[args.rule]}")
    print(f"[INPUT] pred_shape={pred.shape} days={len(target_time_indices)}")
    print(f"[OUTPUT] {corrected_path}")

    ds = xr.open_dataset(cfg.LABEL_FILE)
    try:
        lat_dim, lon_dim = validate_dataset(ds)

        for day_i, target_t in enumerate(target_time_indices):
            source_end = int(target_t) - cfg.LEAD_DAYS
            source_start = source_end - cfg.HISTORY_DAYS + 1

            pred_day = pred[day_i].astype(bool)
            prob_day = prob[day_i].astype(np.float32)
            corrected_day = pred_day.copy()

            true_mask = (
                ds["mhw"]
                .isel(time=int(target_t))
                .transpose(lat_dim, lon_dim)
                .astype("uint8")
                .values
                > 0
            )
            true_labels, true_components = label_components(true_mask)
            true_areas = np.array([len(rows) for rows, _ in true_components], dtype=np.int32)

            mhw_hist = (
                ds["mhw"]
                .isel(time=slice(source_start, source_end + 1))
                .transpose("time", lat_dim, lon_dim)
                .astype("float32")
                .values
            )
            recent_mhw_days = np.nan_to_num((mhw_hist > 0.5).sum(axis=0), nan=0.0).astype(np.float32)

            _, pred_components = label_components(pred_day)
            total_pred_regions += len(pred_components)
            for rows, cols in pred_components:
                area = int(len(rows))
                mean_prob = float(np.nan_to_num(prob_day[rows, cols], nan=0.0).mean())
                mean_recent_mhw_days = float(recent_mhw_days[rows, cols].mean())

                max_iou, overlap_ratio = component_match_metrics(rows, cols, true_labels, true_areas)
                is_valid = max_iou >= args.iou_threshold or overlap_ratio >= args.overlap_threshold
                if is_valid:
                    valid_pred_regions += 1
                else:
                    invalid_pred_regions += 1

                if not should_remove(args.rule, area, mean_prob, mean_recent_mhw_days):
                    continue

                removed_regions += 1
                if is_valid:
                    wrongly_removed_valid_regions += 1
                else:
                    correctly_removed_invalid_regions += 1
                corrected_day[rows, cols] = False

            update_confusion(before_counts, true_mask[ocean_mask], pred_day[ocean_mask])
            update_confusion(after_counts, true_mask[ocean_mask], corrected_day[ocean_mask])
            corrected[day_i] = corrected_day.astype(np.uint8)

            if (day_i + 1) % 100 == 0 or day_i == 0 or day_i + 1 == len(target_time_indices):
                print(
                    f"[APPLY] {day_i + 1}/{len(target_time_indices)} "
                    f"target_t={int(target_t)} date={target_times[day_i]} "
                    f"removed={removed_regions}"
                )

        corrected.flush()
    finally:
        ds.close()

    before = pixel_metrics(before_counts)
    after = pixel_metrics(after_counts)
    removal_precision = correctly_removed_invalid_regions / max(removed_regions, 1)
    valid_loss_ratio = wrongly_removed_valid_regions / max(valid_pred_regions, 1)

    summary = {
        "rule_name": args.rule,
        "rule": RULE_DESCRIPTIONS[args.rule],
        "corrected_pred_file": str(corrected_path),
        "total_pred_regions": total_pred_regions,
        "valid_pred_regions": valid_pred_regions,
        "invalid_pred_regions": invalid_pred_regions,
        "removed_regions": removed_regions,
        "correctly_removed_invalid_regions": correctly_removed_invalid_regions,
        "wrongly_removed_valid_regions": wrongly_removed_valid_regions,
        "removal_precision": removal_precision,
        "valid_loss_ratio": valid_loss_ratio,
        "pixel_metrics_scope": "ocean pixels from full_grid_test_meta.ocean_mask",
        "before": before,
        "after": after,
        "delta": {key: after[key] - before[key] for key in before},
    }

    summary_csv, summary_json = summary_paths(args.rule)
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    flat_row = {
        "rule_name": args.rule,
        "rule": RULE_DESCRIPTIONS[args.rule],
        "corrected_pred_file": str(corrected_path),
        "total_pred_regions": total_pred_regions,
        "valid_pred_regions": valid_pred_regions,
        "invalid_pred_regions": invalid_pred_regions,
        "removed_regions": removed_regions,
        "correctly_removed_invalid_regions": correctly_removed_invalid_regions,
        "wrongly_removed_valid_regions": wrongly_removed_valid_regions,
        "removal_precision": removal_precision,
        "valid_loss_ratio": valid_loss_ratio,
    }
    for prefix, metrics in (("before", before), ("after", after), ("delta", summary["delta"])):
        for key, value in metrics.items():
            flat_row[f"{prefix}_{key}"] = value

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_row.keys()))
        writer.writeheader()
        writer.writerow(flat_row)

    print("[PIXEL METRICS]")
    print("metric,before,after,delta")
    for key in ("accuracy", "precision", "recall", "f1", "iou_csi"):
        print(f"{key},{before[key]:.6f},{after[key]:.6f},{summary['delta'][key]:+.6f}")
    print("[REMOVAL STATS]")
    print(f"total_pred_regions={total_pred_regions}")
    print(f"valid_pred_regions={valid_pred_regions}")
    print(f"invalid_pred_regions={invalid_pred_regions}")
    print(f"removed_regions={removed_regions}")
    print(f"correctly_removed_invalid_regions={correctly_removed_invalid_regions}")
    print(f"wrongly_removed_valid_regions={wrongly_removed_valid_regions}")
    print(f"removal_precision={removal_precision:.6f}")
    print(f"valid_loss_ratio={valid_loss_ratio:.6f}")
    print(f"[SAVED] {summary_csv}")
    print(f"[SAVED] {summary_json}")


if __name__ == "__main__":
    main()
