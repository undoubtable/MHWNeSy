#!/usr/bin/env python
"""Learn symbolic intensity rules for full-grid MHW candidate regions.

Intensity is computed inside predicted MHW regions. The script first builds an
intensity-region dataset, then learns simple threshold rules for weak,
moderate-or-strong, and strong valid MHW regions. This is intentionally a
transparent baseline rather than a full NeurRL learner.
"""

from __future__ import annotations

import argparse
import csv
import json
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


R_THRESHOLDS = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
S_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5]
E_THRESHOLDS = [0.0, 0.2, 0.5, 1.0, 1.5]
A_THRESHOLDS = [10, 20, 50, 100, 200]


@dataclass(frozen=True)
class IntensityRule:
    target: str
    rule_name: str
    rule: str
    feature: str
    op: str
    threshold: float
    area_threshold: float | None = None


def require_xarray():
    try:
        import xarray as xr  # noqa: WPS433 - optional runtime dependency
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[MISSING DEPENDENCY] xarray is required to read LABEL_FILE.\n"
            "Install project dependencies first: pip install -r requirements.txt"
        ) from exc
    return xr


def doy365_index(dates: pd.DatetimeIndex) -> np.ndarray:
    """Map timestamps to the 1..365 climatology convention used by labels."""

    doy = dates.dayofyear.to_numpy().astype(np.int16)
    is_feb29 = (dates.month == 2) & (dates.day == 29)
    after_feb29 = dates.is_leap_year & (
        (dates.month > 2) | ((dates.month == 2) & (dates.day > 29))
    )
    doy = doy - after_feb29.astype(np.int16)
    doy[is_feb29] = 59
    return np.clip(doy, 1, 365).astype(np.int16)


def validate_dataset(ds) -> tuple[str, str]:
    required = ["ssta", "mhw", "clim_mean", "thresh90"]
    missing = [name for name in required if name not in ds.data_vars]
    if missing:
        print("[DATA_VARS]", list(ds.data_vars))
        print("[DIMS]", dict(ds.sizes))
        raise SystemExit(f"[ERROR] LABEL_FILE is missing required variables: {missing}")

    dims = ds["ssta"].dims
    spatial_dims = [dim for dim in dims if dim != "time"]
    if len(spatial_dims) != 2:
        raise SystemExit(f"[ERROR] Expected time + 2 spatial dims for ssta, got {dims}")
    return spatial_dims[0], spatial_dims[1]


def label_components_python(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pure-Python 4-neighbor connected components fallback."""

    mask = mask.astype(bool, copy=False)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[np.ndarray, np.ndarray]] = []

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            rows = []
            cols = []
            queue: deque[tuple[int, int]] = deque([(r0, c0)])
            seen[r0, c0] = True
            while queue:
                r, c = queue.popleft()
                rows.append(r)
                cols.append(c)
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append((np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)))
    return components


def label_components(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return 4-neighbor connected components as row/col arrays."""

    if ndimage is None:
        return label_components_python(mask)

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, n_labels = ndimage.label(mask.astype(bool, copy=False), structure=structure)
    objects = ndimage.find_objects(labels)
    components: list[tuple[np.ndarray, np.ndarray]] = []
    for label_id in range(1, n_labels + 1):
        obj = objects[label_id - 1]
        if obj is None:
            continue
        local_rows, local_cols = np.where(labels[obj] == label_id)
        components.append(
            (
                (local_rows + obj[0].start).astype(np.int32),
                (local_cols + obj[1].start).astype(np.int32),
            )
        )
    return components


def nanmean_safe(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(values[finite].mean())


def nanmax_safe(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(values[finite].max())


def intensity_label(mean_severity_ratio: float) -> tuple[str, int]:
    if not np.isfinite(mean_severity_ratio) or mean_severity_ratio < 1.0:
        return "below_threshold", 0
    if mean_severity_ratio < 1.5:
        return "weak", 1
    if mean_severity_ratio < 2.0:
        return "moderate", 2
    return "strong", 3


def build_intensity_dataset(max_regions: int | None = None) -> pd.DataFrame:
    """Build and save the per-region intensity dataset."""

    required_files = [
        cfg.REGION_DATASET_FULL_GRID_FILE,
        cfg.FULL_GRID_TEST_PRED_FILE,
        cfg.FULL_GRID_TEST_PROB_FILE,
        cfg.FULL_GRID_TEST_META_FILE,
        cfg.LABEL_FILE,
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise SystemExit("[MISSING]\n" + "\n".join(missing))

    xr = require_xarray()
    region_df = pd.read_csv(cfg.REGION_DATASET_FULL_GRID_FILE)
    if max_regions is not None:
        region_df = region_df.head(max_regions).copy()
    selected_region_ids = set(int(x) for x in region_df["region_id"].tolist())
    region_by_id = region_df.set_index("region_id", drop=False)

    pred = np.load(cfg.FULL_GRID_TEST_PRED_FILE, mmap_mode="r")
    prob = np.load(cfg.FULL_GRID_TEST_PROB_FILE, mmap_mode="r")
    meta = np.load(cfg.FULL_GRID_TEST_META_FILE, allow_pickle=True)
    target_time_indices = meta["target_time_indices"].astype(np.int64)
    target_times = meta["target_times"] if "target_times" in meta.files else target_time_indices

    rows_out: list[dict[str, object]] = []
    global_region_id = 0

    ds = xr.open_dataset(cfg.LABEL_FILE)
    try:
        lat_dim, lon_dim = validate_dataset(ds)
        dates = pd.DatetimeIndex(ds["time"].values)
        doy365 = doy365_index(dates)
        target_doy_idx = (doy365[target_time_indices] - 1).astype(np.int64)
        print("[DATA_VARS]", list(ds.data_vars))
        print("[DIMS]", dict(ds.sizes))
        print(f"[INPUT] candidate_regions={len(region_df)} max_regions={max_regions}")
        print("[LOAD] preloading TEST_PERIOD ssta and 365-day threshold fields")

        # Loading the full test-period arrays once is much faster than pulling
        # one day at a time from NetCDF while iterating over connected regions.
        ssta_test = (
            ds["ssta"]
            .isel(time=target_time_indices)
            .transpose("time", lat_dim, lon_dim)
            .astype("float32")
            .values
        )
        clim_all = (
            ds["clim_mean"]
            .transpose("dayofyear", lat_dim, lon_dim)
            .astype("float32")
            .values
        )
        thresh_all = (
            ds["thresh90"]
            .transpose("dayofyear", lat_dim, lon_dim)
            .astype("float32")
            .values
        )
        print(
            f"[LOAD] ssta_test={ssta_test.shape} clim={clim_all.shape} "
            f"thresh={thresh_all.shape}"
        )

        for day_i, target_t in enumerate(target_time_indices):
            if max_regions is not None and global_region_id > max(selected_region_ids, default=-1):
                break

            pred_mask = pred[day_i].astype(bool)
            prob_day = np.nan_to_num(prob[day_i].astype(np.float32), nan=0.0)
            components = label_components(pred_mask)
            if not components:
                continue

            doy_idx = int(target_doy_idx[day_i])
            ssta_day = ssta_test[day_i]
            clim = clim_all[doy_idx]
            thresh = thresh_all[doy_idx]
            delta90 = thresh - clim
            delta90 = np.where(np.isfinite(delta90) & (delta90 > 0), delta90, np.nan).astype(np.float32)
            severity = ssta_day / delta90
            threshold_excess = ssta_day - delta90

            for comp_rows, comp_cols in components:
                region_id = global_region_id
                global_region_id += 1
                if region_id not in selected_region_ids:
                    continue

                cached = region_by_id.loc[region_id]
                area = int(len(comp_rows))
                ssta_vals = ssta_day[comp_rows, comp_cols]
                delta_vals = delta90[comp_rows, comp_cols]
                severity_vals = severity[comp_rows, comp_cols]
                excess_vals = threshold_excess[comp_rows, comp_cols]
                prob_vals = prob_day[comp_rows, comp_cols]

                mean_severity = nanmean_safe(severity_vals)
                label, klass = intensity_label(mean_severity)
                rows_out.append(
                    {
                        "region_id": region_id,
                        "target_time_index": int(target_t),
                        "target_time": str(target_times[day_i]),
                        "area": area,
                        "region_label": int(cached["region_label"]),
                        "mean_lstm_prob": float(np.mean(prob_vals)),
                        "mean_ssta": nanmean_safe(ssta_vals),
                        "max_ssta": nanmax_safe(ssta_vals),
                        "mean_delta90": nanmean_safe(delta_vals),
                        "mean_severity_ratio": mean_severity,
                        "max_severity_ratio": nanmax_safe(severity_vals),
                        "mean_threshold_excess": nanmean_safe(excess_vals),
                        "max_threshold_excess": nanmax_safe(excess_vals),
                        "mean_recent_mhw_days": float(cached["mean_recent_mhw_days"]),
                        "mean_recent_exceed90_days": float(cached["mean_recent_exceed90_days"]),
                        "intensity_label": label,
                        "intensity_class": klass,
                    }
                )

            if (day_i + 1) % 100 == 0 or day_i == 0 or len(rows_out) == len(region_df):
                print(
                    f"[INTENSITY] days={day_i + 1}/{len(target_time_indices)} "
                    f"rows={len(rows_out)}"
                )
            if max_regions is not None and len(rows_out) >= len(region_df):
                break
    finally:
        ds.close()

    out = pd.DataFrame(rows_out)
    out.to_csv(cfg.INTENSITY_REGION_DATASET_FILE, index=False)
    print(f"[SAVED] {cfg.INTENSITY_REGION_DATASET_FILE} rows={len(out)}")
    return out


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
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


def add_rule(
    rows: list[dict[str, object]],
    df: pd.DataFrame,
    target: str,
    y_true: np.ndarray,
    rule_name: str,
    rule: str,
    mask: np.ndarray,
) -> None:
    metrics = binary_metrics(y_true, mask)
    row: dict[str, object] = {"target": target, "rule_name": rule_name, "rule": rule}
    row.update(metrics)
    rows.append(row)


def learn_rules(intensity_df: pd.DataFrame) -> pd.DataFrame:
    """Learn simple threshold rules on valid regions only."""

    valid = intensity_df[intensity_df["region_label"] == 1].copy()
    valid = valid[np.isfinite(valid["mean_severity_ratio"])].copy()
    rows: list[dict[str, object]] = []

    targets = {
        "strong": (valid["intensity_class"].to_numpy(dtype=np.int32) == 3),
        "moderate_or_strong": (valid["intensity_class"].to_numpy(dtype=np.int32) >= 2),
        "weak": (valid["intensity_class"].to_numpy(dtype=np.int32) == 1),
    }

    for target in ("strong", "moderate_or_strong"):
        y_true = targets[target]
        for feature in ("mean_severity_ratio", "max_severity_ratio"):
            values = valid[feature].to_numpy(dtype=np.float32)
            for r in R_THRESHOLDS:
                add_rule(rows, valid, target, y_true, f"{feature}_ge_{r:g}", f"{feature} >= {r:g}", values >= r)
        for feature in ("mean_ssta", "max_ssta"):
            values = valid[feature].to_numpy(dtype=np.float32)
            for s in S_THRESHOLDS:
                add_rule(rows, valid, target, y_true, f"{feature}_ge_{s:g}", f"{feature} >= {s:g}", values >= s)
        for feature in ("mean_threshold_excess", "max_threshold_excess"):
            values = valid[feature].to_numpy(dtype=np.float32)
            for e in E_THRESHOLDS:
                add_rule(rows, valid, target, y_true, f"{feature}_ge_{e:g}", f"{feature} >= {e:g}", values >= e)
        area = valid["area"].to_numpy(dtype=np.float32)
        for a in A_THRESHOLDS:
            for r in R_THRESHOLDS:
                mask = (area >= a) & (valid["mean_severity_ratio"].to_numpy(dtype=np.float32) >= r)
                add_rule(rows, valid, target, y_true, f"area_ge_{a:g}_and_mean_severity_ratio_ge_{r:g}", f"area >= {a:g} AND mean_severity_ratio >= {r:g}", mask)
            for s in S_THRESHOLDS:
                mask = (area >= a) & (valid["mean_ssta"].to_numpy(dtype=np.float32) >= s)
                add_rule(rows, valid, target, y_true, f"area_ge_{a:g}_and_mean_ssta_ge_{s:g}", f"area >= {a:g} AND mean_ssta >= {s:g}", mask)

    y_true = targets["weak"]
    target = "weak"
    for feature, thresholds in (
        ("mean_severity_ratio", R_THRESHOLDS),
        ("max_severity_ratio", R_THRESHOLDS),
        ("mean_ssta", S_THRESHOLDS),
        ("mean_threshold_excess", E_THRESHOLDS),
    ):
        values = valid[feature].to_numpy(dtype=np.float32)
        for threshold in thresholds:
            add_rule(rows, valid, target, y_true, f"{feature}_lt_{threshold:g}", f"{feature} < {threshold:g}", values < threshold)
    area = valid["area"].to_numpy(dtype=np.float32)
    severity = valid["mean_severity_ratio"].to_numpy(dtype=np.float32)
    for a in A_THRESHOLDS:
        for r in R_THRESHOLDS:
            mask = (area < a) & (severity < r)
            add_rule(rows, valid, target, y_true, f"area_lt_{a:g}_and_mean_severity_ratio_lt_{r:g}", f"area < {a:g} AND mean_severity_ratio < {r:g}", mask)

    rules = pd.DataFrame(rows)
    rules = rules[["target", "rule_name", "rule", "support", "tp", "fp", "fn", "precision", "recall", "f1"]]
    rules.to_csv(cfg.INTENSITY_RULES_FILE, index=False)
    print(f"[SAVED] {cfg.INTENSITY_RULES_FILE} rules={len(rules)}")
    return rules


def best_rule(rules: pd.DataFrame, target: str) -> dict[str, object] | None:
    sub = rules[rules["target"] == target].sort_values(["f1", "precision", "support"], ascending=False)
    if sub.empty:
        return None
    row = sub.iloc[0].to_dict()
    return {
        key: (int(value) if isinstance(value, (np.integer,)) else float(value) if isinstance(value, (np.floating,)) else value)
        for key, value in row.items()
    }


def print_top(title: str, rules: pd.DataFrame, target: str, sort_cols: list[str], min_support: int | None = None) -> None:
    sub = rules[rules["target"] == target].copy()
    if min_support is not None:
        sub = sub[sub["support"] >= min_support]
    sub = sub.sort_values(sort_cols, ascending=False).head(20)
    print(title)
    if sub.empty:
        print("(none)")
        return
    print(sub.to_string(index=False))


def write_summary(intensity_df: pd.DataFrame, rules: pd.DataFrame) -> None:
    label_counts = intensity_df["intensity_label"].value_counts(dropna=False).to_dict()
    class_counts = intensity_df["intensity_class"].value_counts(dropna=False).sort_index().to_dict()
    summary = {
        "total_regions": int(len(intensity_df)),
        "valid_regions": int((intensity_df["region_label"] == 1).sum()),
        "invalid_regions": int((intensity_df["region_label"] == 0).sum()),
        "intensity_label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "intensity_class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "best_strong_rule": best_rule(rules, "strong"),
        "best_moderate_or_strong_rule": best_rule(rules, "moderate_or_strong"),
        "best_weak_rule": best_rule(rules, "weak"),
    }
    with cfg.INTENSITY_RULE_SUMMARY_FILE.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {cfg.INTENSITY_RULE_SUMMARY_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn intensity-level symbolic rules.")
    parser.add_argument("--max_regions", type=int, default=None, help="Debug limit on candidate regions.")
    args = parser.parse_args()

    cfg.ensure_dirs()
    intensity_df = build_intensity_dataset(max_regions=args.max_regions)
    rules = learn_rules(intensity_df)
    write_summary(intensity_df, rules)

    print_top("[TOP 20 strong RULES BY F1]", rules, "strong", ["f1", "precision", "support"])
    print_top("[TOP 20 strong RULES BY PRECISION support >= 50]", rules, "strong", ["precision", "f1", "support"], min_support=50)
    print_top("[TOP 20 moderate_or_strong RULES BY F1]", rules, "moderate_or_strong", ["f1", "precision", "support"])
    print_top("[TOP 20 weak RULES BY F1]", rules, "weak", ["f1", "precision", "support"])


if __name__ == "__main__":
    main()
