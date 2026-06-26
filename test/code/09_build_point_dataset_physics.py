#!/usr/bin/env python
"""Build a physics-enhanced point-wise MHW forecasting dataset.

Each sample is one ocean pixel and one target day. The input window is
``t-14 ... t-5`` when HISTORY_DAYS=10 and LEAD_DAYS=5, and the target is the
MHW label at day ``t``.

The feature tensor keeps the LSTM-friendly shape:

    X: [num_samples, history_days, num_features]

Window-level rule/physics features are repeated at every time step. This keeps
the model interface simple while making the same features available for later
symbolic rule evaluation.
"""

from __future__ import annotations

import argparse
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd


cfg = SourceFileLoader("test_cfg", str(Path(__file__).with_name("00_test_config.py"))).load_module()

REQUIRED_VARS = ("ssta", "exceed90", "mhw", "clim_mean", "thresh90")


def require_xarray():
    try:
        import xarray as xr  # noqa: WPS433 - optional runtime dependency
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[MISSING DEPENDENCY] xarray is required to read LABEL_FILE.\n"
            "Use the project environment, for example: conda run -n ybzcu121 python ..."
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


def split_target_indices(times: np.ndarray, start: str, end: str) -> np.ndarray:
    """Return target-day indices whose lead-separated history window is valid."""

    start64 = np.datetime64(start)
    end64 = np.datetime64(end)
    min_target = cfg.HISTORY_DAYS + cfg.LEAD_DAYS - 1
    idx = np.arange(len(times), dtype=np.int64)
    mask = (idx >= min_target) & (times >= start64) & (times <= end64)
    return idx[mask]


def validate_dataset(ds) -> tuple[str, str]:
    missing = [name for name in REQUIRED_VARS if name not in ds.data_vars]
    if missing:
        print("[DATA_VARS]", list(ds.data_vars))
        print("[DIMS]", dict(ds.sizes))
        raise SystemExit(f"[ERROR] LABEL_FILE is missing required variables: {missing}")

    dims = ds["ssta"].dims
    if "time" not in dims:
        print("[DATA_VARS]", list(ds.data_vars))
        print("[DIMS]", dict(ds.sizes))
        raise SystemExit(f"[ERROR] Expected time dimension for ssta, got {dims}")

    spatial_dims = [dim for dim in dims if dim != "time"]
    if len(spatial_dims) != 2:
        print("[DATA_VARS]", list(ds.data_vars))
        print("[DIMS]", dict(ds.sizes))
        raise SystemExit(f"[ERROR] Expected two spatial dimensions for ssta, got {dims}")

    for name in ("ssta", "exceed90", "mhw"):
        if set(ds[name].dims) != set(dims):
            raise SystemExit(f"[ERROR] Variable {name!r} dims {ds[name].dims} do not match {dims}")

    print("[DATA_VARS]", list(ds.data_vars))
    print("[DIMS]", dict(ds.sizes))
    return spatial_dims[0], spatial_dims[1]


def load_arrays(ds, lat_dim: str, lon_dim: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load input arrays and compute threshold_gap for every day/grid cell."""

    arrays: dict[str, np.ndarray] = {}
    for name in ("ssta", "exceed90", "mhw"):
        arr = ds[name].transpose("time", lat_dim, lon_dim).astype("float32").values
        arrays[name] = arr
        print(f"[LOAD] {name}: shape={arr.shape} dtype={arr.dtype}")

    dates = pd.DatetimeIndex(ds["time"].values)
    doy_idx = (doy365_index(dates) - 1).astype(np.int64)
    clim = ds["clim_mean"].transpose("dayofyear", lat_dim, lon_dim).astype("float32").values
    thresh = ds["thresh90"].transpose("dayofyear", lat_dim, lon_dim).astype("float32").values
    delta90 = thresh - clim
    delta90 = np.where(np.isfinite(delta90) & (delta90 > 0), delta90, np.nan).astype(np.float32)
    threshold_gap = arrays["ssta"] - delta90[doy_idx]
    arrays["threshold_gap"] = threshold_gap.astype(np.float32)
    print(f"[LOAD] threshold_gap: shape={threshold_gap.shape} dtype={threshold_gap.dtype}")
    return arrays, doy_idx


def make_ocean_points(ssta: np.ndarray) -> np.ndarray:
    ocean_mask = np.isfinite(ssta).any(axis=0)
    rows, cols = np.where(ocean_mask)
    points = np.stack([rows, cols], axis=1).astype(np.int32)
    if len(points) == 0:
        raise SystemExit("[ERROR] No ocean pixels found from finite SSTA values.")
    print(f"[OCEAN] candidate ocean pixels: {len(points)}")
    return points


def feature_names() -> np.ndarray:
    return np.array(
        [
            "ssta",
            "exceed90",
            "mhw",
            "threshold_gap",
            "recent_mhw_days",
            "recent_exceed90_days",
            "latest_ssta",
            "latest_threshold_gap",
            "ssta_trend",
            "threshold_gap_trend",
            "sin_doy",
            "cos_doy",
        ],
        dtype=object,
    )


def build_split(
    arrays: dict[str, np.ndarray],
    target_indices: np.ndarray,
    target_doy_idx: np.ndarray,
    ocean_points: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(target_indices) == 0:
        raise SystemExit("[ERROR] No valid target indices for split.")

    n = min(int(n_samples), len(target_indices) * len(ocean_points))
    chosen_times = rng.choice(target_indices, size=n, replace=True)
    point_ids = rng.integers(0, len(ocean_points), size=n)
    chosen_points = ocean_points[point_ids]

    names = feature_names()
    x = np.empty((n, cfg.HISTORY_DAYS, len(names)), dtype=np.float32)
    y = np.empty((n,), dtype=np.uint8)
    sample_points = np.empty((n, 3), dtype=np.int32)

    for i, target_t in enumerate(chosen_times):
        row, col = chosen_points[i]
        source_end = int(target_t) - cfg.LEAD_DAYS
        source_start = source_end - cfg.HISTORY_DAYS + 1

        ssta_seq = arrays["ssta"][source_start : source_end + 1, row, col]
        exceed_seq = arrays["exceed90"][source_start : source_end + 1, row, col]
        mhw_seq = arrays["mhw"][source_start : source_end + 1, row, col]
        gap_seq = arrays["threshold_gap"][source_start : source_end + 1, row, col]

        recent_mhw_days = float(np.nansum(mhw_seq > 0.5))
        recent_exceed90_days = float(np.nansum(exceed_seq > 0.5))
        latest_ssta = float(ssta_seq[-1]) if np.isfinite(ssta_seq[-1]) else 0.0
        latest_gap = float(gap_seq[-1]) if np.isfinite(gap_seq[-1]) else 0.0
        first_ssta = float(ssta_seq[0]) if np.isfinite(ssta_seq[0]) else 0.0
        first_gap = float(gap_seq[0]) if np.isfinite(gap_seq[0]) else 0.0
        ssta_trend = latest_ssta - first_ssta
        gap_trend = latest_gap - first_gap
        theta = 2.0 * np.pi * float(target_doy_idx[int(target_t)]) / 365.0
        sin_doy = float(np.sin(theta))
        cos_doy = float(np.cos(theta))

        x[i, :, 0] = np.nan_to_num(ssta_seq, nan=0.0, posinf=0.0, neginf=0.0)
        x[i, :, 1] = np.nan_to_num(exceed_seq, nan=0.0, posinf=0.0, neginf=0.0)
        x[i, :, 2] = np.nan_to_num(mhw_seq, nan=0.0, posinf=0.0, neginf=0.0)
        x[i, :, 3] = np.nan_to_num(gap_seq, nan=0.0, posinf=0.0, neginf=0.0)
        x[i, :, 4:] = np.array(
            [
                recent_mhw_days,
                recent_exceed90_days,
                latest_ssta,
                latest_gap,
                ssta_trend,
                gap_trend,
                sin_doy,
                cos_doy,
            ],
            dtype=np.float32,
        )

        y[i] = np.uint8(arrays["mhw"][int(target_t), row, col] > 0.5)
        sample_points[i] = (int(target_t), int(row), int(col))

    return x, y, sample_points


def print_split_summary(name: str, x: np.ndarray, y: np.ndarray, points: np.ndarray) -> None:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    ratio = float(y.mean()) if len(y) else 0.0
    print(
        f"[{name}] samples={len(y)} positives={positives} negatives={negatives} "
        f"positive_ratio={ratio:.6f} X_shape={x.shape} y_shape={y.shape} "
        f"points_shape={points.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build physics-enhanced point LSTM dataset.")
    parser.add_argument("--train_samples", type=int, default=cfg.DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--val_samples", type=int, default=cfg.DEFAULT_VAL_SAMPLES)
    parser.add_argument("--test_samples", type=int, default=cfg.DEFAULT_TEST_SAMPLES)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument("--dry_run", action="store_true", help="Build tiny splits and skip writing output.")
    args = parser.parse_args()

    cfg.ensure_dirs()
    if not cfg.LABEL_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.LABEL_FILE}")
    if args.dry_run:
        args.train_samples = args.val_samples = args.test_samples = cfg.DRY_RUN_SAMPLES
        print(f"[DRY RUN] Using {cfg.DRY_RUN_SAMPLES} samples per split and skipping save.")

    xr = require_xarray()
    ds = xr.open_dataset(cfg.LABEL_FILE)
    try:
        lat_dim, lon_dim = validate_dataset(ds)
        arrays, target_doy_idx = load_arrays(ds, lat_dim, lon_dim)
        ocean_points = make_ocean_points(arrays["ssta"])
        times = ds["time"].values.astype("datetime64[D]")
        splits = {
            "train": split_target_indices(times, cfg.TRAIN_START, cfg.TRAIN_END),
            "val": split_target_indices(times, cfg.VAL_START, cfg.VAL_END),
            "test": split_target_indices(times, cfg.TEST_START, cfg.TEST_END),
        }
        for split_name, indices in splits.items():
            print(
                f"[{split_name.upper()} TARGETS] count={len(indices)} "
                f"first={times[indices[0]] if len(indices) else 'NA'} "
                f"last={times[indices[-1]] if len(indices) else 'NA'}"
            )

        rng = np.random.default_rng(args.seed)
        x_train, y_train, train_points = build_split(
            arrays, splits["train"], target_doy_idx, ocean_points, args.train_samples, rng
        )
        x_val, y_val, val_points = build_split(
            arrays, splits["val"], target_doy_idx, ocean_points, args.val_samples, rng
        )
        x_test, y_test, test_points = build_split(
            arrays, splits["test"], target_doy_idx, ocean_points, args.test_samples, rng
        )

        names = feature_names()
        print_split_summary("TRAIN", x_train, y_train, train_points)
        print_split_summary("VAL", x_val, y_val, val_points)
        print_split_summary("TEST", x_test, y_test, test_points)
        print(f"feature_names: {names.tolist()}")

        if args.dry_run:
            print("[DRY RUN] Dataset construction succeeded; no file was written.")
            return

        np.savez_compressed(
            cfg.POINT_PHYSICS_DATASET_FILE,
            X_train=x_train,
            y_train=y_train,
            X_val=x_val,
            y_val=y_val,
            X_test=x_test,
            y_test=y_test,
            train_points=train_points,
            val_points=val_points,
            test_points=test_points,
            feature_names=names,
        )
        print(f"[SAVED] {cfg.POINT_PHYSICS_DATASET_FILE}")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
