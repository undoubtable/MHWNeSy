#!/usr/bin/env python
"""Build a lightweight point-wise sequence dataset for MHW forecasting.

Each sample is one ocean pixel at one target day:

    X[i] = past HISTORY_DAYS values at that pixel for ssta/exceed90/mhw
    y[i] = MHW label at target day = history_end + LEAD_DAYS

The script uses random point-time sampling so the experimental framework can be
run without materializing every pixel and every day.
"""

from __future__ import annotations

import argparse
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


def split_target_indices(times: np.ndarray, start: str, end: str) -> np.ndarray:
    """Return target-day indices whose history and lead windows are valid."""

    start64 = np.datetime64(start)
    end64 = np.datetime64(end)
    min_target = cfg.HISTORY_DAYS - 1 + cfg.LEAD_DAYS
    idx = np.arange(len(times), dtype=np.int64)
    mask = (idx >= min_target) & (times >= start64) & (times <= end64)
    return idx[mask]


def validate_dataset(ds) -> tuple[str, str]:
    missing = [name for name in cfg.INPUT_VARIABLES if name not in ds.data_vars]
    if missing:
        print("[ERROR] LABEL_FILE is missing required variables:", missing)
        print("[DATA_VARS]", list(ds.data_vars))
        print("[DIMS]", dict(ds.sizes))
        raise SystemExit(2)

    dims = ds[cfg.INPUT_VARIABLES[0]].dims
    if "time" not in dims:
        raise SystemExit(f"[ERROR] Expected a 'time' dimension, got dims={dims}")

    spatial_dims = [dim for dim in dims if dim != "time"]
    if len(spatial_dims) != 2:
        raise SystemExit(f"[ERROR] Expected two spatial dimensions, got dims={dims}")

    for name in cfg.INPUT_VARIABLES:
        if set(ds[name].dims) != set(dims):
            raise SystemExit(
                f"[ERROR] Variable {name!r} dims {ds[name].dims} do not match {dims}."
            )

    print("[DATA_VARS]", list(ds.data_vars))
    print("[DIMS]", dict(ds.sizes))
    return spatial_dims[0], spatial_dims[1]


def load_arrays(ds, lat_dim: str, lon_dim: str) -> dict[str, np.ndarray]:
    """Load required variables as time x lat x lon arrays."""

    arrays: dict[str, np.ndarray] = {}
    for name in cfg.INPUT_VARIABLES:
        arr = ds[name].transpose("time", lat_dim, lon_dim).astype("float32").values
        arrays[name] = arr
        print(f"[LOAD] {name}: shape={arr.shape}, dtype={arr.dtype}")
    return arrays


def make_ocean_points(ssta: np.ndarray) -> np.ndarray:
    """Use finite SSTA at any time as the ocean-pixel mask."""

    ocean_mask = np.isfinite(ssta).any(axis=0)
    rows, cols = np.where(ocean_mask)
    points = np.stack([rows, cols], axis=1).astype(np.int32)
    if len(points) == 0:
        raise SystemExit("[ERROR] No ocean pixels found from finite SSTA values.")
    print(f"[OCEAN] candidate ocean pixels: {len(points)}")
    return points


def build_split(
    arrays: dict[str, np.ndarray],
    target_indices: np.ndarray,
    ocean_points: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly sample point-time windows for one split."""

    if len(target_indices) == 0:
        raise SystemExit("[ERROR] No valid target indices for split.")

    n = min(int(n_samples), len(target_indices) * len(ocean_points))
    chosen_times = rng.choice(target_indices, size=n, replace=True)
    point_ids = rng.integers(0, len(ocean_points), size=n)
    chosen_points = ocean_points[point_ids]

    x = np.empty((n, cfg.HISTORY_DAYS, len(cfg.INPUT_VARIABLES)), dtype=np.float32)
    y = np.empty((n,), dtype=np.uint8)
    sample_points = np.empty((n, 3), dtype=np.int32)

    mhw = arrays["mhw"]
    for i, target_t in enumerate(chosen_times):
        row, col = chosen_points[i]
        source_end = int(target_t) - cfg.LEAD_DAYS
        source_start = source_end - cfg.HISTORY_DAYS + 1

        for feature_id, name in enumerate(cfg.INPUT_VARIABLES):
            seq = arrays[name][source_start : source_end + 1, row, col]
            x[i, :, feature_id] = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        y[i] = np.uint8(mhw[target_t, row, col] > 0.5)
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
    parser = argparse.ArgumentParser(description="Build point-wise MHW sequence dataset.")
    parser.add_argument("--train_samples", type=int, default=cfg.DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--val_samples", type=int, default=cfg.DEFAULT_VAL_SAMPLES)
    parser.add_argument("--test_samples", type=int, default=cfg.DEFAULT_TEST_SAMPLES)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Build tiny in-memory splits and skip writing the .npz file.",
    )
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
        arrays = load_arrays(ds, lat_dim, lon_dim)
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
            arrays, splits["train"], ocean_points, args.train_samples, rng
        )
        x_val, y_val, val_points = build_split(
            arrays, splits["val"], ocean_points, args.val_samples, rng
        )
        x_test, y_test, test_points = build_split(
            arrays, splits["test"], ocean_points, args.test_samples, rng
        )

        print_split_summary("TRAIN", x_train, y_train, train_points)
        print_split_summary("VAL", x_val, y_val, val_points)
        print_split_summary("TEST", x_test, y_test, test_points)

        if args.dry_run:
            print("[DRY RUN] Dataset construction succeeded; no file was written.")
            return

        np.savez_compressed(
            cfg.POINT_DATASET_FILE,
            X_train=x_train,
            y_train=y_train,
            X_val=x_val,
            y_val=y_val,
            X_test=x_test,
            y_test=y_test,
            train_points=train_points,
            val_points=val_points,
            test_points=test_points,
            feature_names=np.array(cfg.INPUT_VARIABLES),
        )
        print(f"[SAVED] {cfg.POINT_DATASET_FILE}")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
