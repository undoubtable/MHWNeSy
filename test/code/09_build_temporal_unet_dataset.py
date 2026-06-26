#!/usr/bin/env python
"""Build a full-grid Temporal U-Net dataset for MHW mask forecasting.

Each sample uses a 10-day history window ending 5 days before the target day:

    input:  t - LEAD_DAYS - HISTORY_DAYS + 1  ...  t - LEAD_DAYS
    target: t

The resulting tensor layout is channel-first for PyTorch:

    X: [N, HISTORY_DAYS * num_features, lat, lon]
    y: [N, lat, lon]

All generated artifacts stay under test/outputs/temporal_unet.
"""

from __future__ import annotations

import argparse
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


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


def split_target_indices(times: np.ndarray, start: str, end: str) -> np.ndarray:
    """Return target-day indices whose lead-separated history window is valid."""

    start64 = np.datetime64(start)
    end64 = np.datetime64(end)
    min_target = cfg.LEAD_DAYS + cfg.HISTORY_DAYS - 1
    idx = np.arange(len(times), dtype=np.int64)
    mask = (idx >= min_target) & (times >= start64) & (times <= end64)
    return idx[mask]


def validate_dataset(ds) -> tuple[str, str]:
    """Check required variables and return spatial dimension names."""

    missing = [name for name in REQUIRED_VARS if name not in ds.data_vars]
    if missing:
        print("[DATA_VARS]", list(ds.data_vars))
        print("[DIMS]", dict(ds.sizes))
        raise SystemExit(f"[ERROR] LABEL_FILE is missing required variables: {missing}")

    dims = ds[cfg.INPUT_VARIABLES[0]].dims
    if "time" not in dims:
        print("[DATA_VARS]", list(ds.data_vars))
        print("[DIMS]", dict(ds.sizes))
        raise SystemExit(f"[ERROR] Expected time dimension in {cfg.INPUT_VARIABLES[0]}, got {dims}")

    spatial_dims = [dim for dim in dims if dim != "time"]
    if len(spatial_dims) != 2:
        print("[DATA_VARS]", list(ds.data_vars))
        print("[DIMS]", dict(ds.sizes))
        raise SystemExit(f"[ERROR] Expected two spatial dims, got {dims}")

    expected = set(dims)
    for name in cfg.INPUT_VARIABLES:
        if set(ds[name].dims) != expected:
            raise SystemExit(f"[ERROR] Variable {name!r} dims {ds[name].dims} do not match {dims}")

    print("[DATA_VARS]", list(ds.data_vars))
    print("[DIMS]", dict(ds.sizes))
    return spatial_dims[0], spatial_dims[1]


def load_input_arrays(ds, lat_dim: str, lon_dim: str) -> dict[str, np.ndarray]:
    """Load model input variables as time x lat x lon float32 arrays."""

    arrays: dict[str, np.ndarray] = {}
    for name in cfg.INPUT_VARIABLES:
        arr = ds[name].transpose("time", lat_dim, lon_dim).astype("float32").values
        arrays[name] = arr
        print(f"[LOAD] {name}: shape={arr.shape} dtype={arr.dtype}")
    return arrays


def make_feature_names() -> np.ndarray:
    """Feature names follow channel order: oldest day first, variables inside day."""

    names: list[str] = []
    first_offset = -(cfg.LEAD_DAYS + cfg.HISTORY_DAYS - 1)
    for day_id in range(cfg.HISTORY_DAYS):
        offset = first_offset + day_id
        for name in cfg.INPUT_VARIABLES:
            names.append(f"t{offset}_{name}")
    return np.array(names, dtype=object)


def estimate_split_gb(n_days: int, n_channels: int, height: int, width: int, dtype: np.dtype) -> float:
    bytes_total = n_days * n_channels * height * width * np.dtype(dtype).itemsize
    return bytes_total / (1024**3)


def build_split(
    split_name: str,
    arrays: dict[str, np.ndarray],
    target_indices: np.ndarray,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize one split into dense Temporal U-Net tensors."""

    n = len(target_indices)
    height, width = arrays[cfg.INPUT_VARIABLES[0]].shape[1:]
    n_channels = cfg.HISTORY_DAYS * len(cfg.INPUT_VARIABLES)
    x = np.empty((n, n_channels, height, width), dtype=dtype)
    y = np.empty((n, height, width), dtype=np.uint8)
    mhw = arrays["mhw"]

    for sample_id, target_t in enumerate(target_indices):
        source_end = int(target_t) - cfg.LEAD_DAYS
        source_start = source_end - cfg.HISTORY_DAYS + 1
        channel_id = 0
        for source_t in range(source_start, source_end + 1):
            for name in cfg.INPUT_VARIABLES:
                frame = arrays[name][source_t]
                x[sample_id, channel_id] = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
                channel_id += 1
        y[sample_id] = np.nan_to_num(mhw[int(target_t)], nan=0.0) > 0.5

        if (sample_id + 1) % 500 == 0 or sample_id == 0 or sample_id + 1 == n:
            print(f"[{split_name}] built {sample_id + 1}/{n} samples")

    return x, y


def positive_ratio(y: np.ndarray, ocean_mask: np.ndarray) -> float:
    values = y[:, ocean_mask].astype(bool)
    return float(values.mean()) if values.size else float("nan")


def print_split_summary(name: str, x: np.ndarray, y: np.ndarray, ocean_mask: np.ndarray) -> None:
    ratio = positive_ratio(y, ocean_mask)
    print(f"{name} X shape: {x.shape}")
    print(f"{name} y shape: {y.shape}")
    print(f"{name} positive ratio on ocean pixels: {ratio:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Temporal U-Net MHW dataset.")
    parser.add_argument("--float16", action="store_true", help="Save X tensors as float16 to reduce disk/RAM.")
    parser.add_argument("--max_train_days", type=int, default=None, help="Debug cap for train target days.")
    parser.add_argument("--max_val_days", type=int, default=None, help="Debug cap for val target days.")
    parser.add_argument("--max_test_days", type=int, default=None, help="Debug cap for test target days.")
    args = parser.parse_args()

    cfg.ensure_dirs()
    if not cfg.LABEL_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.LABEL_FILE}")

    dtype = np.float16 if args.float16 else np.float32
    xr = require_xarray()
    ds = xr.open_dataset(cfg.LABEL_FILE)
    try:
        lat_dim, lon_dim = validate_dataset(ds)
        arrays = load_input_arrays(ds, lat_dim, lon_dim)
        times = ds["time"].values.astype("datetime64[D]")
        lat = ds[lat_dim].values if lat_dim in ds.coords else np.arange(arrays["ssta"].shape[1])
        lon = ds[lon_dim].values if lon_dim in ds.coords else np.arange(arrays["ssta"].shape[2])

        ocean_mask = np.isfinite(arrays["ssta"]).any(axis=0)
        ocean_pixels = int(ocean_mask.sum())
        if ocean_pixels == 0:
            raise SystemExit("[ERROR] No ocean pixels found from finite ssta values.")

        splits = {
            "train": split_target_indices(times, cfg.TRAIN_START, cfg.TRAIN_END),
            "val": split_target_indices(times, cfg.VAL_START, cfg.VAL_END),
            "test": split_target_indices(times, cfg.TEST_START, cfg.TEST_END),
        }
        caps = {
            "train": args.max_train_days,
            "val": args.max_val_days,
            "test": args.max_test_days,
        }
        for split_name, cap in caps.items():
            if cap is not None:
                splits[split_name] = splits[split_name][: int(cap)]

        n_channels = cfg.HISTORY_DAYS * len(cfg.INPUT_VARIABLES)
        height, width = ocean_mask.shape
        for split_name, indices in splits.items():
            gb = estimate_split_gb(len(indices), n_channels, height, width, np.dtype(dtype))
            first = times[indices[0]] if len(indices) else "NA"
            last = times[indices[-1]] if len(indices) else "NA"
            print(
                f"[{split_name.upper()} TARGETS] count={len(indices)} first={first} "
                f"last={last} estimated_X={gb:.3f} GiB dtype={np.dtype(dtype)}"
            )

        x_train, y_train = build_split("TRAIN", arrays, splits["train"], np.dtype(dtype))
        x_val, y_val = build_split("VAL", arrays, splits["val"], np.dtype(dtype))
        x_test, y_test = build_split("TEST", arrays, splits["test"], np.dtype(dtype))

        feature_names = make_feature_names()
        print_split_summary("X_train", x_train, y_train, ocean_mask)
        print_split_summary("X_val", x_val, y_val, ocean_mask)
        print_split_summary("X_test", x_test, y_test, ocean_mask)
        print(f"feature_names: {feature_names.tolist()}")
        print(f"ocean_pixels: {ocean_pixels}")

        # The .npz file is convenient for archival/reproducibility. The sidecar
        # .npy arrays allow training/evaluation scripts to memory-map the large
        # grid tensors instead of unpacking the whole archive into RAM first.
        np.save(cfg.TEMPORAL_UNET_X_TRAIN_FILE, x_train)
        np.save(cfg.TEMPORAL_UNET_Y_TRAIN_FILE, y_train)
        np.save(cfg.TEMPORAL_UNET_X_VAL_FILE, x_val)
        np.save(cfg.TEMPORAL_UNET_Y_VAL_FILE, y_val)
        np.save(cfg.TEMPORAL_UNET_X_TEST_FILE, x_test)
        np.save(cfg.TEMPORAL_UNET_Y_TEST_FILE, y_test)
        np.save(cfg.TEMPORAL_UNET_OCEAN_MASK_FILE, ocean_mask.astype(np.uint8))
        np.savez(
            cfg.TEMPORAL_UNET_DATASET_FILE,
            X_train=x_train,
            y_train=y_train,
            X_val=x_val,
            y_val=y_val,
            X_test=x_test,
            y_test=y_test,
            train_target_indices=splits["train"],
            val_target_indices=splits["val"],
            test_target_indices=splits["test"],
            feature_names=feature_names,
            ocean_mask=ocean_mask.astype(np.uint8),
            lat=lat,
            lon=lon,
        )
        print(f"saved path: {cfg.TEMPORAL_UNET_DATASET_FILE}")
        print(f"[SAVED] {cfg.TEMPORAL_UNET_X_TRAIN_FILE}")
        print(f"[SAVED] {cfg.TEMPORAL_UNET_Y_TRAIN_FILE}")
        print(f"[SAVED] {cfg.TEMPORAL_UNET_X_VAL_FILE}")
        print(f"[SAVED] {cfg.TEMPORAL_UNET_Y_VAL_FILE}")
        print(f"[SAVED] {cfg.TEMPORAL_UNET_X_TEST_FILE}")
        print(f"[SAVED] {cfg.TEMPORAL_UNET_Y_TEST_FILE}")
        print(f"[SAVED] {cfg.TEMPORAL_UNET_OCEAN_MASK_FILE}")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
