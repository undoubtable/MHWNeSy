#!/usr/bin/env python
"""Run point-wise LSTM prediction over the full test-period grid.

The script streams one target day at a time and batches ocean pixels, so it does
not materialize a giant [days, pixels, history, features] tensor in memory.
"""

from __future__ import annotations

import argparse
import csv
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


def require_torch():
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[MISSING DEPENDENCY] torch is required for full-grid LSTM prediction.\n"
            "Install project dependencies first: pip install -r requirements.txt"
        ) from exc
    return torch, nn


def load_checkpoint(torch):
    try:
        return torch.load(cfg.POINT_LSTM_MODEL_FILE, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(cfg.POINT_LSTM_MODEL_FILE, map_location="cpu")


def load_best_threshold(default: float = 0.5) -> tuple[float, str]:
    """Read the F1-best threshold from point_threshold_sweep.csv when present."""

    if not cfg.POINT_THRESHOLD_SWEEP_FILE.exists():
        return default, "default"

    best_threshold = default
    best_f1 = -1.0
    best_precision = -1.0
    with cfg.POINT_THRESHOLD_SWEEP_FILE.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            threshold = float(row["threshold"])
            f1 = float(row["f1"])
            precision = float(row["precision"])
            if (f1, precision, threshold) > (best_f1, best_precision, best_threshold):
                best_threshold = threshold
                best_f1 = f1
                best_precision = precision
    return best_threshold, str(cfg.POINT_THRESHOLD_SWEEP_FILE)


def split_target_indices(times: np.ndarray) -> np.ndarray:
    """Return test target-day indices with valid history and lead windows."""

    start64 = np.datetime64(cfg.TEST_START)
    end64 = np.datetime64(cfg.TEST_END)
    min_target = cfg.HISTORY_DAYS - 1 + cfg.LEAD_DAYS
    idx = np.arange(len(times), dtype=np.int64)
    mask = (idx >= min_target) & (times >= start64) & (times <= end64)
    return idx[mask]


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


def build_ocean_mask(ds, lat_dim: str, lon_dim: str) -> np.ndarray:
    """Use finite SSTA at any time as the full-grid ocean mask."""

    ssta = ds["ssta"].transpose("time", lat_dim, lon_dim)
    ocean_mask = np.isfinite(ssta.values).any(axis=0)
    if not ocean_mask.any():
        raise SystemExit("[ERROR] No ocean pixels found from finite SSTA values.")
    return ocean_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict point LSTM over full test grid.")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument(
        "--max_days",
        type=int,
        default=None,
        help="Optional debug limit for the number of target days to predict.",
    )
    args = parser.parse_args()

    cfg.ensure_dirs()
    if not cfg.LABEL_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.LABEL_FILE}")
    if not cfg.POINT_LSTM_MODEL_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.POINT_LSTM_MODEL_FILE}")

    xr = require_xarray()
    torch, nn = require_torch()
    threshold, threshold_source = load_best_threshold()
    print(f"[THRESHOLD] threshold={threshold:.2f} source={threshold_source}")

    checkpoint = load_checkpoint(torch)

    class PointLSTM(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (hidden, _) = self.lstm(x)
            return self.head(hidden[-1]).squeeze(-1)

    model = PointLSTM(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ds = xr.open_dataset(cfg.LABEL_FILE)
    try:
        lat_dim, lon_dim = validate_dataset(ds)
        times = ds["time"].values.astype("datetime64[D]")
        target_time_indices = split_target_indices(times)
        if args.max_days is not None:
            target_time_indices = target_time_indices[: args.max_days]
        if len(target_time_indices) == 0:
            raise SystemExit("[ERROR] No TEST_PERIOD target days found.")

        lat = ds[lat_dim].values
        lon = ds[lon_dim].values
        h, w = len(lat), len(lon)
        ocean_mask = build_ocean_mask(ds, lat_dim, lon_dim)
        ocean_rows, ocean_cols = np.where(ocean_mask)
        n_ocean = len(ocean_rows)
        print(f"[GRID] target_days={len(target_time_indices)} shape=({h}, {w}) ocean_pixels={n_ocean}")

        prob_out = np.lib.format.open_memmap(
            cfg.FULL_GRID_TEST_PROB_FILE,
            mode="w+",
            dtype="float32",
            shape=(len(target_time_indices), h, w),
        )
        pred_out = np.lib.format.open_memmap(
            cfg.FULL_GRID_TEST_PRED_FILE,
            mode="w+",
            dtype="uint8",
            shape=(len(target_time_indices), h, w),
        )
        prob_out[:] = np.nan
        pred_out[:] = 0

        total_positive = 0
        total_ocean_predictions = len(target_time_indices) * n_ocean
        with torch.no_grad():
            for out_i, target_t in enumerate(target_time_indices):
                source_end = int(target_t) - cfg.LEAD_DAYS
                source_start = source_end - cfg.HISTORY_DAYS + 1

                feature_windows = []
                for name in cfg.INPUT_VARIABLES:
                    arr = (
                        ds[name]
                        .isel(time=slice(source_start, source_end + 1))
                        .transpose("time", lat_dim, lon_dim)
                        .astype("float32")
                        .values
                    )
                    feature_windows.append(arr[:, ocean_rows, ocean_cols])

                x_all = np.stack(feature_windows, axis=-1).transpose(1, 0, 2)
                x_all = np.nan_to_num(x_all, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                day_prob = np.empty((n_ocean,), dtype=np.float32)
                for start in range(0, n_ocean, args.batch_size):
                    end = min(start + args.batch_size, n_ocean)
                    xb = torch.from_numpy(x_all[start:end]).to(device)
                    logits = model(xb)
                    day_prob[start:end] = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

                prob_grid = np.full((h, w), np.nan, dtype=np.float32)
                pred_grid = np.zeros((h, w), dtype=np.uint8)
                prob_grid[ocean_rows, ocean_cols] = day_prob
                pred_grid[ocean_rows, ocean_cols] = (day_prob >= threshold).astype(np.uint8)
                total_positive += int(pred_grid[ocean_rows, ocean_cols].sum())
                prob_out[out_i] = prob_grid
                pred_out[out_i] = pred_grid

                if (out_i + 1) % 100 == 0 or out_i == 0 or out_i + 1 == len(target_time_indices):
                    print(
                        f"[PREDICT] {out_i + 1}/{len(target_time_indices)} "
                        f"target_t={int(target_t)} date={times[target_t]}"
                    )

        prob_out.flush()
        pred_out.flush()
        positive_ratio = total_positive / max(total_ocean_predictions, 1)

        np.savez_compressed(
            cfg.FULL_GRID_TEST_META_FILE,
            target_time_indices=target_time_indices.astype(np.int64),
            target_times=times[target_time_indices],
            lat=lat,
            lon=lon,
            threshold=np.array(threshold, dtype=np.float32),
            ocean_mask=ocean_mask.astype(np.uint8),
        )
        print(f"[SAVED] {cfg.FULL_GRID_TEST_PROB_FILE} shape={prob_out.shape}")
        print(f"[SAVED] {cfg.FULL_GRID_TEST_PRED_FILE} shape={pred_out.shape}")
        print(f"[SAVED] {cfg.FULL_GRID_TEST_META_FILE}")
        print(f"[SUMMARY] threshold={threshold:.2f} predicted_positive_ratio={positive_ratio:.6f}")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
