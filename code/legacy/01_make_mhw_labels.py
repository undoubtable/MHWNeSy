# -*- coding: utf-8 -*-
"""
Strict Hobday-style Marine Heatwave label generation.

Literature-aligned settings:
    - Seasonally varying daily climatology
    - 90th percentile threshold
    - 11-day percentile window: day j ± 5 days
    - 31-day circular smoothing for climatology and threshold
    - Minimum duration: 5 days
    - Join events separated by short gaps <= 2 days

This follows the commonly used Hobday et al. MHW definition and the
marineHeatWaves.py default numerical settings.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def doy365_index(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Convert dates to no-leap day-of-year in [1, 365].
    Feb 29 is mapped to Feb 28; days after Feb 29 in leap years shift by -1.
    """
    doy = dates.dayofyear.to_numpy().astype(np.int16)
    month = dates.month.to_numpy()
    day = dates.day.to_numpy()
    is_leap = dates.is_leap_year

    is_feb29 = (month == 2) & (day == 29)
    after_feb29 = is_leap & ((month > 2) | is_feb29)

    doy = doy - after_feb29.astype(np.int16)
    doy[is_feb29] = 59
    doy = np.clip(doy, 1, 365)

    return doy.astype(np.int16)


def circular_doy_distance(a: np.ndarray, b: int) -> np.ndarray:
    """
    Circular distance on 365-day calendar.
    """
    return np.abs(((a - b + 182) % 365) - 182)


def smooth_circular_dayofyear(arr: np.ndarray, width: int = 31) -> np.ndarray:
    """
    Circular moving-average smoothing along dayofyear dimension.

    arr shape:
        [365, lat, lon]
    """
    if width <= 1:
        return arr.astype(np.float32)

    if width % 2 == 0:
        raise ValueError("Smoothing width should be odd, e.g. 31.")

    pad = width // 2
    padded = np.concatenate([arr[-pad:], arr, arr[:pad]], axis=0)

    out = np.full_like(arr, np.nan, dtype=np.float32)

    for d in tqdm(range(365), desc=f"{width}-day circular smoothing"):
        out[d] = np.nanmean(padded[d:d + width], axis=0).astype(np.float32)

    return out


def fill_short_false_gaps(x: np.ndarray, max_gap: int = 2) -> np.ndarray:
    """
    Join warm events separated by short gaps <= max_gap.
    Gap days are included in the final event mask.
    """
    y = x.copy().astype(bool)
    n = len(y)
    i = 0

    while i < n:
        if y[i]:
            i += 1
            continue

        start = i
        while i < n and not y[i]:
            i += 1
        end = i

        gap_len = end - start
        left_true = start > 0 and y[start - 1]
        right_true = end < n and y[end]

        if left_true and right_true and gap_len <= max_gap:
            y[start:end] = True

    return y


def mark_persistent_runs(x: np.ndarray, min_duration: int = 5) -> np.ndarray:
    """
    Keep only True runs with length >= min_duration.
    """
    y = np.zeros_like(x, dtype=bool)
    n = len(x)
    i = 0

    while i < n:
        if not x[i]:
            i += 1
            continue

        start = i
        while i < n and x[i]:
            i += 1
        end = i

        if end - start >= min_duration:
            y[start:end] = True

    return y


def make_mhw_mask(exceed: np.ndarray, min_duration: int, max_gap: int) -> np.ndarray:
    """
    Gridpoint-wise MHW event detection.

    exceed:
        bool array [time, lat, lon]
    """
    T, H, W = exceed.shape
    flat = exceed.reshape(T, H * W)
    out = np.zeros_like(flat, dtype=np.uint8)

    for p in tqdm(range(H * W), desc="MHW persistence and gap-joining"):
        x = flat[:, p].astype(bool)

        if not np.any(x):
            continue

        x_joined = fill_short_false_gaps(x, max_gap=max_gap)
        y = mark_persistent_runs(x_joined, min_duration=min_duration)

        out[:, p] = y.astype(np.uint8)

    return out.reshape(T, H, W)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_nc", type=str, default=str(cfg.RAW_NC))
    parser.add_argument("--out_nc", type=str, default=str(cfg.LABEL_NC))

    parser.add_argument("--clim_start", type=int, default=1982)
    parser.add_argument("--clim_end", type=int, default=2011)

    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--window_half_width", type=int, default=5)
    parser.add_argument("--smooth_width", type=int, default=31)

    parser.add_argument("--min_duration", type=int, default=5)
    parser.add_argument("--max_gap", type=int, default=2)

    args = parser.parse_args()

    raw_nc = Path(args.raw_nc)
    out_nc = Path(args.out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    print("[LOAD]", raw_nc)
    ds = xr.open_dataset(raw_nc)
    ds = ds.sortby("lat").sortby("lon")

    if "sst" not in ds:
        raise ValueError("Input NetCDF must contain variable 'sst'.")

    print("[DATASET]")
    print(ds)

    dates = pd.DatetimeIndex(ds["time"].values)
    years = dates.year.to_numpy()
    doy365 = doy365_index(dates)

    print("[LOAD SST INTO MEMORY]")
    sst = ds["sst"].astype("float32").values
    time = ds["time"].values
    lat = ds["lat"].values
    lon = ds["lon"].values
    ds.close()

    T, H, W = sst.shape
    print("[SHAPE]", sst.shape)

    base_mask = (years >= args.clim_start) & (years <= args.clim_end)
    base_sst = sst[base_mask]
    base_doy = doy365[base_mask]

    print("[BASELINE PERIOD]", args.clim_start, args.clim_end)
    print("[BASELINE DAYS]", int(base_mask.sum()))

    clim_mean_raw = np.full((365, H, W), np.nan, dtype=np.float32)
    thresh_raw = np.full((365, H, W), np.nan, dtype=np.float32)

    print("[DAILY CLIMATOLOGY AND THRESHOLD]")
    print(f"percentile = {args.percentile}")
    print(f"calendar-day window = ±{args.window_half_width} days")

    for d in tqdm(range(1, 366), desc="Daily raw climatology"):
        dist = circular_doy_distance(base_doy, d)
        idx = dist <= args.window_half_width

        arr = base_sst[idx]

        clim_mean_raw[d - 1] = np.nanmean(arr, axis=0).astype(np.float32)
        thresh_raw[d - 1] = np.nanpercentile(arr, args.percentile, axis=0).astype(np.float32)

    print("[SMOOTH CLIMATOLOGY]")
    clim_mean = smooth_circular_dayofyear(clim_mean_raw, width=args.smooth_width)

    print("[SMOOTH THRESHOLD]")
    thresh90 = smooth_circular_dayofyear(thresh_raw, width=args.smooth_width)

    print("[MAP DAILY CLIMATOLOGY TO FULL TIME SERIES]")
    clim_t = clim_mean[doy365 - 1]
    thresh_t = thresh90[doy365 - 1]

    ssta = (sst - clim_t).astype(np.float32)

    exceed90 = (
        (sst > thresh_t)
        & np.isfinite(sst)
        & np.isfinite(thresh_t)
    ).astype(np.uint8)

    print("[MAKE STRICT MHW MASK]")
    mhw = make_mhw_mask(
        exceed=exceed90.astype(bool),
        min_duration=args.min_duration,
        max_gap=args.max_gap,
    )

    out = xr.Dataset(
        data_vars=dict(
            ssta=(("time", "lat", "lon"), ssta.astype(np.float32)),
            exceed90=(("time", "lat", "lon"), exceed90.astype(np.uint8)),
            mhw=(("time", "lat", "lon"), mhw.astype(np.uint8)),
            clim_mean=(("dayofyear", "lat", "lon"), clim_mean.astype(np.float32)),
            thresh90=(("dayofyear", "lat", "lon"), thresh90.astype(np.float32)),
            clim_mean_raw=(("dayofyear", "lat", "lon"), clim_mean_raw.astype(np.float32)),
            thresh90_raw=(("dayofyear", "lat", "lon"), thresh_raw.astype(np.float32)),
        ),
        coords=dict(
            time=time,
            lat=lat,
            lon=lon,
            dayofyear=np.arange(1, 366, dtype=np.int16),
        ),
        attrs=dict(
            title="Strict Hobday-style South China Sea OISST marine heatwave labels",
            definition=(
                f"SST > seasonally varying {args.percentile}th percentile threshold; "
                f"daily threshold estimated using ±{args.window_half_width}-day window; "
                f"climatology and threshold smoothed with {args.smooth_width}-day circular moving average; "
                f"minimum duration {args.min_duration} days; "
                f"events separated by gaps <= {args.max_gap} days are joined."
            ),
            climatology=f"{args.clim_start}-{args.clim_end}",
            method_reference="Hobday et al. 2016; marineHeatWaves.py-style defaults",
        ),
    )

    encoding = {
        "ssta": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "exceed90": {"zlib": True, "complevel": 4, "dtype": "uint8"},
        "mhw": {"zlib": True, "complevel": 4, "dtype": "uint8"},
        "clim_mean": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "thresh90": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "clim_mean_raw": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "thresh90_raw": {"zlib": True, "complevel": 4, "dtype": "float32"},
    }

    print("[SAVE]", out_nc)
    out.to_netcdf(out_nc, encoding=encoding)

    print("[DONE]", out_nc)
    print(out)
    print("mhw ratio:", float(out["mhw"].mean().values))
    print("exceed90 ratio:", float(out["exceed90"].mean().values))


if __name__ == "__main__":
    main()
