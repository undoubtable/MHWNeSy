# -*- coding: utf-8 -*-
"""
Create Hobday-style Marine Heatwave labels from daily OISST.

Input:
    data/oisst_scs_1982_2023.nc

Output:
    outputs/01_mhw_labels/mhw_labels_strict_hobday_1982_2023.nc

Main variables saved:
    ssta(time, lat, lon)          float32
    exceed90(time, lat, lon)      uint8
    mhw(time, lat, lon)           uint8
    clim_mean(dayofyear, lat, lon) float32
    thresh90(dayofyear, lat, lon)  float32

Definition in this first version:
    SST > local 90th percentile threshold
    and warm event persists for at least 5 days.
    Short cold gaps <= 2 days inside warm events are bridged.
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
    Return no-leap day-of-year in [1, 365].
    Feb 29 is mapped to Feb 28.
    Days after Feb 29 in leap years are shifted by -1.
    """
    doy = dates.dayofyear.to_numpy().astype(np.int16)
    month = dates.month.to_numpy()
    day = dates.day.to_numpy()
    is_leap = dates.is_leap_year

    is_feb29 = (month == 2) & (day == 29)
    after_feb29 = is_leap & ((month > 2) | is_feb29)
    doy = doy - after_feb29.astype(np.int16)
    doy = np.clip(doy, 1, 365)
    return doy.astype(np.int16)


def circular_doy_distance(a: np.ndarray, b: int) -> np.ndarray:
    """
    Circular distance on a 365-day calendar.
    """
    return np.abs(((a - b + 182) % 365) - 182)


def fill_short_false_gaps(x: np.ndarray, max_gap: int = 2) -> np.ndarray:
    """
    Fill False gaps of length <= max_gap between True segments.
    """
    if max_gap <= 0:
        return x

    y = x.copy()
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
    Mark True runs with length >= min_duration.
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
    Convert exceedance mask to MHW mask gridpoint by gridpoint.
    exceed: bool array, shape [time, lat, lon]
    """
    T, H, W = exceed.shape
    flat = exceed.reshape(T, H * W)
    out = np.zeros_like(flat, dtype=np.uint8)

    for p in tqdm(range(H * W), desc="MHW persistence check"):
        x = flat[:, p]
        if not np.any(x):
            continue
        x2 = fill_short_false_gaps(x.astype(bool), max_gap=max_gap)
        y = mark_persistent_runs(x2, min_duration=min_duration)
        out[:, p] = y.astype(np.uint8)

    return out.reshape(T, H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_nc", type=str, default=str(cfg.RAW_NC))
    parser.add_argument("--out_nc", type=str, default=str(cfg.LABEL_NC))
    parser.add_argument("--clim_start", type=int, default=1982)
    parser.add_argument("--clim_end", type=int, default=2011)
    parser.add_argument("--window", type=int, default=5, help="Half-window around each calendar day.")
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--min_duration", type=int, default=5)
    parser.add_argument("--max_gap", type=int, default=2)
    args = parser.parse_args()

    raw_nc = Path(args.raw_nc)
    out_nc = Path(args.out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    print("[LOAD]", raw_nc)
    ds = xr.open_dataset(raw_nc)
    if "sst" not in ds:
        raise ValueError("Input NetCDF must contain variable 'sst'.")

    # Ensure ascending coordinates.
    ds = ds.sortby("lat").sortby("lon")

    dates = pd.DatetimeIndex(ds["time"].values)
    years = dates.year.to_numpy()
    doy365 = doy365_index(dates)

    print("[DATASET]")
    print(ds)

    print("[LOAD SST INTO MEMORY]")
    sst = ds["sst"].astype("float32").values
    lat = ds["lat"].values
    lon = ds["lon"].values
    time = ds["time"].values
    ds.close()

    T, H, W = sst.shape
    print("[SHAPE]", sst.shape)

    base_mask = (years >= args.clim_start) & (years <= args.clim_end)
    base_doy = doy365[base_mask]
    base_sst = sst[base_mask]

    clim_mean = np.full((365, H, W), np.nan, dtype=np.float32)
    thresh90 = np.full((365, H, W), np.nan, dtype=np.float32)

    print("[CLIMATOLOGY] years:", args.clim_start, args.clim_end)
    for d in tqdm(range(1, 366), desc="Daily climatology and threshold"):
        dist = circular_doy_distance(base_doy, d)
        idx = dist <= args.window
        arr = base_sst[idx]

        clim_mean[d - 1] = np.nanmean(arr, axis=0).astype(np.float32)
        thresh90[d - 1] = np.nanpercentile(arr, args.percentile, axis=0).astype(np.float32)

    print("[MAP THRESHOLD TO TIME]")
    clim_t = clim_mean[doy365 - 1]
    thr_t = thresh90[doy365 - 1]

    ssta = (sst - clim_t).astype(np.float32)
    exceed90 = ((sst > thr_t) & np.isfinite(sst) & np.isfinite(thr_t)).astype(np.uint8)

    print("[MAKE MHW MASK]")
    mhw = make_mhw_mask(
        exceed=exceed90.astype(bool),
        min_duration=args.min_duration,
        max_gap=args.max_gap,
    )

    out = xr.Dataset(
        data_vars=dict(
            ssta=(("time", "lat", "lon"), ssta),
            exceed90=(("time", "lat", "lon"), exceed90),
            mhw=(("time", "lat", "lon"), mhw),
            clim_mean=(("dayofyear", "lat", "lon"), clim_mean),
            thresh90=(("dayofyear", "lat", "lon"), thresh90),
        ),
        coords=dict(
            time=time,
            lat=lat,
            lon=lon,
            dayofyear=np.arange(1, 366, dtype=np.int16),
        ),
        attrs=dict(
            title="South China Sea OISST marine heatwave labels",
            definition=(
                f"SST > local {args.percentile}th percentile threshold, "
                f"minimum duration {args.min_duration} days, "
                f"gap bridging <= {args.max_gap} days."
            ),
            climatology=f"{args.clim_start}-{args.clim_end}",
        ),
    )

    encoding = {
        "ssta": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "exceed90": {"zlib": True, "complevel": 4, "dtype": "uint8"},
        "mhw": {"zlib": True, "complevel": 4, "dtype": "uint8"},
        "clim_mean": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "thresh90": {"zlib": True, "complevel": 4, "dtype": "float32"},
    }

    print("[SAVE]", out_nc)
    out.to_netcdf(out_nc, encoding=encoding)
    print("[DONE]", out_nc)
    print(out)


if __name__ == "__main__":
    main()
