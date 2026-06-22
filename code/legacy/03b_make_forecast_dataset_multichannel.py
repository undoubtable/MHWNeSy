# -*- coding: utf-8 -*-
"""
Build persistence-aware multichannel forecasting arrays.

Input channels are the past history days of:
    1. SSTA
    2. strict MHW mask
    3. exceed90 mask
    4. threshold_gap = SSTA - (thresh90 - clim_mean)

Output shape:
    X: [N, history * 4, lat, lon]
    y: [N, lat, lon]

Outputs:
    outputs/03b_forecast_dataset_multichannel_h10_l5/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def split_name(year: int) -> str:
    if year <= 2015:
        return "train"
    if year <= 2018:
        return "val"
    return "test"


def doy365_index(dates: pd.DatetimeIndex) -> np.ndarray:
    doy = dates.dayofyear.to_numpy().copy()
    leap_after_feb = dates.is_leap_year & (dates.month > 2)
    doy[leap_after_feb] -= 1
    doy[(dates.month == 2) & (dates.day == 29)] = 59
    return doy.astype(np.int64)


def normalize_train_period(arr, train_time_mask):
    mean = float(np.nanmean(arr[train_time_mask]))
    std = float(np.nanstd(arr[train_time_mask]) + 1e-6)
    out = (arr - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out, {"mean": mean, "std": std}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_nc", type=str, default=str(cfg.LABEL_NC))
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "03b_forecast_dataset_multichannel_h10_l5"),
    )
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--lead", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[LOAD]", args.label_nc)
    ds = xr.open_dataset(args.label_nc)
    ssta = ds["ssta"].astype("float32").values
    mhw = ds["mhw"].astype("uint8").values
    exceed90 = ds["exceed90"].astype("uint8").values
    dates = pd.DatetimeIndex(ds["time"].values)
    lat = ds["lat"].values
    lon = ds["lon"].values

    doy365 = doy365_index(dates)
    threshold_anom = (
        ds["thresh90"].astype("float32").values
        - ds["clim_mean"].astype("float32").values
    ).astype(np.float32)
    ds.close()

    T, H, W = ssta.shape
    print("[SHAPE]", ssta.shape)

    threshold_gap = np.empty_like(ssta, dtype=np.float32)
    for d in range(1, 366):
        idx = doy365 == d
        if np.any(idx):
            threshold_gap[idx] = ssta[idx] - threshold_anom[d - 1]

    train_time_mask = dates.year <= 2015
    ssta, ssta_stats = normalize_train_period(ssta, train_time_mask)
    threshold_gap, gap_stats = normalize_train_period(threshold_gap, train_time_mask)
    mhw = np.nan_to_num(mhw, nan=0).astype(np.float32)
    exceed90 = np.nan_to_num(exceed90, nan=0).astype(np.float32)

    features = {
        "ssta": ssta,
        "mhw": mhw,
        "exceed90": exceed90,
        "threshold_gap": threshold_gap,
    }
    feature_names = list(features.keys())

    records = {"train": [], "val": [], "test": []}
    for t in range(args.history - 1, T - args.lead, args.stride):
        target_t = t + args.lead
        target_date = dates[target_t]
        sp = split_name(target_date.year)
        records[sp].append((t, target_t, str(target_date.date())))

    print("[NUM SAMPLES]", {k: len(v) for k, v in records.items()})

    np_dtype = np.float16 if args.dtype == "float16" else np.float32
    n_channels = args.history * len(feature_names)
    channel_names = [
        f"{name}_t-{args.history - 1 - i}"
        for name in feature_names
        for i in range(args.history)
    ]

    for sp, recs in records.items():
        n = len(recs)
        X_path = out_dir / f"X_{sp}.npy"
        y_path = out_dir / f"y_{sp}.npy"
        d_path = out_dir / f"target_dates_{sp}.npy"

        X = np.lib.format.open_memmap(
            X_path,
            mode="w+",
            dtype=np_dtype,
            shape=(n, n_channels, H, W),
        )
        y = np.lib.format.open_memmap(
            y_path,
            mode="w+",
            dtype=np.uint8,
            shape=(n, H, W),
        )

        target_dates = []
        for i, (t, target_t, target_date) in enumerate(tqdm(recs, desc=f"Writing {sp}")):
            start = t - args.history + 1
            X[i] = np.concatenate(
                [features[name][start:t + 1] for name in feature_names],
                axis=0,
            ).astype(np_dtype)
            y[i] = mhw[target_t].astype(np.uint8)
            target_dates.append(target_date)

        X.flush()
        y.flush()
        np.save(d_path, np.array(target_dates, dtype="U10"))

    meta = {
        "history": args.history,
        "lead": args.lead,
        "stride": args.stride,
        "input": "past SSTA, MHW mask, exceed90 mask, and threshold_gap",
        "target": "future MHW mask",
        "feature_names": feature_names,
        "channel_order": "feature blocks, each containing oldest-to-newest history days",
        "channel_names": channel_names,
        "n_channels": int(n_channels),
        "split": {
            "train": "target year <= 2015",
            "val": "2016 <= target year <= 2018",
            "test": "target year >= 2019",
        },
        "lat_size": int(H),
        "lon_size": int(W),
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "norm_stats.json").write_text(
        json.dumps(
            {
                "ssta": ssta_stats,
                "threshold_gap": gap_stats,
                "mhw": {"mean": 0.0, "std": 1.0, "note": "binary mask, not standardized"},
                "exceed90": {"mean": 0.0, "std": 1.0, "note": "binary mask, not standardized"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    np.save(out_dir / "lat.npy", lat)
    np.save(out_dir / "lon.npy", lon)

    print("[DONE]", out_dir)


if __name__ == "__main__":
    main()
