# -*- coding: utf-8 -*-
"""
Build supervised forecasting arrays for baseline models.

Task:
    Input  : past H days of SSTA, shape [H, lat, lon]
    Target : MHW mask at lead day L, shape [lat, lon]

Default:
    history = 10 days
    lead    = 5 days

Outputs:
    outputs/03_forecast_dataset_h10_l5/
        X_train.npy, y_train.npy, target_dates_train.npy
        X_val.npy,   y_val.npy,   target_dates_val.npy
        X_test.npy,  y_test.npy,  target_dates_test.npy
        norm_stats.json
        meta.json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_nc", type=str, default=str(cfg.LABEL_NC))
    parser.add_argument("--out_dir", type=str, default=str(cfg.FORECAST_DIR))
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
    dates = pd.DatetimeIndex(ds["time"].values)
    lat = ds["lat"].values
    lon = ds["lon"].values
    ds.close()

    T, H, W = ssta.shape
    print("[SHAPE]", ssta.shape)

    # Normalization using train-period times only.
    train_time_mask = dates.year <= 2015
    mean = float(np.nanmean(ssta[train_time_mask]))
    std = float(np.nanstd(ssta[train_time_mask]) + 1e-6)
    print("[NORM] mean:", mean, "std:", std)

    ssta = (ssta - mean) / std
    ssta = np.nan_to_num(ssta, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    records = {"train": [], "val": [], "test": []}

    # t is the last input day.
    for t in range(args.history - 1, T - args.lead, args.stride):
        target_t = t + args.lead
        target_date = dates[target_t]
        sp = split_name(target_date.year)
        records[sp].append((t, target_t, str(target_date.date())))

    print("[NUM SAMPLES]", {k: len(v) for k, v in records.items()})

    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    for sp, recs in records.items():
        n = len(recs)
        X_path = out_dir / f"X_{sp}.npy"
        y_path = out_dir / f"y_{sp}.npy"
        d_path = out_dir / f"target_dates_{sp}.npy"

        X = np.lib.format.open_memmap(
            X_path,
            mode="w+",
            dtype=np_dtype,
            shape=(n, args.history, H, W),
        )
        y = np.lib.format.open_memmap(
            y_path,
            mode="w+",
            dtype=np.uint8,
            shape=(n, H, W),
        )

        target_dates = []

        for i, (t, target_t, target_date) in enumerate(tqdm(recs, desc=f"Writing {sp}")):
            X[i] = ssta[t - args.history + 1:t + 1].astype(np_dtype)
            y[i] = mhw[target_t].astype(np.uint8)
            target_dates.append(target_date)

        X.flush()
        y.flush()
        np.save(d_path, np.array(target_dates, dtype="U10"))

    meta = {
        "history": args.history,
        "lead": args.lead,
        "stride": args.stride,
        "input": "past SSTA",
        "target": "future MHW mask",
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
        json.dumps({"mean": mean, "std": std}, indent=2),
        encoding="utf-8",
    )

    np.save(out_dir / "lat.npy", lat)
    np.save(out_dir / "lon.npy", lon)

    print("[DONE]", out_dir)


if __name__ == "__main__":
    main()
