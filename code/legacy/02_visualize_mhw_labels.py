# -*- coding: utf-8 -*-
"""
Visualize generated MHW labels.

Outputs:
    outputs/02_label_visualization/mhw_map_YYYY-MM-DD.png
    outputs/02_label_visualization/mhw_area_YEAR.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_nc", type=str, default=str(cfg.LABEL_NC))
    parser.add_argument("--date", type=str, default="2023-07-01")
    parser.add_argument("--year", type=int, default=2023)
    args = parser.parse_args()

    out_dir = cfg.FIGURE_DIR / "02_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.label_nc)
    print(ds)

    one = ds.sel(time=args.date, method="nearest")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    one["ssta"].plot(ax=axes[0], cmap="RdBu_r")
    axes[0].set_title(f"SSTA {str(one.time.values)[:10]}")

    one["exceed90"].plot(ax=axes[1], cmap="gray_r", add_colorbar=False)
    axes[1].set_title("SST > 90th percentile")

    one["mhw"].plot(ax=axes[2], cmap="gray_r", add_colorbar=False)
    axes[2].set_title("MHW mask")

    fig_file = out_dir / f"mhw_map_{args.date}.png"
    plt.savefig(fig_file, dpi=200)
    plt.close()
    print("[SAVE]", fig_file)

    sub = ds.sel(time=slice(f"{args.year}-01-01", f"{args.year}-12-31"))
    # Rough area proxy: number of ocean pixels with MHW.
    mhw_area_px = sub["mhw"].sum(dim=("lat", "lon")).values
    dates = pd.DatetimeIndex(sub["time"].values)

    plt.figure(figsize=(12, 4))
    plt.plot(dates, mhw_area_px)
    plt.title(f"MHW area proxy over South China Sea, {args.year}")
    plt.ylabel("MHW pixels")
    plt.xlabel("Date")
    plt.tight_layout()

    fig_file = out_dir / f"mhw_area_{args.year}.png"
    plt.savefig(fig_file, dpi=200)
    plt.close()
    print("[SAVE]", fig_file)

    ds.close()


if __name__ == "__main__":
    main()
