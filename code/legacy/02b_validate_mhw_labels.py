# -*- coding: utf-8 -*-
"""
Validate strict Hobday-style MHW labels.

Purpose:
    This script does not train models.
    It produces paper-ready statistics and figures for checking whether
    the generated MHW labels are reasonable and comparable with MHW literature.

Input:
    outputs/01_mhw_labels/mhw_labels_1982_2023.nc

Outputs:
    outputs/02_label_validation/
        annual_mhw_days.csv
        annual_mhw_intensity.csv
        seasonal_mhw_days.csv
        label_summary.json
        annual_mhw_days_trend.png
        annual_mhw_intensity_trend.png
        seasonal_mhw_days_map.png
        mhw_days_mean_map_1982_2023.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def get_output_dir():
    """
    Compatible with both old and new 00_config.py.
    """
    if hasattr(cfg, "OUTPUT_DIR"):
        out_dir = cfg.OUTPUT_DIR / "02_label_validation"
    else:
        out_dir = cfg.ROOT / "outputs" / "02_label_validation"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def linear_trend(x, y):
    """
    Return slope per year and intercept.
    Ignore NaN values.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan, np.nan

    slope, intercept = np.polyfit(x[valid], y[valid], deg=1)
    return float(slope), float(intercept)


def make_ocean_mask(ssta):
    """
    Ocean mask:
        grid cells with at least one finite SSTA value.
    """
    return np.isfinite(ssta).any(axis=0)


def annual_statistics(ds, out_dir):
    """
    Compute annual MHW days and intensity statistics.
    """
    print("[LOAD ARRAYS]")
    ssta = ds["ssta"].astype("float32").values
    mhw = ds["mhw"].astype("uint8").values
    exceed90 = ds["exceed90"].astype("uint8").values

    dates = pd.DatetimeIndex(ds["time"].values)
    years = np.arange(dates.year.min(), dates.year.max() + 1)

    ocean_mask = make_ocean_mask(ssta)
    ocean_pixels = int(ocean_mask.sum())
    total_pixels = int(ocean_mask.size)

    print("[OCEAN MASK]")
    print("ocean pixels:", ocean_pixels)
    print("total pixels:", total_pixels)

    annual_days_rows = []
    annual_intensity_rows = []

    annual_mhw_days_maps = []
    annual_cum_intensity_maps = []

    for year in years:
        idx = dates.year == year
        mhw_y = mhw[idx]
        ssta_y = ssta[idx]

        # MHW days per grid cell in this year.
        mhw_days_map = mhw_y.sum(axis=0).astype(np.float32)

        # Cumulative intensity: sum of positive SSTA during MHW days.
        intensity = np.where(mhw_y.astype(bool), ssta_y, np.nan)
        cum_intensity_map = np.nansum(intensity, axis=0).astype(np.float32)

        # Mean intensity over MHW pixels/days.
        mean_intensity = float(np.nanmean(intensity[:, ocean_mask]))

        # Maximum intensity over MHW pixels/days.
        max_intensity = float(np.nanmax(intensity[:, ocean_mask]))

        mean_mhw_days = float(np.nanmean(mhw_days_map[ocean_mask]))
        total_mhw_pixel_days = float(np.nansum(mhw_days_map[ocean_mask]))

        mean_cum_intensity = float(np.nanmean(cum_intensity_map[ocean_mask]))
        total_cum_intensity = float(np.nansum(cum_intensity_map[ocean_mask]))

        annual_days_rows.append({
            "year": int(year),
            "mean_mhw_days_per_ocean_pixel": mean_mhw_days,
            "total_mhw_pixel_days": total_mhw_pixel_days,
            "ocean_pixels": ocean_pixels,
        })

        annual_intensity_rows.append({
            "year": int(year),
            "mean_cumulative_intensity_per_ocean_pixel": mean_cum_intensity,
            "total_cumulative_intensity": total_cum_intensity,
            "mean_intensity_on_mhw_days": mean_intensity,
            "max_intensity_on_mhw_days": max_intensity,
        })

        annual_mhw_days_maps.append(mhw_days_map)
        annual_cum_intensity_maps.append(cum_intensity_map)

    annual_days_df = pd.DataFrame(annual_days_rows)
    annual_intensity_df = pd.DataFrame(annual_intensity_rows)

    annual_days_csv = out_dir / "annual_mhw_days.csv"
    annual_intensity_csv = out_dir / "annual_mhw_intensity.csv"

    annual_days_df.to_csv(annual_days_csv, index=False)
    annual_intensity_df.to_csv(annual_intensity_csv, index=False)

    print("[SAVE]", annual_days_csv)
    print("[SAVE]", annual_intensity_csv)

    annual_mhw_days_maps = np.stack(annual_mhw_days_maps, axis=0)
    mean_mhw_days_map = np.nanmean(annual_mhw_days_maps, axis=0)

    return {
        "ssta": ssta,
        "mhw": mhw,
        "exceed90": exceed90,
        "dates": dates,
        "years": years,
        "ocean_mask": ocean_mask,
        "annual_days_df": annual_days_df,
        "annual_intensity_df": annual_intensity_df,
        "mean_mhw_days_map": mean_mhw_days_map,
    }


def seasonal_statistics(cache, out_dir):
    """
    Compute seasonal mean MHW days maps and CSV summary.

    Seasons:
        DJF, MAM, JJA, SON
    """
    mhw = cache["mhw"]
    dates = cache["dates"]
    ocean_mask = cache["ocean_mask"]

    season_defs = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }

    rows = []
    seasonal_maps = {}

    for season, months in season_defs.items():
        idx = np.isin(dates.month, months)

        # Total MHW days across all selected seasons, then divided by number of years.
        season_days_map = mhw[idx].sum(axis=0).astype(np.float32)
        season_days_map = season_days_map / len(cache["years"])

        seasonal_maps[season] = season_days_map

        rows.append({
            "season": season,
            "mean_mhw_days_per_year_per_ocean_pixel": float(np.nanmean(season_days_map[ocean_mask])),
            "max_mhw_days_per_year_pixel": float(np.nanmax(season_days_map[ocean_mask])),
        })

    seasonal_df = pd.DataFrame(rows)
    seasonal_csv = out_dir / "seasonal_mhw_days.csv"
    seasonal_df.to_csv(seasonal_csv, index=False)
    print("[SAVE]", seasonal_csv)

    return seasonal_df, seasonal_maps


def save_summary(cache, seasonal_df, out_dir):
    """
    Save label_summary.json.
    """
    mhw = cache["mhw"]
    exceed90 = cache["exceed90"]
    ssta = cache["ssta"]
    ocean_mask = cache["ocean_mask"]
    dates = cache["dates"]
    annual_days_df = cache["annual_days_df"]
    annual_intensity_df = cache["annual_intensity_df"]

    # Ocean-only ratios.
    ocean_flat = ocean_mask.reshape(-1)
    mhw_flat = mhw.reshape(mhw.shape[0], -1)[:, ocean_flat]
    exceed_flat = exceed90.reshape(exceed90.shape[0], -1)[:, ocean_flat]
    ssta_flat = ssta.reshape(ssta.shape[0], -1)[:, ocean_flat]

    mhw_ratio_ocean = float(np.nanmean(mhw_flat))
    exceed90_ratio_ocean = float(np.nanmean(exceed_flat))

    days_slope, days_intercept = linear_trend(
        annual_days_df["year"].values,
        annual_days_df["mean_mhw_days_per_ocean_pixel"].values,
    )

    intensity_slope, intensity_intercept = linear_trend(
        annual_intensity_df["year"].values,
        annual_intensity_df["mean_cumulative_intensity_per_ocean_pixel"].values,
    )

    summary = {
        "time_start": str(dates.min().date()),
        "time_end": str(dates.max().date()),
        "n_days": int(len(dates)),
        "n_years": int(len(np.unique(dates.year))),
        "ocean_pixels": int(ocean_mask.sum()),
        "total_grid_pixels": int(ocean_mask.size),

        "exceed90_ratio_ocean_only": exceed90_ratio_ocean,
        "mhw_ratio_ocean_only": mhw_ratio_ocean,

        "mean_annual_mhw_days_per_ocean_pixel": float(
            annual_days_df["mean_mhw_days_per_ocean_pixel"].mean()
        ),
        "max_annual_mhw_days_per_ocean_pixel": float(
            annual_days_df["mean_mhw_days_per_ocean_pixel"].max()
        ),
        "year_of_max_mean_mhw_days": int(
            annual_days_df.loc[
                annual_days_df["mean_mhw_days_per_ocean_pixel"].idxmax(), "year"
            ]
        ),

        "annual_mhw_days_trend_days_per_year": days_slope,
        "annual_mhw_days_trend_days_per_decade": days_slope * 10.0,

        "mean_annual_cumulative_intensity_per_ocean_pixel": float(
            annual_intensity_df["mean_cumulative_intensity_per_ocean_pixel"].mean()
        ),
        "annual_cumulative_intensity_trend_per_year": intensity_slope,
        "annual_cumulative_intensity_trend_per_decade": intensity_slope * 10.0,

        "seasonal_summary": seasonal_df.to_dict(orient="records"),

        "method_note": (
            "Strict Hobday-style MHW labels: 90th percentile threshold, "
            "1982-2011 climatology, ±5-day calendar window, 31-day smoothing, "
            "minimum duration 5 days, gaps <= 2 days joined."
        ),
    }

    out_json = out_dir / "label_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[SAVE]", out_json)

    return summary


def plot_annual_days(cache, out_dir):
    df = cache["annual_days_df"]
    years = df["year"].values
    y = df["mean_mhw_days_per_ocean_pixel"].values

    slope, intercept = linear_trend(years, y)
    trend = slope * years + intercept

    plt.figure(figsize=(9, 4))
    plt.plot(years, y, marker="o", linewidth=1.5, markersize=3, label="Annual mean")
    plt.plot(years, trend, linestyle="--", linewidth=1.5, label=f"Trend: {slope*10:.2f} days/decade")
    plt.xlabel("Year")
    plt.ylabel("Mean MHW days per ocean pixel")
    plt.title("Annual marine heatwave days over the South China Sea")
    plt.legend()
    plt.tight_layout()

    out_file = out_dir / "annual_mhw_days_trend.png"
    plt.savefig(out_file, dpi=250)
    plt.close()
    print("[SAVE]", out_file)


def plot_annual_intensity(cache, out_dir):
    df = cache["annual_intensity_df"]
    years = df["year"].values
    y = df["mean_cumulative_intensity_per_ocean_pixel"].values

    slope, intercept = linear_trend(years, y)
    trend = slope * years + intercept

    plt.figure(figsize=(9, 4))
    plt.plot(years, y, marker="o", linewidth=1.5, markersize=3, label="Annual mean")
    plt.plot(years, trend, linestyle="--", linewidth=1.5, label=f"Trend: {slope*10:.2f} °C days/decade")
    plt.xlabel("Year")
    plt.ylabel("Mean cumulative intensity per ocean pixel")
    plt.title("Annual cumulative MHW intensity over the South China Sea")
    plt.legend()
    plt.tight_layout()

    out_file = out_dir / "annual_mhw_intensity_trend.png"
    plt.savefig(out_file, dpi=250)
    plt.close()
    print("[SAVE]", out_file)


def plot_mean_mhw_days_map(ds, cache, out_dir):
    mean_map = cache["mean_mhw_days_map"]
    ocean_mask = cache["ocean_mask"]
    mean_map = np.where(ocean_mask, mean_map, np.nan)

    lat = ds["lat"].values
    lon = ds["lon"].values

    plt.figure(figsize=(7, 5))
    plt.pcolormesh(lon, lat, mean_map, shading="auto")
    plt.colorbar(label="Mean MHW days per year")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Mean annual MHW days, 1982-2023")
    plt.tight_layout()

    out_file = out_dir / "mhw_days_mean_map_1982_2023.png"
    plt.savefig(out_file, dpi=250)
    plt.close()
    print("[SAVE]", out_file)


def plot_seasonal_maps(ds, seasonal_maps, cache, out_dir):
    lat = ds["lat"].values
    lon = ds["lon"].values
    ocean_mask = cache["ocean_mask"]

    seasons = ["DJF", "MAM", "JJA", "SON"]

    vals = []
    for s in seasons:
        vals.append(np.where(ocean_mask, seasonal_maps[s], np.nan))
    vmax = np.nanmax(vals)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    for ax, season, arr in zip(axes.ravel(), seasons, vals):
        im = ax.pcolormesh(lon, lat, arr, shading="auto", vmin=0, vmax=vmax)
        ax.set_title(season)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Mean seasonal MHW days per year")

    out_file = out_dir / "seasonal_mhw_days_map.png"
    plt.savefig(out_file, dpi=250)
    plt.close()
    print("[SAVE]", out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_nc", type=str, default=str(cfg.LABEL_NC))
    args = parser.parse_args()

    label_nc = Path(args.label_nc)
    out_dir = get_output_dir()

    print("[LOAD]", label_nc)
    ds = xr.open_dataset(label_nc)
    print(ds)

    cache = annual_statistics(ds, out_dir)
    seasonal_df, seasonal_maps = seasonal_statistics(cache, out_dir)

    summary = save_summary(cache, seasonal_df, out_dir)

    plot_annual_days(cache, out_dir)
    plot_annual_intensity(cache, out_dir)
    plot_mean_mhw_days_map(ds, cache, out_dir)
    plot_seasonal_maps(ds, seasonal_maps, cache, out_dir)

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2))

    print("\n[DONE]")
    print("Validation outputs saved to:", out_dir)

    ds.close()


if __name__ == "__main__":
    main()
