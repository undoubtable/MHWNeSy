# -*- coding: utf-8 -*-
"""
Compare generated MHW labels with published South China Sea MHW literature.

Main literature benchmark:
    Yao & Wang, 2021, JGR Oceans:
        South China Sea summer MHWs, 1982-2020.
        Reported trend:
            total summer MHW days increased by nearly 3.0 days/decade
            duration increased by nearly 1.0 days/time/decade
            frequency increased by nearly 0.2 times/decade

This script focuses on the most directly comparable metric:
    JJA mean MHW days trend over 1982-2020.

Outputs:
    outputs/02_label_validation/literature_comparison/
        jja_mhw_days_1982_2020.csv
        literature_comparison_table.csv
        literature_comparison_summary.json
        jja_mhw_days_trend_vs_literature.png
        decade_comparison_jja.csv
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


def get_out_dir():
    if hasattr(cfg, "OUTPUT_DIR"):
        out_dir = cfg.OUTPUT_DIR / "02_label_validation" / "literature_comparison"
    else:
        out_dir = cfg.ROOT / "outputs" / "02_label_validation" / "literature_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def linear_trend(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 3:
        return np.nan, np.nan

    slope, intercept = np.polyfit(x[valid], y[valid], deg=1)
    return float(slope), float(intercept)


def compute_ocean_mask(ds):
    # Ocean cells are grid cells that have finite SSTA at least once.
    ssta = ds["ssta"].astype("float32").values
    ocean_mask = np.isfinite(ssta).any(axis=0)
    return ocean_mask


def compute_seasonal_mhw_days(ds, ocean_mask, season_months, start_year, end_year):
    mhw = ds["mhw"].astype("uint8").values
    dates = pd.DatetimeIndex(ds["time"].values)

    rows = []

    for year in range(start_year, end_year + 1):
        idx = (dates.year == year) & np.isin(dates.month, season_months)

        if idx.sum() == 0:
            continue

        mhw_days_map = mhw[idx].sum(axis=0).astype(np.float32)
        mean_days = float(np.nanmean(mhw_days_map[ocean_mask]))
        total_pixel_days = float(np.nansum(mhw_days_map[ocean_mask]))

        rows.append({
            "year": year,
            "mean_mhw_days_per_ocean_pixel": mean_days,
            "total_mhw_pixel_days": total_pixel_days,
            "n_days_in_season": int(idx.sum()),
            "ocean_pixels": int(ocean_mask.sum()),
        })

    return pd.DataFrame(rows)


def make_decade_comparison(df):
    periods = {
        "1980s_1982_1989": (1982, 1989),
        "1990s_1990_1999": (1990, 1999),
        "2000s_2000_2009": (2000, 2009),
        "2010s_2010_2019": (2010, 2019),
        "2020s_2020_2023": (2020, 2023),
    }

    rows = []
    for name, (a, b) in periods.items():
        sub = df[(df["year"] >= a) & (df["year"] <= b)]
        if len(sub) == 0:
            continue
        rows.append({
            "period": name,
            "start_year": a,
            "end_year": b,
            "mean_jja_mhw_days_per_ocean_pixel": float(sub["mean_mhw_days_per_ocean_pixel"].mean()),
            "max_jja_mhw_days_per_ocean_pixel": float(sub["mean_mhw_days_per_ocean_pixel"].max()),
        })

    out = pd.DataFrame(rows)

    # Ratio of 2010s to 1980s, useful because some SCS studies discuss stronger occurrence in 2010s.
    if "1980s_1982_1989" in out["period"].values and "2010s_2010_2019" in out["period"].values:
        base = float(out.loc[out["period"] == "1980s_1982_1989", "mean_jja_mhw_days_per_ocean_pixel"].iloc[0])
        recent = float(out.loc[out["period"] == "2010s_2010_2019", "mean_jja_mhw_days_per_ocean_pixel"].iloc[0])
        ratio = recent / (base + 1e-8)
    else:
        ratio = np.nan

    out.attrs["ratio_2010s_to_1980s"] = ratio
    return out, ratio


def plot_jja_trend(df, our_slope_decade, lit_slope_decade, out_file):
    years = df["year"].values
    y = df["mean_mhw_days_per_ocean_pixel"].values

    slope, intercept = linear_trend(years, y)
    trend = slope * years + intercept

    plt.figure(figsize=(9, 4.5))
    plt.plot(
        years, y,
        marker="o",
        linewidth=1.5,
        markersize=3,
        label="This study: JJA MHW days"
    )
    plt.plot(
        years, trend,
        linestyle="--",
        linewidth=1.8,
        label=f"This study trend: {our_slope_decade:.2f} days/decade"
    )

    # Literature slope is shown as reference text because exact baseline value depends on their region/mask.
    plt.axhline(
        y=np.nanmean(y),
        linestyle=":",
        linewidth=1.2,
        label=f"Literature reference: ~{lit_slope_decade:.1f} days/decade"
    )

    plt.xlabel("Year")
    plt.ylabel("JJA mean MHW days per ocean pixel")
    plt.title("South China Sea summer MHW days: this study vs literature benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_nc", type=str, default=str(cfg.LABEL_NC))
    parser.add_argument("--start_year", type=int, default=1982)
    parser.add_argument("--end_year", type=int, default=2020)
    parser.add_argument("--literature_jja_days_trend", type=float, default=3.0)
    args = parser.parse_args()

    out_dir = get_out_dir()

    print("[LOAD]", args.label_nc)
    ds = xr.open_dataset(args.label_nc)
    print(ds)

    print("[OCEAN MASK]")
    ocean_mask = compute_ocean_mask(ds)
    print("ocean pixels:", int(ocean_mask.sum()))

    print("[COMPUTE JJA MHW DAYS]")
    jja_df = compute_seasonal_mhw_days(
        ds=ds,
        ocean_mask=ocean_mask,
        season_months=[6, 7, 8],
        start_year=args.start_year,
        end_year=args.end_year,
    )

    jja_csv = out_dir / f"jja_mhw_days_{args.start_year}_{args.end_year}.csv"
    jja_df.to_csv(jja_csv, index=False)
    print("[SAVE]", jja_csv)

    slope_year, intercept = linear_trend(
        jja_df["year"].values,
        jja_df["mean_mhw_days_per_ocean_pixel"].values,
    )
    slope_decade = slope_year * 10.0

    decade_df, ratio_2010s_to_1980s = make_decade_comparison(jja_df)
    decade_csv = out_dir / "decade_comparison_jja.csv"
    decade_df.to_csv(decade_csv, index=False)
    print("[SAVE]", decade_csv)

    comparison_rows = [
        {
            "reference": "Yao & Wang 2021, JGR Oceans",
            "reported_metric": "Average total summer MHW days trend",
            "reported_period": "1982-2020",
            "reported_region": "South China Sea",
            "reported_value": "nearly 3.0 days/decade",
            "this_study_metric": "JJA mean MHW days per ocean pixel trend",
            "this_study_period": f"{args.start_year}-{args.end_year}",
            "this_study_value": f"{slope_decade:.3f} days/decade",
            "comparison_note": (
                "Comparable in season and period, but not identical because "
                "spatial masks, regional averaging, and exact preprocessing may differ."
            ),
        },
        {
            "reference": "Li et al. 2022, Remote Sensing",
            "reported_metric": "Regional MHW count",
            "reported_period": "1982-2020",
            "reported_region": "South China Sea",
            "reported_value": "37 regional MHWs",
            "this_study_metric": "Not directly computed here",
            "this_study_period": f"{args.start_year}-{args.end_year}",
            "this_study_value": "N/A",
            "comparison_note": (
                "This requires event-system tracking with a regional area criterion; "
                "it should be compared after building event-level MHW systems."
            ),
        },
    ]

    comparison_df = pd.DataFrame(comparison_rows)
    comp_csv = out_dir / "literature_comparison_table.csv"
    comparison_df.to_csv(comp_csv, index=False)
    print("[SAVE]", comp_csv)

    fig_file = out_dir / "jja_mhw_days_trend_vs_literature.png"
    plot_jja_trend(
        jja_df,
        our_slope_decade=slope_decade,
        lit_slope_decade=args.literature_jja_days_trend,
        out_file=fig_file,
    )
    print("[SAVE]", fig_file)

    summary = {
        "label_file": str(args.label_nc),
        "comparison_period": f"{args.start_year}-{args.end_year}",
        "season": "JJA",
        "ocean_pixels": int(ocean_mask.sum()),
        "this_study_jja_mhw_days_trend_days_per_year": float(slope_year),
        "this_study_jja_mhw_days_trend_days_per_decade": float(slope_decade),
        "literature_reference_jja_mhw_days_trend_days_per_decade": float(args.literature_jja_days_trend),
        "difference_this_minus_literature_days_per_decade": float(slope_decade - args.literature_jja_days_trend),
        "ratio_2010s_to_1980s_jja_mean_days": float(ratio_2010s_to_1980s),
        "caution": (
            "This is a benchmark-level comparison, not an exact replication. "
            "Direct numerical equality is not expected because published studies "
            "may use different SCS boundaries, land masks, preprocessing, and "
            "regional aggregation definitions."
        ),
        "literature_notes": [
            "Yao & Wang 2021 reported nearly 3.0 days/decade increase in average total summer MHW days during 1982-2020.",
            "Li et al. 2022 identified 37 regional MHWs in the SCS from 1982 to 2020; this requires event-system tracking for direct comparison.",
        ],
    }

    summary_json = out_dir / "literature_comparison_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[SAVE]", summary_json)

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2))

    ds.close()


if __name__ == "__main__":
    main()
