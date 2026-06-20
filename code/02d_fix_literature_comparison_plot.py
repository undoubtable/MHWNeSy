# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("/ybz/ybz/2026/MHWNeurRL/outputs/02_label_validation/literature_comparison")
csv_path = OUT_DIR / "jja_mhw_days_1982_2020.csv"
summary_path = OUT_DIR / "literature_comparison_summary.json"

df = pd.read_csv(csv_path)
summary = json.loads(summary_path.read_text())

years = df["year"].values
y = df["mean_mhw_days_per_ocean_pixel"].values

# This study trend
slope_year = summary["this_study_jja_mhw_days_trend_days_per_year"]
slope_decade = summary["this_study_jja_mhw_days_trend_days_per_decade"]

# Literature reference trend
lit_slope_decade = summary["literature_reference_jja_mhw_days_trend_days_per_decade"]
lit_slope_year = lit_slope_decade / 10.0

# Anchor both trend lines at the first-year fitted/observed value
anchor_year = years[0]
anchor_value = y[0]

this_trend = anchor_value + slope_year * (years - anchor_year)
lit_trend = anchor_value + lit_slope_year * (years - anchor_year)

# Figure 1: corrected trend comparison
plt.figure(figsize=(9, 4.5))
plt.plot(years, y, marker="o", linewidth=1.4, markersize=3, label="This study: JJA MHW days")
plt.plot(years, this_trend, "--", linewidth=2.0, label=f"This study trend: {slope_decade:.2f} days/decade")
plt.plot(years, lit_trend, ":", linewidth=2.5, label=f"Literature trend: ~{lit_slope_decade:.1f} days/decade")

plt.xlabel("Year")
plt.ylabel("JJA mean MHW days per ocean pixel")
plt.title("South China Sea summer MHW days trend comparison")
plt.legend()
plt.tight_layout()

out_file = OUT_DIR / "jja_mhw_days_trend_vs_literature_corrected.png"
plt.savefig(out_file, dpi=250)
plt.close()
print("[SAVE]", out_file)

# Figure 2: direct slope comparison
plt.figure(figsize=(5.5, 4.2))
names = ["Literature\nbenchmark", "This study"]
vals = [lit_slope_decade, slope_decade]

plt.bar(names, vals)
plt.ylabel("Trend (days/decade)")
plt.title("JJA MHW days trend comparison")
for i, v in enumerate(vals):
    plt.text(i, v + 0.15, f"{v:.2f}", ha="center", va="bottom")

plt.tight_layout()
out_file = OUT_DIR / "jja_mhw_days_trend_bar_comparison.png"
plt.savefig(out_file, dpi=250)
plt.close()
print("[SAVE]", out_file)
