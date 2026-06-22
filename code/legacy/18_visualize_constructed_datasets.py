# -*- coding: utf-8 -*-
"""
Visualize constructed forecast datasets.

Full forecast figures show the complete 10-day multichannel input.
Compact forecast figures show t-9, t-5, t0, and target t+5 for presentation.

Outputs:
    outputs/18_dataset_visualization/forecast_samples_full/
    outputs/18_dataset_visualization/forecast_samples_compact/
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


FEATURES = [
    {
        "name": "SSTA",
        "block": 0,
        "cmap": "coolwarm",
        "continuous": True,
        "unit": "°C anomaly",
    },
    {
        "name": "MHW mask",
        "block": 1,
        "continuous": False,
        "positive_label": "event",
        "negative_label": "no event",
    },
    {
        "name": "exceed90 mask",
        "block": 2,
        "continuous": False,
        "positive_label": "exceedance",
        "negative_label": "no exceedance",
    },
    {
        "name": "threshold_gap",
        "block": 3,
        "cmap": "coolwarm",
        "continuous": True,
        "unit": "°C relative to 90th percentile threshold",
    },
]

BINARY_CMAP = ListedColormap(["#f7f7f7", "#2166ac"])
BINARY_NORM = BoundaryNorm([-0.5, 0.5, 1.5], BINARY_CMAP.N)


def parse_indices(s):
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def choose_indices(n, requested, n_samples, seed):
    if requested:
        return [i for i in requested if 0 <= i < n]

    rng = np.random.default_rng(seed)
    size = min(n_samples, n)
    return sorted(rng.choice(np.arange(n), size=size, replace=False).tolist())


def rel_date_label(target_date, rel_day, lead):
    target_ts = pd.Timestamp(str(target_date))
    date = target_ts - pd.Timedelta(days=lead - rel_day)
    if rel_day == 0:
        tag = "t0"
    else:
        tag = f"t{rel_day}"
    return f"{tag}\n{date.date()}"


def target_label(target_date, lead):
    return f"target t+{lead}\n{pd.Timestamp(str(target_date)).date()}"


def history_channel(feature, hist_idx, history):
    return feature["block"] * history + hist_idx


def add_binary_legend(ax, positive_label, negative_label):
    handles = [
        Patch(facecolor="#2166ac", edgecolor="black", label=f"1 = {positive_label}"),
        Patch(facecolor="#f7f7f7", edgecolor="black", label=f"0 = {negative_label}"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=1,
        fontsize=7,
        frameon=False,
    )


def imshow_feature(ax, arr, feature, title="", add_colorbar=False, vmin=None, vmax=None):
    if feature["continuous"]:
        if vmax is None:
            vmax = float(np.nanpercentile(np.abs(arr), 98))
            vmax = max(vmax, 1e-6)
        if vmin is None:
            vmin = -vmax
        im = ax.imshow(arr, origin="lower", cmap=feature["cmap"], vmin=vmin, vmax=vmax)
        if add_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label(feature["unit"], fontsize=8)
    else:
        im = ax.imshow(arr, origin="lower", cmap=BINARY_CMAP, norm=BINARY_NORM)

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_full_sample(X, target_date, sample_index, out_path, history, lead):
    fig, axes = plt.subplots(4, history, figsize=(2.0 * history, 8.2), constrained_layout=True)

    for r, feature in enumerate(FEATURES):
        for h in range(history):
            ax = axes[r, h]
            ch = history_channel(feature, h, history)
            rel = h - (history - 1)
            title = rel_date_label(target_date, rel, lead) if r == 0 else ""
            imshow_feature(
                ax,
                np.array(X[ch], dtype=np.float32),
                feature,
                title=title,
                add_colorbar=(feature["continuous"] and h == history - 1),
            )
            if h == 0:
                ax.set_ylabel(feature["name"], fontsize=10)
            if (not feature["continuous"]) and h == history - 1:
                add_binary_legend(ax, feature["positive_label"], feature["negative_label"])

    fig.suptitle(f"Full 10-day forecast input | sample {sample_index} | target {target_date}", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_compact_sample(X, y, target_date, sample_index, out_path, history, lead):
    rel_days = [-9, -5, 0]
    hist_indices = [0, 4, history - 1]
    col_titles = [
        rel_date_label(target_date, rel, lead)
        for rel in rel_days
    ] + [target_label(target_date, lead)]

    row_labels = {
        "SSTA": "SSTA\n(°C anomaly)",
        "MHW mask": "MHW mask",
        "exceed90 mask": "exceed90 mask",
        "threshold_gap": "threshold_gap\n(°C vs 90th pct.)",
    }

    fig = plt.figure(figsize=(15.2, 9.8))
    gs = fig.add_gridspec(
        nrows=4,
        ncols=5,
        width_ratios=[1, 1, 1, 1, 0.72],
        left=0.09,
        right=0.985,
        top=0.89,
        bottom=0.08,
        wspace=0.18,
        hspace=0.24,
    )
    axes = np.empty((4, 4), dtype=object)
    side_axes = []
    for r in range(4):
        for c in range(4):
            axes[r, c] = fig.add_subplot(gs[r, c])
        side_axes.append(fig.add_subplot(gs[r, 4]))

    for r, feature in enumerate(FEATURES):
        side_ax = side_axes[r]
        side_ax.axis("off")

        shared_vmin = shared_vmax = None
        if feature["continuous"]:
            vals = []
            for hist_idx in hist_indices:
                ch = history_channel(feature, hist_idx, history)
                vals.append(np.array(X[ch], dtype=np.float32))
            vmax = float(np.nanpercentile(np.abs(np.stack(vals)), 98))
            shared_vmax = max(vmax, 1e-6)
            shared_vmin = -shared_vmax

        last_im = None
        for c in range(4):
            ax = axes[r, c]
            if c < 3:
                ch = history_channel(feature, hist_indices[c], history)
                arr = np.array(X[ch], dtype=np.float32)
                last_im = imshow_feature(
                    ax,
                    arr,
                    feature,
                    title=col_titles[c] if r == 0 else "",
                    add_colorbar=False,
                    vmin=shared_vmin,
                    vmax=shared_vmax,
                )
            else:
                if feature["name"] == "MHW mask":
                    target_feature = {
                        "name": "Target MHW",
                        "continuous": False,
                        "positive_label": "event",
                        "negative_label": "no event",
                    }
                    imshow_feature(
                        ax,
                        np.array(y, dtype=np.float32),
                        target_feature,
                        title="Target MHW",
                    )
                else:
                    ax.axis("off")
                    if r == 0:
                        ax.set_title(col_titles[c], fontsize=9)

            if c == 0:
                ax.set_ylabel(row_labels[feature["name"]], fontsize=10)

        if feature["continuous"]:
            cax = side_ax.inset_axes([0.05, 0.0, 0.18, 1.0])
            cbar = fig.colorbar(last_im, cax=cax)
            cbar.set_label(feature["unit"], fontsize=9)
        else:
            side_ax.axis("on")
            side_ax.set_frame_on(False)
            side_ax.set_xticks([])
            side_ax.set_yticks([])
            side_ax.set_xlim(0, 1)
            side_ax.set_ylim(0, 1)
            handles = [
                Patch(facecolor="#2166ac", edgecolor="black", label=f"1 = {feature['positive_label']}"),
                Patch(facecolor="#f7f7f7", edgecolor="black", label=f"0 = {feature['negative_label']}"),
            ]
            side_ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, 0.5), fontsize=9, frameon=False)

    fig.suptitle(f"Compact forecast dataset view | sample {sample_index} | target {target_date}", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "03b_forecast_dataset_multichannel_h10_l5"),
    )
    parser.add_argument("--out_dir", type=str, default=str(cfg.OUTPUT_DIR / "18_dataset_visualization"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--lead", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=6)
    parser.add_argument("--sample_indices", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="both", choices=["full", "compact", "both"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    full_dir = out_dir / "forecast_samples_full"
    compact_dir = out_dir / "forecast_samples_compact"
    if args.mode in ["full", "both"]:
        full_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ["compact", "both"]:
        compact_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_dir / f"X_{args.split}.npy", mmap_mode="r")
    y = np.load(data_dir / f"y_{args.split}.npy", mmap_mode="r")
    target_dates = np.load(data_dir / f"target_dates_{args.split}.npy")

    expected_channels = args.history * len(FEATURES)
    if X.shape[1] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {X.shape[1]}")
    if X.shape[0] != y.shape[0] or X.shape[0] != len(target_dates):
        raise ValueError(f"Sample count mismatch: X={X.shape}, y={y.shape}, dates={len(target_dates)}")

    indices = choose_indices(
        n=X.shape[0],
        requested=parse_indices(args.sample_indices),
        n_samples=args.n_samples,
        seed=args.seed,
    )

    rows = []
    for idx in indices:
        target_date = str(target_dates[idx])
        date_tag = target_date.replace("-", "")

        row = {
            "split": args.split,
            "sample_index": int(idx),
            "target_date": target_date,
        }

        if args.mode in ["full", "both"]:
            full_path = full_dir / f"forecast_sample_{args.split}_{idx:04d}_{date_tag}_full.png"
            plot_full_sample(
                X=np.array(X[idx], dtype=np.float32),
                target_date=target_date,
                sample_index=idx,
                out_path=full_path,
                history=args.history,
                lead=args.lead,
            )
            row["full_figure"] = str(full_path)
            print("[SAVE]", full_path)

        if args.mode in ["compact", "both"]:
            compact_path = compact_dir / f"forecast_sample_{args.split}_{idx:04d}_{date_tag}_compact.png"
            plot_compact_sample(
                X=np.array(X[idx], dtype=np.float32),
                y=np.array(y[idx], dtype=np.uint8),
                target_date=target_date,
                sample_index=idx,
                out_path=compact_path,
                history=args.history,
                lead=args.lead,
            )
            row["compact_figure"] = str(compact_path)
            print("[SAVE]", compact_path)

        rows.append(row)

    manifest = out_dir / f"forecast_samples_{args.split}_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[SAVE]", manifest)
    print("[DONE]", out_dir)


if __name__ == "__main__":
    main()
