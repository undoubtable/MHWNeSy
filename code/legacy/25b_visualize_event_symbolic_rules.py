#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize event-level symbolic rules learned from event statistics.

This is independent from code/25_visualize_symbolic_rules.py.

Outputs:
    outputs/25b_event_symbolic_rule_visualization/
        rule_summary.csv
        event_rule_metrics_bar.png
        rule_trigger_examples/
        atom_distribution_plots/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


BINARY_CMAP = ListedColormap(["#f7f7f7", "#2166ac"])
BINARY_NORM = BoundaryNorm([-0.5, 0.5, 1.5], BINARY_CMAP.N)
OVERLAY_CMAP = ListedColormap(["#2b2b2b", "#2ca25f", "#de2d26", "#3182bd"])
OVERLAY_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], OVERLAY_CMAP.N)

ATOM_LABELS = {
    "last_gap_positive": "recent threshold_gap > 0",
    "gap_last_not_positive": "recent threshold_gap <= 0",
    "positive_gap_days_ge_3": "threshold_gap > 0 for >= 3 days",
    "positive_gap_days_ge_5": "threshold_gap > 0 for >= 5 days",
    "positive_gap_days_ge_8": "threshold_gap > 0 for >= 8 days",
    "few_mhw_days": "historical MHW days <= 1",
    "few_exceed90_days": "historical exceed90 days <= 1",
    "mhw_days_ge_3": "historical MHW days >= 3",
    "exceed90_days_ge_5": "historical exceed90 days >= 5",
    "area_large": "component area is large",
    "area_stable": "component area is stable",
}


def parse_rule_file(path: Path, rule_type: str):
    text = path.read_text(encoding="utf-8").splitlines()
    rules = []
    current = None
    bullet_re = re.compile(r"^\s*-\s+([^:]+):\s+([A-Za-z0-9_]+)\s+(<=|>=|>|<)\s+([-+0-9.eE]+)")
    rank_re = re.compile(r"^(\d+)\.\s+(.+)$")
    metric_re = re.compile(r"val precision=([0-9.]+), recall=([0-9.]+), f1=([0-9.]+)")

    for line in text:
        m = rank_re.match(line.strip())
        if m:
            if current is not None:
                rules.append(current)
            rule_text = m.group(2).strip()
            current = {
                "rule_type": rule_type,
                "rank": int(m.group(1)),
                "rule_text": rule_text,
                "atoms": [a.strip() for a in rule_text.split(" AND ")],
                "atom_specs": {},
                "val_precision": np.nan,
                "val_recall": np.nan,
                "val_f1": np.nan,
            }
            continue

        if current is None:
            continue

        m = metric_re.search(line)
        if m:
            current["val_precision"] = float(m.group(1))
            current["val_recall"] = float(m.group(2))
            current["val_f1"] = float(m.group(3))
            continue

        m = bullet_re.match(line)
        if m:
            atom, col, op, value = m.groups()
            current["atom_specs"][atom.strip()] = (col, op, float(value))

    if current is not None:
        rules.append(current)
    return rules


def eval_atom(df, spec):
    col, op, value = spec
    x = df[col].to_numpy(dtype=np.float32)
    if op == ">":
        return x > value
    if op == ">=":
        return x >= value
    if op == "<":
        return x < value
    if op == "<=":
        return x <= value
    raise ValueError(f"Unsupported op: {op}")


def rule_mask(df, rule):
    mask = np.ones(len(df), dtype=bool)
    for atom in rule["atoms"]:
        spec = rule["atom_specs"].get(atom)
        if spec is None:
            raise KeyError(f"Missing atom spec for {atom} in rule {rule['rule_text']}")
        mask &= eval_atom(df, spec)
    return mask


def metrics(y_true, pred):
    y_true = y_true.astype(bool)
    pred = pred.astype(bool)
    tp = int((pred & y_true).sum())
    fp = int((pred & ~y_true).sum())
    fn = int((~pred & y_true).sum())
    tn = int((~pred & ~y_true).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support_count": int(pred.sum()),
        "support_ratio": float(pred.mean()),
    }


def simplify_rule(rule):
    labels = [ATOM_LABELS.get(a, a) for a in rule["atoms"]]
    return " AND ".join(labels)


def crop_2d(arr, center_r, center_c, crop_size, fill_value=0):
    h, w = arr.shape
    half = crop_size // 2
    cr = int(round(center_r))
    cc = int(round(center_c))
    r0, r1 = cr - half, cr - half + crop_size
    c0, c1 = cc - half, cc - half + crop_size
    out = np.full((crop_size, crop_size), fill_value, dtype=arr.dtype)
    sr0, sr1 = max(r0, 0), min(r1, h)
    sc0, sc1 = max(c0, 0), min(c1, w)
    dr0, dc0 = sr0 - r0, sc0 - c0
    if sr1 > sr0 and sc1 > sc0:
        out[dr0:dr0 + (sr1 - sr0), dc0:dc0 + (sc1 - sc0)] = arr[sr0:sr1, sc0:sc1]
    return out


def get_component_mask(pred_mask, component_id, expected_area):
    from scipy import ndimage

    lab, nlab = ndimage.label(pred_mask.astype(bool), structure=np.ones((3, 3), dtype=int))
    if 1 <= component_id <= nlab:
        comp = lab == component_id
        area = int(comp.sum())
        if abs(area - expected_area) <= max(2, 0.02 * expected_area):
            return comp
    if nlab <= 0:
        return None
    areas = ndimage.sum(pred_mask.astype(bool), lab, index=np.arange(1, nlab + 1))
    idx = int(np.argmin(np.abs(np.asarray(areas) - expected_area))) + 1
    return lab == idx


def find_center(full_comp, comp_patch, crop_size):
    rr, cc = np.where(full_comp)
    if len(rr) == 0:
        return full_comp.shape[0] / 2, full_comp.shape[1] / 2
    candidates = [
        (float(rr.mean()), float(cc.mean())),
        ((float(rr.min()) + float(rr.max())) / 2, (float(cc.min()) + float(cc.max())) / 2),
    ]
    best, best_mismatch = candidates[0], float("inf")
    target = comp_patch > 0.5
    for r, c in candidates:
        crop = crop_2d(full_comp.astype(np.uint8), r, c, crop_size)
        mismatch = float(np.mean((crop > 0) != target))
        if mismatch < best_mismatch:
            best, best_mismatch = (r, c), mismatch
    return best


def overlay(pred, target):
    p = pred > 0.5
    t = target > 0.5
    out = np.zeros(pred.shape, dtype=np.uint8)
    out[p & t] = 1
    out[p & ~t] = 2
    out[~p & t] = 3
    return out


def plot_binary(ax, arr, title):
    ax.imshow(arr > 0.5, origin="lower", cmap=BINARY_CMAP, norm=BINARY_NORM)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_example(out_file, x_patch, target_crop, pred_crop, comp_patch, row, rule, dates):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    gap = np.array(x_patch[39], dtype=np.float32)
    vmax = max(float(np.nanpercentile(np.abs(gap), 98)), 1e-6)
    im = axes[0, 0].imshow(gap, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title("Recent threshold_gap", fontsize=10)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    cbar = fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.03)
    cbar.set_label("threshold_gap", fontsize=9)

    plot_binary(axes[0, 1], target_crop, "Target MHW")
    plot_binary(axes[0, 2], pred_crop, "Original prediction")
    plot_binary(axes[1, 0], comp_patch, "Candidate component")
    axes[1, 1].imshow(overlay(pred_crop, target_crop), origin="lower", cmap=OVERLAY_CMAP, norm=OVERLAY_NORM)
    axes[1, 1].set_title("Overlay: TP / FP / FN", fontsize=10)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    axes[1, 2].axis("off")
    atom_lines = []
    for atom in rule["atoms"]:
        col, op, value = rule["atom_specs"][atom]
        actual = row[col]
        atom_lines.append(f"{atom}: {actual:.3g} {op} {value:.3g}")
    sample_index = int(row["sample_index"])
    text = "\n".join([
        f"Rule: {rule['rule_text']}",
        f"Simplified: {simplify_rule(rule)}",
        "",
        *atom_lines,
        "",
        f"satisfies_rule: yes",
        f"y_valid: {int(row['y_valid'])}",
        f"best_iou: {float(row['best_iou']):.3f}",
        f"overlap_ratio: {float(row['overlap_ratio']):.3f}",
        f"sample_index: {sample_index}",
        f"component_id: {int(row['component_id'])}",
        f"target_date: {dates[sample_index]}",
    ])
    axes[1, 2].text(0.0, 1.0, text, va="top", ha="left", fontsize=9)
    fig.suptitle(f"{rule['rule_type'].upper()} rule R{rule['rank']} trigger example", fontsize=13)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_metrics_bar(summary, out_file, top_n):
    df = summary[summary["rank"] <= top_n].copy()
    labels = [f"{r.rule_type[0].upper()}{int(r.rank)}" for r in df.itertuples()]
    x = np.arange(len(df))
    width = 0.13
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(df)), 5.2), constrained_layout=True)
    for j, (col, label) in enumerate([
        ("val_precision", "Val precision"),
        ("val_recall", "Val recall"),
        ("val_f1", "Val F1"),
        ("test_precision", "Test precision"),
        ("test_recall", "Test recall"),
        ("test_f1", "Test F1"),
    ]):
        ax.bar(x + (j - 2.5) * width, df[col], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Event symbolic rule metrics")
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    print("[SAVE]", out_file)


def plot_atom_distributions(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    atoms = [
        ("mean_threshold_gap_inside_candidate_last", "threshold_gap_last"),
        ("threshold_gap_positive_days", "positive_gap_days"),
        ("historical_mhw_days", "historical_mhw_days"),
        ("historical_exceed90_days", "historical_exceed90_days"),
        ("component_area_fraction_mean", "area_fraction_mean"),
        ("mean_ssta_inside_candidate_last", "ssta_last"),
    ]
    for col, name in atoms:
        if col not in df:
            continue
        valid = df[df["y_valid"] == 1][col].dropna()
        invalid = df[df["y_valid"] == 0][col].dropna()
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.hist(invalid, bins=30, alpha=0.6, label="invalid", density=True)
        ax.hist(valid, bins=30, alpha=0.6, label="valid", density=True)
        ax.set_title(f"Distribution: {name}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()
        out_file = out_dir / f"{name}_valid_vs_invalid.png"
        fig.savefig(out_file, dpi=180)
        plt.close(fig)
        print("[SAVE]", out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule_dir", type=str, default=str(cfg.OUTPUT_DIR / "20_event_rule_learning"))
    parser.add_argument("--event_dir", type=str, default=str(cfg.OUTPUT_DIR / "06c_neurrl_event_dataset_from_multichannel_h10_l5"))
    parser.add_argument("--data_dir", type=str, default=str(cfg.OUTPUT_DIR / "03b_forecast_dataset_multichannel_h10_l5"))
    parser.add_argument("--pred_dir", type=str, default=str(cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5"))
    parser.add_argument("--out_dir", type=str, default=str(cfg.OUTPUT_DIR / "25b_event_symbolic_rule_visualization"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--top_rules", type=int, default=5)
    parser.add_argument("--examples_per_rule", type=int, default=3)
    parser.add_argument("--crop_size", type=int, default=64)
    args = parser.parse_args()

    rule_dir = Path(args.rule_dir)
    out_dir = Path(args.out_dir)
    example_dir = out_dir / "rule_trigger_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)

    rules = (
        parse_rule_file(rule_dir / "learned_valid_rules.txt", "valid")
        + parse_rule_file(rule_dir / "learned_invalid_rules.txt", "invalid")
    )
    test = pd.read_csv(rule_dir / f"event_rule_table_{args.split}.csv")
    val_metrics = pd.read_csv(rule_dir / "rule_metrics.csv")
    y_valid = test["y_valid"].to_numpy(dtype=np.uint8).astype(bool)

    rows = []
    for rule in rules:
        pred = rule_mask(test, rule)
        target = y_valid if rule["rule_type"] == "valid" else ~y_valid
        m = metrics(target, pred)
        rows.append({
            "rule_type": rule["rule_type"],
            "rank": rule["rank"],
            "rule_text": rule["rule_text"],
            "simplified_rule": simplify_rule(rule),
            "n_terms": len(rule["atoms"]),
            "val_precision": rule["val_precision"],
            "val_recall": rule["val_recall"],
            "val_f1": rule["val_f1"],
            "test_precision": m["precision"],
            "test_recall": m["recall"],
            "test_f1": m["f1"],
            "support_count_test": m["support_count"],
            "support_ratio_test": m["support_ratio"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "rule_summary.csv", index=False)
    print("[SAVE]", out_dir / "rule_summary.csv")

    plot_metrics_bar(summary, out_dir / "event_rule_metrics_bar.png", top_n=args.top_rules)
    plot_atom_distributions(test, out_dir / "atom_distribution_plots")

    z = np.load(Path(args.event_dir) / f"figure_event_{args.split}.npz", allow_pickle=True)
    X_event = z["X"]
    y = np.load(Path(args.data_dir) / f"y_{args.split}.npy", mmap_mode="r")
    dates = np.load(Path(args.data_dir) / f"target_dates_{args.split}.npy", allow_pickle=True)
    pred_mask = np.load(Path(args.pred_dir) / f"pred_mask_{args.split}.npy", mmap_mode="r")

    for rule in rules:
        if rule["rank"] > args.top_rules:
            continue
        mask = rule_mask(test, rule)
        target_ok = (y_valid if rule["rule_type"] == "valid" else ~y_valid)
        candidates = test[mask & target_ok].copy()
        if candidates.empty:
            candidates = test[mask].copy()
        candidates = candidates.sort_values(["area_px", "best_iou"], ascending=[False, False]).head(args.examples_per_rule)
        rule_dir_out = example_dir / f"{rule['rule_type']}_rule_{rule['rank']:02d}"
        rule_dir_out.mkdir(parents=True, exist_ok=True)
        for _, row in candidates.iterrows():
            event_index = int(row["event_index"])
            sample_index = int(row["sample_index"])
            component_id = int(row["component_id"])
            x_patch = X_event[event_index]
            full_comp = get_component_mask(pred_mask[sample_index], component_id, float(row["area_px"]))
            if full_comp is None:
                center = (pred_mask.shape[1] / 2, pred_mask.shape[2] / 2)
            else:
                center = find_center(full_comp, x_patch[40], args.crop_size)
            target_crop = crop_2d(y[sample_index].astype(np.uint8), center[0], center[1], args.crop_size)
            pred_crop = crop_2d(pred_mask[sample_index].astype(np.uint8), center[0], center[1], args.crop_size)
            out_file = rule_dir_out / f"event{event_index:05d}_sample{sample_index:04d}_comp{component_id}.png"
            plot_example(out_file, x_patch, target_crop, pred_crop, x_patch[40], row, rule, dates)
            print("[SAVE]", out_file)

    # Touch val metrics so users can cross-check aggregate selected-rule behavior.
    val_metrics.to_csv(out_dir / "combined_rule_metrics_reference.csv", index=False)
    print("[DONE]", out_dir)


if __name__ == "__main__":
    main()
