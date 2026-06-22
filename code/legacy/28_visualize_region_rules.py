#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize NeurRL-style region rules.

Outputs:
    outputs/28_region_rule_visualization/
        top_valid_region_rules.png
        top_invalid_region_rules.png
        representative_rule_cases/
        region_rule_summary.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


BINARY_CMAP = ListedColormap(["#f7f7f7", "#2166ac"])
BINARY_NORM = BoundaryNorm([-0.5, 0.5, 1.5], BINARY_CMAP.N)


def parse_rule_file(path, target):
    lines = path.read_text(encoding="utf-8").splitlines()
    rules = []
    rank_re = re.compile(r"^(\d+)\.\s+IF\s+(.+)\s+THEN\s+(\w+)")
    metric_re = re.compile(r"val precision=([0-9.]+), recall=([0-9.]+), f1=([0-9.]+)")
    current = None
    for line in lines:
        m = rank_re.match(line.strip())
        if m:
            if current is not None:
                rules.append(current)
            rule_text = m.group(2)
            current = {
                "target": target,
                "rank": int(m.group(1)),
                "rule_text": rule_text,
                "atoms": [x.strip() for x in rule_text.split(" AND ")],
                "val_precision": np.nan,
                "val_recall": np.nan,
                "val_f1": np.nan,
            }
            continue
        m = metric_re.search(line)
        if m and current is not None:
            current["val_precision"] = float(m.group(1))
            current["val_recall"] = float(m.group(2))
            current["val_f1"] = float(m.group(3))
    if current is not None:
        rules.append(current)
    return rules


def parse_atom(atom):
    m = re.match(r"([A-Z0-9]+)_R(\d+)C(\d+)_(.+)", atom)
    if not m:
        return None
    var, r, c, state = m.groups()
    return var, int(r), int(c), state


def channel_for_var(var):
    return {
        "SSTA": 9,
        "TGAP": 39,
        "MHW": 19,
        "EXC": 29,
        "EXCEED90": 29,
        "COMP": 40,
    }.get(var, 40)


def title_for_var(var):
    return {
        "SSTA": "Recent SSTA",
        "TGAP": "Recent threshold_gap",
        "MHW": "Recent historical MHW",
        "EXC": "Recent exceed90",
        "EXCEED90": "Recent exceed90",
        "COMP": "Predicted component",
    }.get(var, var)


def region_rect(r, c, grid, size=64):
    h = size / grid
    return (c - 1) * h, (r - 1) * h, h, h


def atom_actual(row, atom):
    parsed = parse_atom(atom)
    if parsed is None:
        return ""
    var, r, c, state = parsed
    tag = f"R{r}C{c}"
    candidates = [
        f"{var}_{tag}_mean",
        f"{var}_{tag}_occ",
        f"{var}_{tag}_{state}",
    ]
    if var == "EXCEED90":
        candidates.insert(0, f"EXC_{tag}_occ")
    for col in candidates:
        if col in row:
            return row[col]
    return ""


def rule_pred(df, atoms):
    pred = np.ones(len(df), dtype=bool)
    for atom in atoms:
        pred &= df[atom].to_numpy(dtype=np.uint8).astype(bool)
    return pred


def metrics(y, pred):
    y = y.astype(bool)
    pred = pred.astype(bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1, int(pred.sum())


def plot_case(out_file, rule, row, X_event, grid):
    atoms = rule["atoms"]
    n = min(len(atoms), 3)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, atom in zip(axes, atoms[:n]):
        parsed = parse_atom(atom)
        if parsed is None:
            ax.axis("off")
            continue
        var, r, c, state = parsed
        ch = channel_for_var(var)
        arr = X_event[int(row["event_index"]), ch]
        if var in ["MHW", "EXC", "EXCEED90", "COMP"]:
            ax.imshow(arr > 0.5, origin="lower", cmap=BINARY_CMAP, norm=BINARY_NORM)
        else:
            vmax = max(float(np.nanpercentile(np.abs(arr), 98)), 1e-6)
            ax.imshow(arr, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        x, y, w, h = region_rect(r, c, grid, size=arr.shape[0])
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="yellow", linewidth=2.5))
        actual = atom_actual(row, atom)
        ax.set_title(f"{title_for_var(var)}\nR({r},{c}) {state}\nvalue={actual:.3g}" if actual != "" else atom, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{rule['target'].upper()} R{rule['rank']} | {rule['rule_text']} | "
        f"y_valid={int(row['y_valid'])} | event={int(row['event_index'])}",
        fontsize=11,
    )
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_rule_overview(out_file, rules, summary, title):
    fig, axes = plt.subplots(len(rules), 1, figsize=(13, max(3, 1.2 * len(rules))), constrained_layout=True)
    if len(rules) == 1:
        axes = [axes]
    for ax, rule in zip(axes, rules):
        row = summary[(summary["target"] == rule["target"]) & (summary["rank"] == rule["rank"])].iloc[0]
        ax.axis("off")
        ax.text(
            0.01, 0.5,
            f"R{rule['rank']}: IF {rule['rule_text']} THEN {rule['target']} | "
            f"val P/R/F1={row['val_precision']:.3f}/{row['val_recall']:.3f}/{row['val_f1']:.3f} | "
            f"test P/R/F1={row['test_precision']:.3f}/{row['test_recall']:.3f}/{row['test_f1']:.3f}",
            va="center",
            ha="left",
            fontsize=11,
        )
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule_dir", type=str, default=str(cfg.OUTPUT_DIR / "26_region_rule_learning"))
    parser.add_argument("--event_dir", type=str, default=str(cfg.OUTPUT_DIR / "06c_neurrl_event_dataset_from_multichannel_h10_l5"))
    parser.add_argument("--out_dir", type=str, default=str(cfg.OUTPUT_DIR / "28_region_rule_visualization"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--examples_per_rule", type=int, default=2)
    parser.add_argument("--grid", type=int, default=4)
    args = parser.parse_args()

    rule_dir = Path(args.rule_dir)
    out_dir = Path(args.out_dir)
    case_dir = out_dir / "representative_rule_cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    rules = (
        parse_rule_file(rule_dir / "learned_valid_region_rules.txt", "valid")
        + parse_rule_file(rule_dir / "learned_invalid_region_rules.txt", "invalid")
    )
    atoms = pd.read_csv(rule_dir / f"region_atoms_{args.split}.csv")
    z = np.load(Path(args.event_dir) / f"figure_event_{args.split}.npz", allow_pickle=True)
    X_event = z["X"].astype(np.float32)
    y_valid = atoms["y_valid"].to_numpy(dtype=np.uint8).astype(bool)

    rows = []
    for rule in rules:
        pred = rule_pred(atoms, rule["atoms"])
        y = y_valid if rule["target"] == "valid" else ~y_valid
        p, r, f1, support = metrics(y, pred)
        rows.append({
            "target": rule["target"],
            "rank": rule["rank"],
            "rule_text": rule["rule_text"],
            "simplified_rule": rule["rule_text"],
            "val_precision": rule["val_precision"],
            "val_recall": rule["val_recall"],
            "val_f1": rule["val_f1"],
            "test_precision": p,
            "test_recall": r,
            "test_f1": f1,
            "support_count_test": support,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "region_rule_summary.csv", index=False)
    print("[SAVE]", out_dir / "region_rule_summary.csv")

    valid_rules = [r for r in rules if r["target"] == "valid" and r["rank"] <= args.top_k]
    invalid_rules = [r for r in rules if r["target"] == "invalid" and r["rank"] <= args.top_k]
    plot_rule_overview(out_dir / "top_valid_region_rules.png", valid_rules, summary, "Top valid region rules")
    plot_rule_overview(out_dir / "top_invalid_region_rules.png", invalid_rules, summary, "Top invalid region rules")

    for rule in valid_rules + invalid_rules:
        pred = rule_pred(atoms, rule["atoms"])
        y = y_valid if rule["target"] == "valid" else ~y_valid
        candidates = atoms[pred & y].sort_values("area_px", ascending=False).head(args.examples_per_rule)
        if candidates.empty:
            candidates = atoms[pred].sort_values("area_px", ascending=False).head(args.examples_per_rule)
        subdir = case_dir / f"{rule['target']}_rule_{rule['rank']:02d}"
        subdir.mkdir(parents=True, exist_ok=True)
        for _, row in candidates.iterrows():
            out_file = subdir / f"event{int(row['event_index']):05d}_sample{int(row['sample_index']):04d}.png"
            plot_case(out_file, rule, row, X_event, args.grid)
            print("[SAVE]", out_file)

    print("[DONE]", out_dir)


if __name__ == "__main__":
    main()
