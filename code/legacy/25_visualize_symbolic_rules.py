#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize learned event-level symbolic rules.

Outputs:
  outputs/25_symbolic_rule_visualization/atom_discriminability_test.png
  outputs/25_symbolic_rule_visualization/atom_rates_heatmap_test.png
  outputs/25_symbolic_rule_visualization/invalid_rule_metrics_test.png
  outputs/25_symbolic_rule_visualization/valid_rule_metrics_test.png
  outputs/25_symbolic_rule_visualization/top_invalid_rule_card.png
  outputs/25_symbolic_rule_visualization/top_valid_rule_card.png
  outputs/25_symbolic_rule_visualization/atom_statistics.csv
  outputs/25_symbolic_rule_visualization/individual_rule_metrics.csv
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ATOM_DESCRIPTIONS = {
    "mean_gap_high": "Mean threshold_gap inside candidate is high",
    "last_gap_positive": "Recent mean threshold_gap is positive",
    "positive_gap_days_ge_3": "threshold_gap is positive for at least 3 days",
    "positive_gap_days_ge_5": "threshold_gap is positive for at least 5 days",
    "positive_gap_days_ge_8": "threshold_gap is positive for at least 8 days",
    "max_gap_high": "Maximum threshold_gap inside candidate is high",
    "min_gap_positive": "Minimum mean threshold_gap is positive",
    "mhw_days_ge_1": "Historical MHW exists for at least 1 day",
    "mhw_days_ge_3": "Historical MHW exists for at least 3 days",
    "mhw_days_ge_5": "Historical MHW exists for at least 5 days",
    "mhw_last": "Recent historical MHW is present",
    "exceed90_days_ge_3": "Historical exceed90 exists for at least 3 days",
    "exceed90_days_ge_5": "Historical exceed90 exists for at least 5 days",
    "exceed90_days_ge_8": "Historical exceed90 exists for at least 8 days",
    "exceed90_last": "Recent exceed90 is present",
    "area_large": "Candidate component area is large",
    "area_small": "Candidate component area is small",
    "area_stable": "Candidate component area is constant in constructed sequence",
    "area_increasing": "Candidate component area is increasing",
    "ssta_mean_high": "Mean SSTA inside candidate is high",
    "ssta_last_high": "Recent mean SSTA inside candidate is high",
    "ssta_increasing": "Mean SSTA inside candidate is increasing",
    "context_ssta_high": "Context-region mean SSTA is high",
    "no_positive_gap_days": "No positive threshold_gap days",
    "few_exceed90_days": "Historical exceed90 days are few",
    "few_mhw_days": "Historical MHW days are few",
    "gap_last_not_positive": "Recent mean threshold_gap is not positive",
}


def import_rule_module():
    script = Path("code/21_learn_event_rules.py")
    spec = importlib.util.spec_from_file_location("rule21", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simplify_rule(rule, atom_values):
    """
    Remove atoms that are always true, e.g., area_stable.
    Keep the original rule elsewhere; this is only for visualization.
    """
    simplified = []
    removed = []

    for atom in rule:
        values = atom_values[atom]
        if values.all():
            removed.append(atom)
        elif not values.any():
            simplified.append(atom + " [always false]")
        else:
            simplified.append(atom)

    if not simplified:
        simplified = ["TRUE"]

    return simplified, removed


def format_atom(atom, spec_map):
    col, op, value = spec_map[atom]
    desc = ATOM_DESCRIPTIONS.get(atom, atom)
    return f"{atom}: {desc}\n    {col} {op} {value:.6g}"


def format_rule(rule, spec_map, atom_values=None):
    if atom_values is not None:
        simple, removed = simplify_rule(rule, atom_values)
    else:
        simple, removed = list(rule), []

    if simple == ["TRUE"]:
        text = "TRUE"
    else:
        text = "\nAND\n".join(format_atom(a.replace(' [always false]', ''), spec_map) for a in simple)

    if removed:
        text += "\n\nRemoved from visualization because always true:\n" + ", ".join(removed)

    return text


def make_atom_statistics(dfs, atom_specs, rule21):
    rows = []

    for split, df in dfs.items():
        atoms = rule21.materialize_atoms(df, atom_specs)
        y = df["y_valid"].to_numpy(dtype=np.uint8).astype(bool)

        for name, col, op, value in atom_specs:
            a = atoms[name]
            p_valid = float(a[y].mean()) if y.any() else np.nan
            p_invalid = float(a[~y].mean()) if (~y).any() else np.nan

            rows.append({
                "split": split,
                "atom": name,
                "description": ATOM_DESCRIPTIONS.get(name, name),
                "column": col,
                "op": op,
                "threshold": value,
                "p_atom_valid": p_valid,
                "p_atom_invalid": p_invalid,
                "diff_valid_minus_invalid": p_valid - p_invalid,
                "atom_rate_all": float(a.mean()),
                "is_always_true": bool(a.all()),
                "is_always_false": bool((~a).all()),
            })

    return pd.DataFrame(rows)


def evaluate_individual_rules(dfs, atom_specs, valid_rules, invalid_rules, rule21):
    rows = []
    spec_map = {name: (col, op, value) for name, col, op, value in atom_specs}

    train_atoms = rule21.materialize_atoms(dfs["train"], atom_specs)

    for target_name, rules in [("valid", valid_rules), ("invalid", invalid_rules)]:
        for rank, rule_row in enumerate(rules, start=1):
            rule = rule_row["rule"]
            simple_rule, removed_atoms = simplify_rule(rule, train_atoms)
            raw_text = " AND ".join(rule)
            simple_text = " AND ".join(simple_rule)

            for split, df in dfs.items():
                atoms = rule21.materialize_atoms(df, atom_specs)
                y_valid = df["y_valid"].to_numpy(dtype=np.uint8).astype(bool)
                y_target = y_valid if target_name == "valid" else ~y_valid

                pred = rule21.rule_mask(atoms, rule)
                m = rule21.metrics(y_target, pred)

                rows.append({
                    "target": target_name,
                    "rank": rank,
                    "split": split,
                    "raw_rule": raw_text,
                    "simplified_rule": simple_text,
                    "removed_constant_atoms": ", ".join(removed_atoms),
                    "n_terms": len(rule),
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "accuracy": m["accuracy"],
                    "tp": m["tp"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                    "tn": m["tn"],
                    "pred_pos_ratio": m["pred_pos_ratio"],
                    "true_pos_ratio": m["true_pos_ratio"],
                    "rule_detail": format_rule(rule, spec_map, train_atoms),
                })

    return pd.DataFrame(rows)


def plot_atom_discriminability(atom_df, out_file, split="test"):
    df = atom_df[atom_df["split"] == split].copy()
    df = df.sort_values("diff_valid_minus_invalid", ascending=True)

    plt.figure(figsize=(10, max(6, 0.35 * len(df))))
    plt.barh(df["atom"], df["diff_valid_minus_invalid"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("P(atom | valid) - P(atom | invalid)")
    plt.title(f"Atom discriminability on {split} set")
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()
    print("[SAVE]", out_file)


def plot_atom_rates_heatmap(atom_df, out_file, split="test"):
    df = atom_df[atom_df["split"] == split].copy()

    # Select most discriminative atoms.
    df["abs_diff"] = df["diff_valid_minus_invalid"].abs()
    df = df.sort_values("abs_diff", ascending=False).head(20)
    df = df.sort_values("diff_valid_minus_invalid", ascending=True)

    mat = df[["p_atom_invalid", "p_atom_valid"]].to_numpy()

    plt.figure(figsize=(7, max(5, 0.35 * len(df))))
    im = plt.imshow(mat, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="Atom occurrence probability")

    plt.yticks(np.arange(len(df)), df["atom"])
    plt.xticks([0, 1], ["Invalid events", "Valid events"])
    plt.title(f"Atom occurrence rates on {split} set")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()
    print("[SAVE]", out_file)


def plot_rule_metrics(rule_df, out_file, target="invalid", split="test"):
    df = rule_df[(rule_df["target"] == target) & (rule_df["split"] == split)].copy()
    df = df.sort_values("rank")

    if df.empty:
        return

    labels = [
        f"R{int(r['rank'])}: {str(r['simplified_rule'])[:45]}"
        for _, r in df.iterrows()
    ]

    y = np.arange(len(df))
    h = 0.25

    plt.figure(figsize=(11, max(5, 0.45 * len(df))))
    plt.barh(y - h, df["precision"], height=h, label="Precision")
    plt.barh(y, df["recall"], height=h, label="Recall")
    plt.barh(y + h, df["f1"], height=h, label="F1")

    plt.yticks(y, labels)
    plt.xlim(0, 1)
    plt.xlabel("Score")
    plt.title(f"{target.capitalize()} rule metrics on {split} set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()
    print("[SAVE]", out_file)


def plot_rule_card(rule_df, out_file, target="invalid", split="test"):
    df = rule_df[(rule_df["target"] == target) & (rule_df["split"] == split)].copy()
    df = df.sort_values(["precision", "f1", "recall"], ascending=False)

    if df.empty:
        return

    row = df.iloc[0]

    action = "REMOVE candidate as invalid" if target == "invalid" else "ACCEPT candidate as valid"
    title = f"Top {target} symbolic rule on {split} set"

    detail = row["rule_detail"]

    metrics_text = (
        f"Precision = {row['precision']:.4f}\n"
        f"Recall    = {row['recall']:.4f}\n"
        f"F1        = {row['f1']:.4f}\n"
        f"TP / FP   = {int(row['tp'])} / {int(row['fp'])}\n"
        f"Coverage  = {row['pred_pos_ratio']:.4f}"
    )

    fig = plt.figure(figsize=(12, 7))
    ax = plt.gca()
    ax.axis("off")

    ax.text(
        0.5, 0.94, title,
        ha="center", va="center",
        fontsize=18, fontweight="bold"
    )

    ax.text(
        0.05, 0.78,
        "IF",
        ha="left", va="center",
        fontsize=16, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black")
    )

    ax.text(
        0.15, 0.78,
        detail,
        ha="left", va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="white", edgecolor="black")
    )

    ax.text(
        0.05, 0.30,
        "THEN",
        ha="left", va="center",
        fontsize=16, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black")
    )

    ax.text(
        0.15, 0.30,
        action,
        ha="left", va="center",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="white", edgecolor="black")
    )

    ax.text(
        0.70, 0.30,
        metrics_text,
        ha="left", va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="white", edgecolor="black")
    )

    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()
    print("[SAVE]", out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table_dir", default="outputs/20_event_rule_learning")
    parser.add_argument("--out_dir", default="outputs/25_symbolic_rule_visualization")
    parser.add_argument("--max_terms", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--min_precision", type=float, default=0.50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table_dir = Path(args.table_dir)
    dfs = {
        split: pd.read_csv(table_dir / f"event_rule_table_{split}.csv")
        for split in ["train", "val", "test"]
    }

    rule21 = import_rule_module()

    train = dfs["train"]
    val = dfs["val"]

    atom_specs = rule21.build_atom_specs(train)
    train_atoms = rule21.materialize_atoms(train, atom_specs)
    val_atoms = rule21.materialize_atoms(val, atom_specs)

    y_train_valid = train["y_valid"].to_numpy(dtype=np.uint8).astype(bool)
    y_val_valid = val["y_valid"].to_numpy(dtype=np.uint8).astype(bool)

    valid_rules, _ = rule21.select_rules(
        train_atoms, val_atoms,
        y_train_valid, y_val_valid,
        max_terms=args.max_terms,
        top_k=args.top_k,
        min_precision=args.min_precision,
    )

    invalid_rules, _ = rule21.select_rules(
        train_atoms, val_atoms,
        ~y_train_valid, ~y_val_valid,
        max_terms=args.max_terms,
        top_k=args.top_k,
        min_precision=args.min_precision,
    )

    atom_df = make_atom_statistics(dfs, atom_specs, rule21)
    atom_csv = out_dir / "atom_statistics.csv"
    atom_df.to_csv(atom_csv, index=False)
    print("[SAVE]", atom_csv)

    rule_df = evaluate_individual_rules(dfs, atom_specs, valid_rules, invalid_rules, rule21)
    rule_csv = out_dir / "individual_rule_metrics.csv"
    rule_df.to_csv(rule_csv, index=False)
    print("[SAVE]", rule_csv)

    plot_atom_discriminability(
        atom_df,
        out_dir / "atom_discriminability_test.png",
        split="test",
    )

    plot_atom_rates_heatmap(
        atom_df,
        out_dir / "atom_rates_heatmap_test.png",
        split="test",
    )

    plot_rule_metrics(
        rule_df,
        out_dir / "invalid_rule_metrics_test.png",
        target="invalid",
        split="test",
    )

    plot_rule_metrics(
        rule_df,
        out_dir / "valid_rule_metrics_test.png",
        target="valid",
        split="test",
    )

    plot_rule_card(
        rule_df,
        out_dir / "top_invalid_rule_card.png",
        target="invalid",
        split="test",
    )

    plot_rule_card(
        rule_df,
        out_dir / "top_valid_rule_card.png",
        target="valid",
        split="test",
    )

    print("[DONE]", out_dir)


if __name__ == "__main__":
    main()
