#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learn NeurRL-style region rules from region atoms.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


LEAK_COLUMNS = {
    "split", "event_index", "y_valid", "sample_index", "component_id",
    "area_px", "best_iou", "overlap_ratio",
}


def parse_splits(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def atom_columns(df):
    return [
        c for c in df.columns
        if c not in LEAK_COLUMNS and (
            c.endswith("_ACTIVE") or c.endswith("_HIGH") or c.endswith("_LOW")
        )
    ]


def metrics(y, pred):
    y = y.astype(bool)
    pred = pred.astype(bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
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
        "support": int(pred.sum()),
    }


def rule_pred(df, rule):
    pred = np.ones(len(df), dtype=bool)
    for atom in rule:
        pred &= df[atom].to_numpy(dtype=np.uint8).astype(bool)
    return pred


def search(train, val, target_name, max_terms, min_precision):
    atoms = atom_columns(train)
    y_train = train["y_valid"].to_numpy(dtype=np.uint8).astype(bool)
    y_val = val["y_valid"].to_numpy(dtype=np.uint8).astype(bool)
    if target_name == "invalid":
        y_train = ~y_train
        y_val = ~y_val

    rows = []
    for k in range(1, max_terms + 1):
        for rule in combinations(atoms, k):
            pred_train = rule_pred(train, rule)
            pred_val = rule_pred(val, rule)
            if pred_train.sum() == 0 or pred_val.sum() == 0:
                continue
            train_m = metrics(y_train, pred_train)
            val_m = metrics(y_val, pred_val)
            if val_m["tp"] <= 0:
                continue
            rows.append({
                "target": target_name,
                "rule": rule,
                "rule_text": " AND ".join(rule),
                "n_terms": k,
                **{f"train_{kk}": vv for kk, vv in train_m.items()},
                **{f"val_{kk}": vv for kk, vv in val_m.items()},
            })
    eligible = [r for r in rows if r["val_precision"] >= min_precision]
    if not eligible:
        eligible = rows
    return sorted(eligible, key=lambda r: (r["val_f1"], r["val_precision"], r["val_recall"]), reverse=True), rows


def write_rules(path, title, rows, top_k):
    lines = [title, ""]
    for rank, row in enumerate(rows[:top_k], start=1):
        then = "valid" if row["target"] == "valid" else "invalid"
        lines.append(f"{rank}. IF {row['rule_text']} THEN {then}")
        lines.append(
            f"   val precision={row['val_precision']:.4f}, recall={row['val_recall']:.4f}, f1={row['val_f1']:.4f}"
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print("[SAVE]", path)


def combined(df, rules):
    hits = np.zeros(len(df), dtype=np.int32)
    for row in rules:
        hits += rule_pred(df, row["rule"]).astype(np.int32)
    return hits > 0, hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atom_dir", type=str, default=str(cfg.OUTPUT_DIR / "26_region_rule_learning"))
    parser.add_argument("--out_dir", type=str, default=str(cfg.OUTPUT_DIR / "26_region_rule_learning"))
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--max_terms", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--min_precision", type=float, default=0.50)
    args = parser.parse_args()

    atom_dir = Path(args.atom_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dfs = {s: pd.read_csv(atom_dir / f"region_atoms_{s}.csv") for s in parse_splits(args.splits)}

    valid_ranked, valid_all = search(dfs["train"], dfs["val"], "valid", args.max_terms, args.min_precision)
    invalid_ranked, invalid_all = search(dfs["train"], dfs["val"], "invalid", args.max_terms, args.min_precision)
    valid_rules = valid_ranked[:args.top_k]
    invalid_rules = invalid_ranked[:args.top_k]

    write_rules(out_dir / "learned_valid_region_rules.txt", "Learned valid region rules", valid_rules, args.top_k)
    write_rules(out_dir / "learned_invalid_region_rules.txt", "Learned invalid region rules", invalid_rules, args.top_k)

    metric_rows = []
    for split, df in dfs.items():
        y_valid = df["y_valid"].to_numpy(dtype=np.uint8).astype(bool)
        valid_pred, valid_hits = combined(df, valid_rules)
        invalid_pred, invalid_hits = combined(df, invalid_rules)
        for target, y, pred in [("valid", y_valid, valid_pred), ("invalid", ~y_valid, invalid_pred)]:
            row = metrics(y, pred)
            row.update({"split": split, "target": target})
            metric_rows.append(row)

        out = df[["split", "event_index", "y_valid", "sample_index", "component_id", "area_px", "best_iou", "overlap_ratio"]].copy()
        out["valid_region_rule_hits"] = valid_hits
        out["invalid_region_rule_hits"] = invalid_hits
        out["pred_valid_region"] = valid_pred.astype(np.uint8)
        out["pred_invalid_region"] = invalid_pred.astype(np.uint8)
        out.to_csv(out_dir / f"region_rule_predictions_{split}.csv", index=False)
        print("[SAVE]", out_dir / f"region_rule_predictions_{split}.csv")

    pd.DataFrame(metric_rows).to_csv(out_dir / "region_rule_metrics.csv", index=False)
    pd.DataFrame(valid_all).drop(columns=["rule"], errors="ignore").to_csv(out_dir / "valid_region_rule_candidates.csv", index=False)
    pd.DataFrame(invalid_all).drop(columns=["rule"], errors="ignore").to_csv(out_dir / "invalid_region_rule_candidates.csv", index=False)
    print("[SAVE]", out_dir / "region_rule_metrics.csv")


if __name__ == "__main__":
    main()
