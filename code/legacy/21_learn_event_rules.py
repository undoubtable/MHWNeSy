# -*- coding: utf-8 -*-
"""
Learn simple symbolic event verifier rules from event rule tables.

Rules are conjunctions of one, two, or three boolean atoms.
Valid and invalid rules are searched separately and selected on validation F1.

Outputs:
    outputs/20_event_rule_learning/
        learned_valid_rules.txt
        learned_invalid_rules.txt
        rule_metrics.csv
        event_rule_predictions_train/val/test.csv
"""

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def parse_splits(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def q(train, col, quantile, default=0.0):
    if col not in train:
        return default
    return float(train[col].quantile(quantile))


def build_atom_specs(train):
    specs = []

    def add(name, col, op, value):
        if col in train.columns:
            specs.append((name, col, op, float(value)))

    add("mean_gap_high", "mean_threshold_gap_inside_candidate_mean", ">", max(0.0, q(train, "mean_threshold_gap_inside_candidate_mean", 0.50)))
    add("last_gap_positive", "mean_threshold_gap_inside_candidate_last", ">", 0.0)
    add("positive_gap_days_ge_3", "threshold_gap_positive_days", ">=", 3)
    add("positive_gap_days_ge_5", "threshold_gap_positive_days", ">=", 5)
    add("positive_gap_days_ge_8", "threshold_gap_positive_days", ">=", 8)
    add("max_gap_high", "max_threshold_gap_inside_candidate_max", ">", q(train, "max_threshold_gap_inside_candidate_max", 0.75))
    add("min_gap_positive", "mean_threshold_gap_inside_candidate_min", ">", 0.0)

    add("mhw_days_ge_1", "historical_mhw_days", ">=", 1)
    add("mhw_days_ge_3", "historical_mhw_days", ">=", 3)
    add("mhw_days_ge_5", "historical_mhw_days", ">=", 5)
    add("mhw_last", "historical_mhw_last", ">=", 1)

    add("exceed90_days_ge_3", "historical_exceed90_days", ">=", 3)
    add("exceed90_days_ge_5", "historical_exceed90_days", ">=", 5)
    add("exceed90_days_ge_8", "historical_exceed90_days", ">=", 8)
    add("exceed90_last", "historical_exceed90_last", ">=", 1)

    add("area_large", "component_area_fraction_mean", ">", q(train, "component_area_fraction_mean", 0.75))
    add("area_small", "component_area_fraction_mean", "<=", q(train, "component_area_fraction_mean", 0.25))
    add("area_stable", "component_area_fraction_std", "<=", q(train, "component_area_fraction_std", 0.25))
    add("area_increasing", "component_area_fraction_trend", ">", 0.0)

    add("ssta_mean_high", "mean_ssta_inside_candidate_mean", ">", q(train, "mean_ssta_inside_candidate_mean", 0.75))
    add("ssta_last_high", "mean_ssta_inside_candidate_last", ">", q(train, "mean_ssta_inside_candidate_last", 0.75))
    add("ssta_increasing", "mean_ssta_inside_candidate_trend", ">", 0.0)
    add("context_ssta_high", "mean_ssta_context_mean", ">", q(train, "mean_ssta_context_mean", 0.75))

    # Negative atoms help invalid rule discovery stay interpretable.
    add("no_positive_gap_days", "threshold_gap_positive_days", "<=", 0)
    add("few_exceed90_days", "historical_exceed90_days", "<=", 1)
    add("few_mhw_days", "historical_mhw_days", "<=", 1)
    add("gap_last_not_positive", "mean_threshold_gap_inside_candidate_last", "<=", 0.0)

    # Preserve first occurrence if two semantic atoms collapse to the same name.
    seen = set()
    unique = []
    for spec in specs:
        if spec[0] in seen:
            continue
        seen.add(spec[0])
        unique.append(spec)
    return unique


def eval_atom(df, spec):
    _, col, op, value = spec
    if op == ">":
        return (df[col].to_numpy(dtype=np.float32) > value)
    if op == ">=":
        return (df[col].to_numpy(dtype=np.float32) >= value)
    if op == "<=":
        return (df[col].to_numpy(dtype=np.float32) <= value)
    if op == "<":
        return (df[col].to_numpy(dtype=np.float32) < value)
    raise ValueError(f"Unsupported op: {op}")


def materialize_atoms(df, atom_specs):
    return {spec[0]: eval_atom(df, spec) for spec in atom_specs}


def rule_mask(atom_values, rule):
    mask = None
    for atom in rule:
        if mask is None:
            mask = atom_values[atom].copy()
        else:
            mask &= atom_values[atom]
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
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_pos_ratio": float(pred.mean()),
        "true_pos_ratio": float(y_true.mean()),
    }


def search_rules(atom_values, y_target, max_terms):
    atoms = list(atom_values.keys())
    rows = []
    for k in range(1, max_terms + 1):
        for rule in combinations(atoms, k):
            pred = rule_mask(atom_values, rule)
            m = metrics(y_target, pred)
            rows.append({
                "rule": rule,
                "rule_text": " AND ".join(rule),
                "n_terms": k,
                **m,
            })
    return rows


def select_rules(train_atoms, val_atoms, y_train, y_val, max_terms, top_k, min_precision):
    candidates = search_rules(train_atoms, y_train, max_terms=max_terms)
    evaluated = []
    for row in candidates:
        pred_val = rule_mask(val_atoms, row["rule"])
        val_m = metrics(y_val, pred_val)
        out = dict(row)
        out.update({f"val_{k}": v for k, v in val_m.items()})
        evaluated.append(out)

    eligible = [r for r in evaluated if r["val_precision"] >= min_precision and r["val_tp"] > 0]
    if not eligible:
        eligible = [r for r in evaluated if r["val_tp"] > 0]
    eligible = sorted(
        eligible,
        key=lambda r: (r["val_f1"], r["val_precision"], r["val_recall"], -r["n_terms"]),
        reverse=True,
    )
    return eligible[:top_k], evaluated


def combined_prediction(atom_values, rules):
    if not rules:
        first = next(iter(atom_values.values()))
        return np.zeros_like(first, dtype=bool), np.zeros_like(first, dtype=np.int32)
    hits = np.zeros_like(next(iter(atom_values.values())), dtype=np.int32)
    for row in rules:
        hits += rule_mask(atom_values, row["rule"]).astype(np.int32)
    return hits > 0, hits


def write_rules(path, title, rules, atom_specs):
    spec_map = {name: (col, op, value) for name, col, op, value in atom_specs}
    lines = [title, ""]
    for i, row in enumerate(rules, start=1):
        lines.append(f"{i}. {row['rule_text']}")
        lines.append(
            f"   val precision={row['val_precision']:.4f}, recall={row['val_recall']:.4f}, f1={row['val_f1']:.4f}"
        )
        for atom in row["rule"]:
            col, op, value = spec_map[atom]
            lines.append(f"   - {atom}: {col} {op} {value:.6g}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print("[SAVE]", path)


def write_predictions(path, df, valid_pred, valid_hits, invalid_pred, invalid_hits):
    cols = ["split", "event_index", "y_valid", "sample_index", "component_id", "area_px", "best_iou", "overlap_ratio"]
    rows = []
    for i in range(len(df)):
        row = {c: df.iloc[i][c] for c in cols if c in df.columns}
        row.update({
            "valid_rule_hits": int(valid_hits[i]),
            "invalid_rule_hits": int(invalid_hits[i]),
            "pred_valid": int(valid_pred[i]),
            "pred_invalid": int(invalid_pred[i]),
            "remove_by_rule": int(invalid_pred[i]),
        })
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print("[SAVE]", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table_dir", type=str, default=str(cfg.OUTPUT_DIR / "20_event_rule_learning"))
    parser.add_argument("--out_dir", type=str, default=str(cfg.OUTPUT_DIR / "20_event_rule_learning"))
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--max_terms", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--min_precision", type=float, default=0.50)
    args = parser.parse_args()

    table_dir = Path(args.table_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = {split: pd.read_csv(table_dir / f"event_rule_table_{split}.csv") for split in parse_splits(args.splits)}
    train = dfs["train"]
    val = dfs["val"]

    atom_specs = build_atom_specs(train)
    train_atoms = materialize_atoms(train, atom_specs)
    val_atoms = materialize_atoms(val, atom_specs)

    y_train_valid = train["y_valid"].to_numpy(dtype=np.uint8).astype(bool)
    y_val_valid = val["y_valid"].to_numpy(dtype=np.uint8).astype(bool)

    valid_rules, valid_candidates = select_rules(
        train_atoms, val_atoms, y_train_valid, y_val_valid,
        max_terms=args.max_terms, top_k=args.top_k, min_precision=args.min_precision,
    )
    invalid_rules, invalid_candidates = select_rules(
        train_atoms, val_atoms, ~y_train_valid, ~y_val_valid,
        max_terms=args.max_terms, top_k=args.top_k, min_precision=args.min_precision,
    )

    write_rules(out_dir / "learned_valid_rules.txt", "Learned valid event rules", valid_rules, atom_specs)
    write_rules(out_dir / "learned_invalid_rules.txt", "Learned invalid event rules", invalid_rules, atom_specs)

    metric_rows = []
    for split, df in dfs.items():
        atoms = materialize_atoms(df, atom_specs)
        y_valid = df["y_valid"].to_numpy(dtype=np.uint8).astype(bool)
        valid_pred, valid_hits = combined_prediction(atoms, valid_rules)
        invalid_pred, invalid_hits = combined_prediction(atoms, invalid_rules)

        for target_name, target_y, pred in [
            ("valid", y_valid, valid_pred),
            ("invalid", ~y_valid, invalid_pred),
        ]:
            row = metrics(target_y, pred)
            row.update({"split": split, "target": target_name, "n_rules": args.top_k})
            metric_rows.append(row)

        write_predictions(
            out_dir / f"event_rule_predictions_{split}.csv",
            df=df,
            valid_pred=valid_pred,
            valid_hits=valid_hits,
            invalid_pred=invalid_pred,
            invalid_hits=invalid_hits,
        )

    metrics_csv = out_dir / "rule_metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(metric_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    print("[SAVE]", metrics_csv)

    # Keep full candidate rankings for auditability.
    pd.DataFrame(valid_candidates).drop(columns=["rule"], errors="ignore").to_csv(
        out_dir / "valid_rule_candidates.csv", index=False
    )
    pd.DataFrame(invalid_candidates).drop(columns=["rule"], errors="ignore").to_csv(
        out_dir / "invalid_rule_candidates.csv", index=False
    )


if __name__ == "__main__":
    main()
