#!/usr/bin/env python
"""Strict validation-to-test point residual correction.

Rules are selected only on the validation split. The selected remove/add/
combined rules are then applied once to the test split for final evaluation.
The test split is never used for threshold selection or rule selection.
"""

from __future__ import annotations

import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd


cfg = SourceFileLoader("test_cfg", str(Path(__file__).with_name("00_test_config.py"))).load_module()
residual = SourceFileLoader(
    "point_residual_helpers",
    str(Path(__file__).with_name("12_learn_point_residual_correction_rules.py")),
).load_module()

RULE_CSV = cfg.TEST_OUTPUT_DIR / "point_residual_val_to_test_rules.csv"
SUMMARY_JSON = cfg.TEST_OUTPUT_DIR / "point_residual_val_to_test_summary.json"
PRED_FILE = cfg.TEST_OUTPUT_DIR / "point_residual_val_to_test_corrected_predictions.npz"


def jsonable(row: pd.Series | None) -> dict[str, object] | None:
    if row is None:
        return None
    out: dict[str, object] = {}
    for key, value in row.to_dict().items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def metric_subset(metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
    }


def add_output_context(
    row: dict[str, object],
    selection_set: str,
    evaluation_set: str,
) -> dict[str, object]:
    return {
        "selection_set": selection_set,
        "evaluation_set": evaluation_set,
        "rule_type": row["rule_type"],
        "rule_name": row["rule_name"],
        "rule": row["rule"],
        "support": row["support"],
        "correction_precision": row["correction_precision"],
        "before_precision": row["before_precision"],
        "before_recall": row["before_recall"],
        "before_f1": row["before_f1"],
        "after_precision": row["after_precision"],
        "after_recall": row["after_recall"],
        "after_f1": row["after_f1"],
        "delta_precision": row["delta_precision"],
        "delta_recall": row["delta_recall"],
        "delta_f1": row["delta_f1"],
        "before_accuracy": row["before_accuracy"],
        "after_accuracy": row["after_accuracy"],
        "correctly_removed_fp": row.get("correctly_removed_fp", 0),
        "wrongly_removed_tp": row.get("wrongly_removed_tp", 0),
        "correctly_added_fn": row.get("correctly_added_fn", 0),
        "wrongly_added_tn": row.get("wrongly_added_tn", 0),
    }


def choose_best(df: pd.DataFrame) -> pd.Series:
    return df.sort_values(
        ["delta_f1", "correction_precision", "support"],
        ascending=[False, False, False],
    ).iloc[0]


def apply_named_rule(
    rule_row: pd.Series,
    base_pred: np.ndarray,
    remove_masks: dict[str, np.ndarray],
    add_masks: dict[str, np.ndarray],
) -> np.ndarray:
    rule_type = str(rule_row["rule_type"])
    rule_name = str(rule_row["rule_name"])
    zeros = np.zeros_like(base_pred, dtype=bool)
    if rule_type == "remove":
        return residual.apply_correction(base_pred, remove_masks[rule_name], zeros)
    if rule_type == "add":
        return residual.apply_correction(base_pred, zeros, add_masks[rule_name])

    remove_name, add_name = rule_name.split("__PLUS__")
    return residual.apply_correction(base_pred, remove_masks[remove_name], add_masks[add_name])


def evaluate_named_rule(
    rule_row: pd.Series,
    y: np.ndarray,
    base_pred: np.ndarray,
    base_metrics: dict[str, float | int],
    remove_masks: dict[str, np.ndarray],
    add_masks: dict[str, np.ndarray],
) -> dict[str, object]:
    rule_type = str(rule_row["rule_type"])
    rule_name = str(rule_row["rule_name"])
    zeros = np.zeros_like(base_pred, dtype=bool)
    if rule_type == "remove":
        return residual.evaluate_candidate(
            rule_type,
            rule_name,
            str(rule_row["rule"]),
            y,
            base_pred,
            base_metrics,
            remove_masks[rule_name],
            zeros,
        )
    if rule_type == "add":
        return residual.evaluate_candidate(
            rule_type,
            rule_name,
            str(rule_row["rule"]),
            y,
            base_pred,
            base_metrics,
            zeros,
            add_masks[rule_name],
        )

    remove_name, add_name = rule_name.split("__PLUS__")
    return residual.evaluate_candidate(
        rule_type,
        rule_name,
        str(rule_row["rule"]),
        y,
        base_pred,
        base_metrics,
        remove_masks[remove_name],
        add_masks[add_name],
    )


def main() -> None:
    cfg.ensure_dirs()
    base = residual.choose_base_model()
    if base["y_val"] is None or base["val_prob"] is None:
        raise SystemExit(
            "[ERROR] Strict val-to-test correction requires validation predictions. "
            f"Found base_model_used={base['base_model_used']} threshold_source={base['threshold_source']}"
        )

    threshold = float(base["threshold"])
    y_val = base["y_val"]
    y_test = base["y_test"]
    val_prob = base["val_prob"]
    test_prob = base["test_prob"]
    val_base_pred = (val_prob >= threshold).astype(np.uint8)
    test_base_pred = (test_prob >= threshold).astype(np.uint8)
    val_base_metrics = residual.binary_metrics(y_val, val_base_pred)
    test_base_metrics = residual.binary_metrics(y_test, test_base_pred)

    val_features, test_features, feature_source, feature_names = residual.extract_features(base)
    if not val_features:
        raise SystemExit("[ERROR] Validation correction features are unavailable; cannot select rules on val.")

    remove_val_rules = residual.make_rules_for_split(val_features, val_prob, val_base_pred, threshold, "remove")
    add_val_rules = residual.make_rules_for_split(val_features, val_prob, val_base_pred, threshold, "add")
    remove_test_rules = residual.make_rules_for_split(test_features, test_prob, test_base_pred, threshold, "remove")
    add_test_rules = residual.make_rules_for_split(test_features, test_prob, test_base_pred, threshold, "add")

    remove_val_masks = {name: mask for name, _, mask in remove_val_rules}
    add_val_masks = {name: mask for name, _, mask in add_val_rules}
    remove_test_masks = {name: mask for name, _, mask in remove_test_rules}
    add_test_masks = {name: mask for name, _, mask in add_test_rules}

    val_rows: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []

    for name, expr, mask in remove_val_rules:
        row = residual.evaluate_candidate(
            "remove",
            name,
            expr,
            y_val,
            val_base_pred,
            val_base_metrics,
            mask,
            np.zeros_like(mask, dtype=bool),
        )
        val_rows.append(row)
        csv_rows.append(add_output_context(row, "val", "val"))

    for name, expr, mask in add_val_rules:
        row = residual.evaluate_candidate(
            "add",
            name,
            expr,
            y_val,
            val_base_pred,
            val_base_metrics,
            np.zeros_like(mask, dtype=bool),
            mask,
        )
        val_rows.append(row)
        csv_rows.append(add_output_context(row, "val", "val"))

    val_df = pd.DataFrame(val_rows)
    remove_df = val_df[val_df["rule_type"] == "remove"].copy()
    add_df = val_df[val_df["rule_type"] == "add"].copy()
    best_remove = choose_best(remove_df)
    best_add = choose_best(add_df)

    remove_top = remove_df.sort_values(
        ["delta_f1", "correction_precision", "support"],
        ascending=[False, False, False],
    ).head(20)
    add_top = add_df.sort_values(
        ["delta_f1", "correction_precision", "support"],
        ascending=[False, False, False],
    ).head(20)

    combo_rows: list[dict[str, object]] = []
    for _, remove_row in remove_top.iterrows():
        for _, add_row in add_top.iterrows():
            remove_name = str(remove_row["rule_name"])
            add_name = str(add_row["rule_name"])
            combo_name = f"{remove_name}__PLUS__{add_name}"
            combo_rule = f"REMOVE[{remove_row['rule']}] + ADD[{add_row['rule']}]"
            row = residual.evaluate_candidate(
                "remove+add",
                combo_name,
                combo_rule,
                y_val,
                val_base_pred,
                val_base_metrics,
                remove_val_masks[remove_name],
                add_val_masks[add_name],
            )
            combo_rows.append(row)
            csv_rows.append(add_output_context(row, "val", "val"))

    combo_df = pd.DataFrame(combo_rows)
    best_combined = choose_best(combo_df)

    selected_df = pd.DataFrame([best_remove, best_add, best_combined])
    best_overall = choose_best(selected_df)

    test_selected_rows = []
    for selected_name, selected_row in (
        ("best_remove_rule_on_val", best_remove),
        ("best_add_rule_on_val", best_add),
        ("best_combined_rule_on_val", best_combined),
    ):
        test_row = evaluate_named_rule(
            selected_row,
            y_test,
            test_base_pred,
            test_base_metrics,
            remove_test_masks,
            add_test_masks,
        )
        test_row["selected_as"] = selected_name
        test_selected_rows.append(test_row)
        csv_rows.append(add_output_context(test_row, "val", "test"))

    corrected_pred = apply_named_rule(best_overall, test_base_pred, remove_test_masks, add_test_masks)
    corrected_metrics = residual.binary_metrics(y_test, corrected_pred)
    test_delta_metrics = {
        "delta_precision": float(corrected_metrics["precision"]) - float(test_base_metrics["precision"]),
        "delta_recall": float(corrected_metrics["recall"]) - float(test_base_metrics["recall"]),
        "delta_f1": float(corrected_metrics["f1"]) - float(test_base_metrics["f1"]),
    }

    pd.DataFrame(csv_rows).to_csv(RULE_CSV, index=False)
    np.savez_compressed(
        PRED_FILE,
        y_true=y_test,
        y_prob=test_prob,
        base_pred=test_base_pred,
        corrected_pred=corrected_pred,
        best_val_threshold=np.array(threshold, dtype=np.float32),
        selected_rule_name=np.array(str(best_overall["rule_name"]), dtype=object),
        selected_rule_type=np.array(str(best_overall["rule_type"]), dtype=object),
    )

    summary = {
        "base_model_used": base["base_model_used"],
        "threshold_source": base["threshold_source"],
        "threshold_warning": base["warning"],
        "best_val_threshold": threshold,
        "feature_source": feature_source,
        "feature_names": feature_names,
        "val_base_metrics": metric_subset(val_base_metrics),
        "val_best_remove_rule": jsonable(best_remove),
        "val_best_add_rule": jsonable(best_add),
        "val_best_combined_rule": jsonable(best_combined),
        "val_selected_overall_rule": jsonable(best_overall),
        "test_base_metrics": metric_subset(test_base_metrics),
        "test_evaluation_of_val_selected_rules": {
            row["selected_as"]: jsonable(pd.Series(row)) for row in test_selected_rows
        },
        "test_corrected_metrics": metric_subset(corrected_metrics),
        "test_delta_metrics": test_delta_metrics,
    }
    with SUMMARY_JSON.open("w") as f:
        json.dump(summary, f, indent=2)

    print("Rules were selected on validation set and applied once to test set.")
    print(f"[SAVED] {RULE_CSV}")
    print(f"[SAVED] {SUMMARY_JSON}")
    print(f"[SAVED] {PRED_FILE}")
    print("[VAL BASE METRICS]")
    print({k: val_base_metrics[k] for k in ("accuracy", "precision", "recall", "f1")})
    print("[VAL BEST REMOVE RULE]")
    print(best_remove.to_string())
    print("[VAL BEST ADD RULE]")
    print(best_add.to_string())
    print("[VAL BEST COMBINED RULE]")
    print(best_combined.to_string())
    print("[TEST BASE METRICS]")
    print({k: test_base_metrics[k] for k in ("accuracy", "precision", "recall", "f1")})
    print("[TEST CORRECTED METRICS AFTER APPLYING FIXED VAL-SELECTED RULE]")
    print({k: corrected_metrics[k] for k in ("accuracy", "precision", "recall", "f1")})
    print(f"final test delta_f1: {test_delta_metrics['delta_f1']:.6f}")


if __name__ == "__main__":
    main()
