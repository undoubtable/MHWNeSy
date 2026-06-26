#!/usr/bin/env python
"""Evaluate validation-selected residual classifiers once on the test split."""

from __future__ import annotations

import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd


cfg = SourceFileLoader("test_cfg", str(Path(__file__).with_name("00_test_config.py"))).load_module()
train_helpers = SourceFileLoader(
    "point_residual_classifier_train_helpers",
    str(Path(__file__).with_name("14_train_point_residual_classifier.py")),
).load_module()


def require_eval_deps():
    try:
        from joblib import load
        from sklearn.inspection import permutation_importance
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ModuleNotFoundError as exc:
        raise SystemExit("[MISSING DEPENDENCY] scikit-learn/joblib is required for evaluation.") from exc
    return load, permutation_importance, DecisionTreeClassifier, export_text


def try_auc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ModuleNotFoundError:
        return float("nan"), float("nan")
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    try:
        return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan"), float("nan")


def apply_residual_correction(
    y: np.ndarray,
    base_pred: np.ndarray,
    remove_scores: np.ndarray,
    add_scores: np.ndarray,
    remove_threshold: float,
    add_threshold: float,
) -> tuple[np.ndarray, dict[str, int]]:
    corrected = base_pred.astype(np.uint8).copy()
    remove_mask = (base_pred == 1) & (remove_scores >= remove_threshold)
    add_mask = (base_pred == 0) & (add_scores >= add_threshold)
    corrected[remove_mask] = 0
    corrected[add_mask] = 1
    y_bool = y.astype(bool)
    base_bool = base_pred.astype(bool)
    stats = {
        "num_removed_test": int(remove_mask.sum()),
        "num_added_test": int(add_mask.sum()),
        "correctly_removed_fp": int((remove_mask & base_bool & ~y_bool).sum()),
        "wrongly_removed_tp": int((remove_mask & base_bool & y_bool).sum()),
        "correctly_added_fn": int((add_mask & ~base_bool & y_bool).sum()),
        "wrongly_added_tn": int((add_mask & ~base_bool & ~y_bool).sum()),
    }
    return corrected, stats


def add_delta(base: dict[str, float | int], corrected: dict[str, float | int]) -> dict[str, float]:
    return {
        "delta_accuracy": float(corrected["accuracy"]) - float(base["accuracy"]),
        "delta_precision": float(corrected["precision"]) - float(base["precision"]),
        "delta_recall": float(corrected["recall"]) - float(base["recall"]),
        "delta_f1": float(corrected["f1"]) - float(base["f1"]),
    }


def classifier_feature_rows(
    label: str,
    model,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    feature_names: list[str],
    permutation_importance,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if hasattr(model, "coef_"):
        coef = model.coef_[0]
        positive_idx = [idx for idx in np.argsort(coef)[::-1] if coef[int(idx)] > 0]
        negative_idx = [idx for idx in np.argsort(coef) if coef[int(idx)] < 0]
        for rank, idx in enumerate(positive_idx[:20], start=1):
            rows.append(
                {
                    "classifier": label,
                    "importance_type": "positive_coefficient",
                    "rank": rank,
                    "feature": feature_names[int(idx)],
                    "value": float(coef[int(idx)]),
                }
            )
        for rank, idx in enumerate(negative_idx[:20], start=1):
            rows.append(
                {
                    "classifier": label,
                    "importance_type": "negative_coefficient",
                    "rank": rank,
                    "feature": feature_names[int(idx)],
                    "value": float(coef[int(idx)]),
                }
            )
        return rows

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        for rank, idx in enumerate(np.argsort(imp)[::-1][:20], start=1):
            rows.append(
                {
                    "classifier": label,
                    "importance_type": "feature_importance",
                    "rank": rank,
                    "feature": feature_names[int(idx)],
                    "value": float(imp[int(idx)]),
                }
            )
        return rows

    # HistGradientBoostingClassifier has no native feature_importances_.
    # Use validation-split permutation importance for explanation only.
    if len(np.unique(y_ref)) >= 2 and len(y_ref) > 0:
        try:
            n = min(len(y_ref), 5000)
            result = permutation_importance(
                model,
                x_ref[:n],
                y_ref[:n],
                n_repeats=5,
                random_state=cfg.RANDOM_SEED,
                scoring="f1",
            )
            imp = result.importances_mean
            for rank, idx in enumerate(np.argsort(imp)[::-1][:20], start=1):
                rows.append(
                    {
                        "classifier": label,
                        "importance_type": "permutation_importance_val",
                        "rank": rank,
                        "feature": feature_names[int(idx)],
                        "value": float(imp[int(idx)]),
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "classifier": label,
                    "importance_type": "warning",
                    "rank": 0,
                    "feature": "permutation_importance_failed",
                    "value": str(exc),
                }
            )
    return rows


def main() -> None:
    cfg.ensure_dirs()
    load, permutation_importance, DecisionTreeClassifier, export_text = require_eval_deps()

    if not cfg.POINT_RESIDUAL_CLASSIFIER_METADATA_JSON.exists():
        raise SystemExit(f"[MISSING] {cfg.POINT_RESIDUAL_CLASSIFIER_METADATA_JSON}")
    if not cfg.POINT_RESIDUAL_REMOVE_CLASSIFIER_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.POINT_RESIDUAL_REMOVE_CLASSIFIER_FILE}")
    if not cfg.POINT_RESIDUAL_ADD_CLASSIFIER_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.POINT_RESIDUAL_ADD_CLASSIFIER_FILE}")

    metadata = json.loads(cfg.POINT_RESIDUAL_CLASSIFIER_METADATA_JSON.read_text())
    threshold = float(metadata["best_val_threshold"])
    remove_threshold = float(metadata["remove_threshold"])
    add_threshold = float(metadata["add_threshold"])
    feature_names = [str(x) for x in metadata["feature_names"]]

    base = train_helpers.predict_original_point_lstm(batch_size=4096)
    feature_data, feature_source = train_helpers.load_physics_dataset_or_base(base)
    raw_feature_names = [str(x) for x in feature_data["feature_names"].tolist()]
    x_train, feature_names_train = train_helpers.correction_features(
        feature_data["X_train"], raw_feature_names, base["train_prob"], threshold
    )
    x_val, feature_names_val = train_helpers.correction_features(
        feature_data["X_val"], raw_feature_names, base["val_prob"], threshold
    )
    x_test, feature_names_test = train_helpers.correction_features(
        feature_data["X_test"], raw_feature_names, base["test_prob"], threshold
    )
    if feature_names != feature_names_test or feature_names != feature_names_val or feature_names != feature_names_train:
        raise SystemExit("[ERROR] Feature names from current data do not match training metadata.")

    y_train = base["y_train"]
    y_val = base["y_val"]
    y_test = base["y_test"]
    train_base_pred = (base["train_prob"] >= threshold).astype(np.uint8)
    val_base_pred = (base["val_prob"] >= threshold).astype(np.uint8)
    test_base_pred = (base["test_prob"] >= threshold).astype(np.uint8)

    remove_model = load(cfg.POINT_RESIDUAL_REMOVE_CLASSIFIER_FILE)
    add_model = load(cfg.POINT_RESIDUAL_ADD_CLASSIFIER_FILE)
    remove_scores = np.zeros(len(y_test), dtype=np.float32)
    add_scores = np.zeros(len(y_test), dtype=np.float32)
    remove_idx_test = test_base_pred == 1
    add_idx_test = test_base_pred == 0
    remove_scores[remove_idx_test] = train_helpers.classifier_scores(remove_model, x_test[remove_idx_test])
    add_scores[add_idx_test] = train_helpers.classifier_scores(add_model, x_test[add_idx_test])

    corrected_pred, correction_stats = apply_residual_correction(
        y_test,
        test_base_pred,
        remove_scores,
        add_scores,
        remove_threshold,
        add_threshold,
    )
    base_metrics = train_helpers.binary_metrics(y_test, test_base_pred)
    corrected_metrics = train_helpers.binary_metrics(y_test, corrected_pred)
    delta_metrics = add_delta(base_metrics, corrected_metrics)
    base_roc_auc, base_pr_auc = try_auc_metrics(y_test, base["test_prob"])

    remove_idx_train = train_base_pred == 1
    add_idx_train = train_base_pred == 0
    remove_target_train = (y_train[remove_idx_train] == 0).astype(np.uint8)
    add_target_train = (y_train[add_idx_train] == 1).astype(np.uint8)
    remove_idx_val = val_base_pred == 1
    add_idx_val = val_base_pred == 0
    remove_target_val = (y_val[remove_idx_val] == 0).astype(np.uint8)
    add_target_val = (y_val[add_idx_val] == 1).astype(np.uint8)

    importance_rows = []
    importance_rows.extend(
        classifier_feature_rows(
            "remove",
            remove_model,
            x_val[remove_idx_val],
            remove_target_val,
            feature_names,
            permutation_importance,
        )
    )
    importance_rows.extend(
        classifier_feature_rows(
            "add",
            add_model,
            x_val[add_idx_val],
            add_target_val,
            feature_names,
            permutation_importance,
        )
    )
    pd.DataFrame(importance_rows).to_csv(cfg.POINT_RESIDUAL_CLASSIFIER_FEATURE_IMPORTANCE_FILE, index=False)

    remove_surrogate = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=cfg.RANDOM_SEED)
    add_surrogate = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=cfg.RANDOM_SEED)
    remove_surrogate.fit(x_train[remove_idx_train], remove_target_train)
    add_surrogate.fit(x_train[add_idx_train], add_target_train)
    surrogate_text = [
        "residual classifiers were trained on train set, selected on validation set, and evaluated once on test set.",
        "",
        "[REMOVE SURROGATE max_depth=3]",
        export_text(remove_surrogate, feature_names=feature_names),
        "",
        "[ADD SURROGATE max_depth=3]",
        export_text(add_surrogate, feature_names=feature_names),
    ]
    cfg.POINT_RESIDUAL_CLASSIFIER_SURROGATE_RULES_FILE.write_text("\n".join(surrogate_text))

    summary_row = {
        "base_model": metadata["base_model"],
        "best_val_threshold": threshold,
        "remove_model_name": metadata["remove_model_name"],
        "add_model_name": metadata["add_model_name"],
        "remove_threshold": remove_threshold,
        "add_threshold": add_threshold,
        "val_base_f1": metadata["best_val_search_row"]["val_base_f1"],
        "val_corrected_f1": metadata["best_val_search_row"]["val_corrected_f1"],
        "val_delta_f1": metadata["best_val_search_row"]["val_delta_f1"],
        "test_base_accuracy": base_metrics["accuracy"],
        "test_base_precision": base_metrics["precision"],
        "test_base_recall": base_metrics["recall"],
        "test_base_f1": base_metrics["f1"],
        "test_corrected_accuracy": corrected_metrics["accuracy"],
        "test_corrected_precision": corrected_metrics["precision"],
        "test_corrected_recall": corrected_metrics["recall"],
        "test_corrected_f1": corrected_metrics["f1"],
        "test_delta_accuracy": delta_metrics["delta_accuracy"],
        "test_delta_precision": delta_metrics["delta_precision"],
        "test_delta_recall": delta_metrics["delta_recall"],
        "test_delta_f1": delta_metrics["delta_f1"],
        "test_base_roc_auc": base_roc_auc,
        "test_base_pr_auc": base_pr_auc,
        "delta_roc_auc": float("nan"),
        "delta_pr_auc": float("nan"),
        **correction_stats,
    }
    pd.DataFrame([summary_row]).to_csv(cfg.POINT_RESIDUAL_CLASSIFIER_TEST_SUMMARY_CSV, index=False)

    summary_json = {
        "split_protocol": "residual classifiers were trained on train set, selected on validation set, and evaluated once on test set.",
        "metadata": metadata,
        "feature_source_eval": feature_source,
        "test_base_metrics": {k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in base_metrics.items()},
        "test_corrected_metrics": {k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in corrected_metrics.items()},
        "test_delta_metrics": delta_metrics,
        "test_base_roc_auc": base_roc_auc,
        "test_base_pr_auc": base_pr_auc,
        "correction_stats": correction_stats,
    }
    with cfg.POINT_RESIDUAL_CLASSIFIER_TEST_SUMMARY_JSON.open("w") as f:
        json.dump(summary_json, f, indent=2)

    np.savez_compressed(
        cfg.POINT_RESIDUAL_CLASSIFIER_CORRECTED_PRED_FILE,
        y_true=y_test,
        y_prob=base["test_prob"],
        base_pred=test_base_pred,
        corrected_pred=corrected_pred,
        remove_scores=remove_scores,
        add_scores=add_scores,
        best_val_threshold=np.array(threshold, dtype=np.float32),
        remove_threshold=np.array(remove_threshold, dtype=np.float32),
        add_threshold=np.array(add_threshold, dtype=np.float32),
    )

    print("residual classifiers were trained on train set, selected on validation set, and evaluated once on test set.")
    print(f"[SAVED] {cfg.POINT_RESIDUAL_CLASSIFIER_TEST_SUMMARY_JSON}")
    print(f"[SAVED] {cfg.POINT_RESIDUAL_CLASSIFIER_TEST_SUMMARY_CSV}")
    print(f"[SAVED] {cfg.POINT_RESIDUAL_CLASSIFIER_CORRECTED_PRED_FILE}")
    print(f"[SAVED] {cfg.POINT_RESIDUAL_CLASSIFIER_FEATURE_IMPORTANCE_FILE}")
    print(f"[SAVED] {cfg.POINT_RESIDUAL_CLASSIFIER_SURROGATE_RULES_FILE}")
    print("[TEST SUMMARY]")
    print(pd.DataFrame([summary_row]).to_string(index=False))


if __name__ == "__main__":
    main()
