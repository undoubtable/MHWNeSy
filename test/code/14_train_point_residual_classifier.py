#!/usr/bin/env python
"""Train residual classifiers for point-level forecast correction.

Strict split discipline:

* train split: fit remove/add residual classifiers
* validation split: choose base threshold, model family, and correction thresholds
* test split: untouched here, except probabilities/features are prepared for
  shape checks and later evaluation by 15_eval_point_residual_classifier.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd


cfg = SourceFileLoader("test_cfg", str(Path(__file__).with_name("00_test_config.py"))).load_module()


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[MISSING DEPENDENCY] torch is required to compute base point LSTM predictions.\n"
            "Use: conda run -n ybzcu121 python test/code/14_train_point_residual_classifier.py"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def require_sklearn():
    try:
        from joblib import dump
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[MISSING DEPENDENCY] scikit-learn/joblib is required for residual classifiers."
        ) from exc
    return {
        "dump": dump,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "DecisionTreeClassifier": DecisionTreeClassifier,
    }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.logical_and(y_true, y_pred).sum())
    fp = int(np.logical_and(~y_true, y_pred).sum())
    fn = int(np.logical_and(y_true, ~y_pred).sum())
    tn = int(np.logical_and(~y_true, ~y_pred).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_threshold_from_val(y_val: np.ndarray, val_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.arange(0.05, 1.00, 0.05):
        metrics = binary_metrics(y_val, val_prob >= threshold)
        rows.append({"threshold": float(round(threshold, 2)), **metrics})
    sweep = pd.DataFrame(rows)
    best = sweep.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
    return float(best["threshold"]), sweep


def predict_original_point_lstm(batch_size: int) -> dict[str, object]:
    """Compute train/val/test probabilities from the fixed original point LSTM."""

    if not cfg.POINT_DATASET_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.POINT_DATASET_FILE}")
    if not cfg.POINT_LSTM_MODEL_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.POINT_LSTM_MODEL_FILE}")

    torch, nn, DataLoader, TensorDataset = require_torch()
    data = np.load(cfg.POINT_DATASET_FILE, allow_pickle=True)
    checkpoint = torch.load(cfg.POINT_LSTM_MODEL_FILE, map_location="cpu")

    class PointLSTM(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (hidden, _) = self.lstm(x)
            return self.head(hidden[-1]).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointLSTM(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def predict(x: np.ndarray, split_name: str) -> np.ndarray:
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x.astype(np.float32))),
            batch_size=batch_size,
            shuffle=False,
        )
        probs = []
        with torch.no_grad():
            for batch_id, (xb,) in enumerate(loader, start=1):
                logits = model(xb.to(device))
                probs.append(torch.sigmoid(logits).cpu().numpy())
                if batch_id == 1 or batch_id == len(loader):
                    print(f"[BASE PREDICT {split_name}] batch={batch_id}/{len(loader)}")
        return np.concatenate(probs).astype(np.float32)

    return {
        "base_model": "original_point_lstm",
        "dataset": data,
        "train_prob": predict(data["X_train"], "TRAIN"),
        "val_prob": predict(data["X_val"], "VAL"),
        "test_prob": predict(data["X_test"], "TEST"),
        "y_train": data["y_train"].astype(np.uint8),
        "y_val": data["y_val"].astype(np.uint8),
        "y_test": data["y_test"].astype(np.uint8),
        "train_points": data["train_points"],
        "val_points": data["val_points"],
        "test_points": data["test_points"],
    }


def load_physics_dataset_or_base(base: dict[str, object]) -> tuple[dict[str, np.ndarray], str]:
    """Prefer aligned physics features, otherwise use the original point dataset."""

    base_data = base["dataset"]
    if cfg.POINT_PHYSICS_DATASET_FILE.exists():
        physics = np.load(cfg.POINT_PHYSICS_DATASET_FILE, allow_pickle=True)
        aligned = (
            np.array_equal(base["train_points"], physics["train_points"])
            and np.array_equal(base["val_points"], physics["val_points"])
            and np.array_equal(base["test_points"], physics["test_points"])
        )
        if aligned:
            print("[FEATURES] using aligned physics dataset")
            return dict(physics), "aligned_physics_dataset"
        print("[WARN] physics dataset is not aligned; falling back to base point dataset")
        print("base train/val/test:", base["train_points"].shape, base["val_points"].shape, base["test_points"].shape)
        print("physics train/val/test:", physics["train_points"].shape, physics["val_points"].shape, physics["test_points"].shape)
        print("physics feature_names:", physics["feature_names"].tolist())

    print("[FEATURES] using base point dataset")
    return dict(base_data), "base_point_dataset"


def correction_features(
    x: np.ndarray,
    raw_feature_names: list[str],
    lstm_prob: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list[str]]:
    idx = {name: i for i, name in enumerate(raw_feature_names)}

    def get_seq(name: str, default: float = 0.0) -> np.ndarray:
        if name in idx:
            return x[:, :, idx[name]].astype(np.float32)
        return np.full((len(x), x.shape[1]), default, dtype=np.float32)

    def get_repeated_or_calc(name: str, fallback: np.ndarray) -> np.ndarray:
        if name in idx:
            return x[:, -1, idx[name]].astype(np.float32)
        return fallback.astype(np.float32)

    ssta = get_seq("ssta")
    threshold_gap = get_seq("threshold_gap", np.nan)
    exceed90 = get_seq("exceed90")
    mhw = get_seq("mhw")

    recent_mhw = (mhw > 0.5).sum(axis=1).astype(np.float32)
    recent_exceed90 = (exceed90 > 0.5).sum(axis=1).astype(np.float32)
    latest_ssta = ssta[:, -1].astype(np.float32)
    latest_gap = threshold_gap[:, -1].astype(np.float32)
    ssta_trend = (ssta[:, -1] - ssta[:, 0]).astype(np.float32)
    gap_trend = (threshold_gap[:, -1] - threshold_gap[:, 0]).astype(np.float32)
    latest_exceed90 = exceed90[:, -1].astype(np.float32)
    latest_mhw = mhw[:, -1].astype(np.float32)

    feature_values = [
        lstm_prob.astype(np.float32),
        (lstm_prob - threshold).astype(np.float32),
        (lstm_prob >= threshold).astype(np.float32),
        get_repeated_or_calc("recent_mhw_days", recent_mhw),
        get_repeated_or_calc("recent_exceed90_days", recent_exceed90),
        get_repeated_or_calc("latest_ssta", latest_ssta),
        get_repeated_or_calc("latest_threshold_gap", latest_gap),
        get_repeated_or_calc("ssta_trend", ssta_trend),
        get_repeated_or_calc("threshold_gap_trend", gap_trend),
        latest_exceed90,
        latest_mhw,
        np.nanmean(ssta, axis=1).astype(np.float32),
        np.nanmean(threshold_gap, axis=1).astype(np.float32),
        np.nanmax(ssta, axis=1).astype(np.float32),
        np.nanmax(threshold_gap, axis=1).astype(np.float32),
        np.nanmin(threshold_gap, axis=1).astype(np.float32),
    ]
    names = [
        "lstm_prob",
        "lstm_margin",
        "base_pred",
        "recent_mhw_days",
        "recent_exceed90_days",
        "latest_ssta",
        "latest_threshold_gap",
        "ssta_trend",
        "threshold_gap_trend",
        "latest_exceed90",
        "latest_mhw",
        "mean_ssta",
        "mean_threshold_gap",
        "max_ssta",
        "max_threshold_gap",
        "min_threshold_gap",
    ]

    if "sin_doy" in idx:
        feature_values.append(x[:, -1, idx["sin_doy"]].astype(np.float32))
        names.append("sin_doy")
    if "cos_doy" in idx:
        feature_values.append(x[:, -1, idx["cos_doy"]].astype(np.float32))
        names.append("cos_doy")

    mat = np.stack(feature_values, axis=1)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return mat, names


def make_model_specs(sklearn_objs: dict[str, object], seed: int) -> list[tuple[str, object]]:
    LogisticRegression = sklearn_objs["LogisticRegression"]
    RandomForestClassifier = sklearn_objs["RandomForestClassifier"]
    HistGradientBoostingClassifier = sklearn_objs["HistGradientBoostingClassifier"]
    DecisionTreeClassifier = sklearn_objs["DecisionTreeClassifier"]

    specs: list[tuple[str, object]] = [
        (
            "logistic_balanced",
            LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs"),
        )
    ]
    for depth in (3, 5, 8):
        specs.append(
            (
                f"random_forest_depth_{depth}",
                RandomForestClassifier(
                    class_weight="balanced",
                    n_estimators=200,
                    max_depth=depth,
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        )
    for leaf_nodes in (8, 16, 31):
        specs.append(
            (
                f"hist_gradient_boosting_leaf_{leaf_nodes}",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    max_leaf_nodes=leaf_nodes,
                    random_state=seed,
                ),
            )
        )
    for depth in (2, 3, 4):
        specs.append(
            (
                f"decision_tree_depth_{depth}",
                DecisionTreeClassifier(
                    max_depth=depth,
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
        )
    return specs


def classifier_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            return np.zeros(len(x), dtype=np.float32)
        return proba[:, 1].astype(np.float32)
    if hasattr(model, "decision_function"):
        score = model.decision_function(x)
        return (1.0 / (1.0 + np.exp(-score))).astype(np.float32)
    return model.predict(x).astype(np.float32)


def corrected_metrics(
    y: np.ndarray,
    base_pred: np.ndarray,
    remove_scores: np.ndarray,
    add_scores: np.ndarray,
    remove_threshold: float,
    add_threshold: float,
) -> dict[str, float | int]:
    corrected = base_pred.astype(np.uint8).copy()
    remove_mask = (base_pred == 1) & (remove_scores >= remove_threshold)
    add_mask = (base_pred == 0) & (add_scores >= add_threshold)
    corrected[remove_mask] = 0
    corrected[add_mask] = 1
    metrics = binary_metrics(y, corrected)
    metrics["num_removed"] = int(remove_mask.sum())
    metrics["num_added"] = int(add_mask.sum())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual point correction classifiers.")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    args = parser.parse_args()

    cfg.ensure_dirs()
    sklearn_objs = require_sklearn()
    base = predict_original_point_lstm(batch_size=args.batch_size)
    threshold, base_sweep = best_threshold_from_val(base["y_val"], base["val_prob"])
    base_sweep.to_csv(cfg.POINT_RESIDUAL_CLASSIFIER_BASE_SWEEP_FILE, index=False)

    feature_data, feature_source = load_physics_dataset_or_base(base)
    raw_feature_names = [str(x) for x in feature_data["feature_names"].tolist()]
    x_train, feature_names = correction_features(feature_data["X_train"], raw_feature_names, base["train_prob"], threshold)
    x_val, feature_names_val = correction_features(feature_data["X_val"], raw_feature_names, base["val_prob"], threshold)
    x_test, feature_names_test = correction_features(feature_data["X_test"], raw_feature_names, base["test_prob"], threshold)
    if feature_names != feature_names_val or feature_names != feature_names_test:
        raise SystemExit("[ERROR] correction feature names differ across splits.")

    y_train = base["y_train"]
    y_val = base["y_val"]
    train_base_pred = (base["train_prob"] >= threshold).astype(np.uint8)
    val_base_pred = (base["val_prob"] >= threshold).astype(np.uint8)
    test_base_pred = (base["test_prob"] >= threshold).astype(np.uint8)
    train_base_metrics = binary_metrics(y_train, train_base_pred)
    val_base_metrics = binary_metrics(y_val, val_base_pred)
    test_base_metrics = binary_metrics(base["y_test"], test_base_pred)

    print(f"correction feature matrix shape train={x_train.shape} val={x_val.shape} test={x_test.shape}")
    print(f"feature_names: {feature_names}")
    print(f"train/val/test samples: {len(y_train)}/{len(y_val)}/{len(base['y_test'])}")
    print(
        "base positive ratio train/val/test: "
        f"{train_base_pred.mean():.6f}/{val_base_pred.mean():.6f}/{test_base_pred.mean():.6f}"
    )
    print(f"base F1 train/val/test: {train_base_metrics['f1']:.6f}/{val_base_metrics['f1']:.6f}/{test_base_metrics['f1']:.6f}")

    remove_train_idx = train_base_pred == 1
    add_train_idx = train_base_pred == 0
    remove_val_idx = val_base_pred == 1
    add_val_idx = val_base_pred == 0
    remove_target_train = (y_train[remove_train_idx] == 0).astype(np.uint8)
    add_target_train = (y_train[add_train_idx] == 1).astype(np.uint8)

    if len(np.unique(remove_target_train)) < 2:
        raise SystemExit("[ERROR] Remove target has a single class on train split.")
    if len(np.unique(add_target_train)) < 2:
        raise SystemExit("[ERROR] Add target has a single class on train split.")

    threshold_values = [float(round(x, 2)) for x in np.arange(0.50, 1.00, 0.05)]
    search_rows: list[dict[str, object]] = []
    trained_pairs = []

    for model_name, remove_model in make_model_specs(sklearn_objs, args.seed):
        add_model = make_model_specs(sklearn_objs, args.seed)[[name for name, _ in make_model_specs(sklearn_objs, args.seed)].index(model_name)][1]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                remove_model.fit(x_train[remove_train_idx], remove_target_train)
                add_model.fit(x_train[add_train_idx], add_target_train)
        except Exception as exc:
            print(f"[WARN] Skipping {model_name}: {exc}")
            continue

        remove_scores_val = np.zeros(len(y_val), dtype=np.float32)
        add_scores_val = np.zeros(len(y_val), dtype=np.float32)
        remove_scores_val[remove_val_idx] = classifier_scores(remove_model, x_val[remove_val_idx])
        add_scores_val[add_val_idx] = classifier_scores(add_model, x_val[add_val_idx])
        trained_pairs.append((model_name, remove_model, add_model, remove_scores_val, add_scores_val))

        for remove_threshold in threshold_values:
            for add_threshold in threshold_values:
                metrics = corrected_metrics(
                    y_val,
                    val_base_pred,
                    remove_scores_val,
                    add_scores_val,
                    remove_threshold,
                    add_threshold,
                )
                search_rows.append(
                    {
                        "model_name": model_name,
                        "remove_model_name": model_name,
                        "add_model_name": model_name,
                        "remove_threshold": remove_threshold,
                        "add_threshold": add_threshold,
                        "val_base_accuracy": val_base_metrics["accuracy"],
                        "val_base_precision": val_base_metrics["precision"],
                        "val_base_recall": val_base_metrics["recall"],
                        "val_base_f1": val_base_metrics["f1"],
                        "val_corrected_accuracy": metrics["accuracy"],
                        "val_corrected_precision": metrics["precision"],
                        "val_corrected_recall": metrics["recall"],
                        "val_corrected_f1": metrics["f1"],
                        "val_delta_accuracy": float(metrics["accuracy"]) - float(val_base_metrics["accuracy"]),
                        "val_delta_precision": float(metrics["precision"]) - float(val_base_metrics["precision"]),
                        "val_delta_recall": float(metrics["recall"]) - float(val_base_metrics["recall"]),
                        "val_delta_f1": float(metrics["f1"]) - float(val_base_metrics["f1"]),
                        "num_removed_val": metrics["num_removed"],
                        "num_added_val": metrics["num_added"],
                        "num_corrections_val": int(metrics["num_removed"]) + int(metrics["num_added"]),
                    }
                )

    if not search_rows:
        raise SystemExit("[ERROR] No residual classifier models were successfully trained.")

    search = pd.DataFrame(search_rows)
    search = search.sort_values(
        ["val_corrected_f1", "val_corrected_precision", "num_corrections_val"],
        ascending=[False, False, True],
    )
    search.to_csv(cfg.POINT_RESIDUAL_CLASSIFIER_VAL_SEARCH_FILE, index=False)
    best = search.iloc[0].to_dict()
    best_name = str(best["model_name"])
    selected_pair = next(pair for pair in trained_pairs if pair[0] == best_name)
    _, best_remove_model, best_add_model, _, _ = selected_pair

    sklearn_objs["dump"](best_remove_model, cfg.POINT_RESIDUAL_REMOVE_CLASSIFIER_FILE)
    sklearn_objs["dump"](best_add_model, cfg.POINT_RESIDUAL_ADD_CLASSIFIER_FILE)

    metadata = {
        "split_protocol": "residual classifiers were trained on train set, selected on validation set, and evaluated once on test set.",
        "base_model": "original_point_lstm",
        "base_model_file": str(cfg.POINT_LSTM_MODEL_FILE),
        "base_dataset_file": str(cfg.POINT_DATASET_FILE),
        "feature_source": feature_source,
        "physics_dataset_file": str(cfg.POINT_PHYSICS_DATASET_FILE),
        "best_val_threshold": float(threshold),
        "remove_model_name": str(best["remove_model_name"]),
        "add_model_name": str(best["add_model_name"]),
        "remove_threshold": float(best["remove_threshold"]),
        "add_threshold": float(best["add_threshold"]),
        "feature_names": feature_names,
        "train_base_metrics": {k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in train_base_metrics.items()},
        "val_base_metrics": {k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in val_base_metrics.items()},
        "test_base_metrics_cached_for_reference": {k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in test_base_metrics.items()},
        "best_val_search_row": {
            k: (int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in best.items()
        },
    }
    with cfg.POINT_RESIDUAL_CLASSIFIER_METADATA_JSON.open("w") as f:
        json.dump(metadata, f, indent=2)

    print("residual classifiers were trained on train set, selected on validation set, and evaluated once on test set.")
    print(f"[SAVED] {cfg.POINT_RESIDUAL_CLASSIFIER_BASE_SWEEP_FILE}")
    print(f"[SAVED] {cfg.POINT_RESIDUAL_CLASSIFIER_VAL_SEARCH_FILE}")
    print(f"[SAVED] {cfg.POINT_RESIDUAL_CLASSIFIER_METADATA_JSON}")
    print(f"[SAVED] {cfg.POINT_RESIDUAL_REMOVE_CLASSIFIER_FILE}")
    print(f"[SAVED] {cfg.POINT_RESIDUAL_ADD_CLASSIFIER_FILE}")
    print("[BEST VAL SETTING]")
    print(pd.DataFrame([best]).to_string(index=False))


if __name__ == "__main__":
    main()
