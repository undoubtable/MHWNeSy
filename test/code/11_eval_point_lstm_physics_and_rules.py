#!/usr/bin/env python
"""Evaluate physics-enhanced point LSTM and point-level rule corrections."""

from __future__ import annotations

import argparse
import json
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
            "[MISSING DEPENDENCY] torch is required to evaluate point_lstm_physics.\n"
            "Use the project environment, for example: conda run -n ybzcu121 python ..."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def try_auc_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ModuleNotFoundError:
        print("[WARN] scikit-learn not installed; ROC-AUC and PR-AUC set to nan.")
        return float("nan"), float("nan")

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob)), float(average_precision_score(y_true, y_prob))
    except ValueError:
        return float("nan"), float("nan")


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
        "support": int(y_pred.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    return binary_metrics(y_true, y_prob >= threshold)


def load_dataset() -> dict[str, np.ndarray]:
    if not cfg.POINT_PHYSICS_DATASET_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.POINT_PHYSICS_DATASET_FILE}")
    return dict(np.load(cfg.POINT_PHYSICS_DATASET_FILE, allow_pickle=True))


def feature_matrix(x: np.ndarray, feature_names: list[str]) -> dict[str, np.ndarray]:
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}
    required = [
        "recent_mhw_days",
        "recent_exceed90_days",
        "latest_threshold_gap",
        "threshold_gap_trend",
    ]
    missing = [name for name in required if name not in feature_to_idx]
    if missing:
        raise SystemExit(f"[ERROR] Missing required physics features: {missing}")

    # Window-level features are repeated at every time step by the dataset
    # builder, so the last time step is a convenient canonical copy.
    return {name: x[:, -1, feature_to_idx[name]].astype(np.float32) for name in required}


def build_rule_rows(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    base_pred: np.ndarray,
    base_f1: float,
    features: dict[str, np.ndarray],
    best_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_standalone(name: str, rule: str, pred: np.ndarray) -> None:
        metrics = binary_metrics(y_true, pred.astype(np.uint8))
        rows.append(
            {
                "rule_type": "standalone",
                "rule_name": name,
                "rule": rule,
                "removed_points": 0,
                "delta_f1": float(metrics["f1"]) - base_f1,
                **metrics,
            }
        )

    def add_deletion(name: str, rule: str, delete_mask: np.ndarray) -> None:
        corrected = base_pred.astype(bool) & ~delete_mask.astype(bool)
        metrics = binary_metrics(y_true, corrected.astype(np.uint8))
        rows.append(
            {
                "rule_type": "deletion_correction",
                "rule_name": name,
                "rule": rule,
                # For deletion rules, support is the number of removed predicted-positive points.
                "support": int(delete_mask.sum()),
                "removed_points": int(delete_mask.sum()),
                "delta_f1": float(metrics["f1"]) - base_f1,
                "accuracy": metrics["accuracy"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    recent_mhw = features["recent_mhw_days"]
    recent_exceed90 = features["recent_exceed90_days"]
    latest_gap = features["latest_threshold_gap"]
    gap_trend = features["threshold_gap_trend"]
    prob_rule = y_prob > best_threshold

    add_standalone("recent_mhw_days_ge_2", "recent_mhw_days >= 2", recent_mhw >= 2)
    add_standalone("recent_exceed90_days_ge_3", "recent_exceed90_days >= 3", recent_exceed90 >= 3)
    add_standalone("latest_threshold_gap_gt_0", "latest_threshold_gap > 0", latest_gap > 0)
    add_standalone("threshold_gap_trend_gt_0", "threshold_gap_trend > 0", gap_trend > 0)
    add_standalone("lstm_prob_gt_best_val_threshold", f"lstm_prob > {best_threshold:.2f}", prob_rule)
    add_standalone(
        "lstm_and_recent_mhw_days_ge_2",
        f"lstm_prob > {best_threshold:.2f} AND recent_mhw_days >= 2",
        prob_rule & (recent_mhw >= 2),
    )
    add_standalone(
        "lstm_and_recent_exceed90_days_ge_3",
        f"lstm_prob > {best_threshold:.2f} AND recent_exceed90_days >= 3",
        prob_rule & (recent_exceed90 >= 3),
    )
    add_standalone(
        "lstm_and_latest_threshold_gap_gt_0",
        f"lstm_prob > {best_threshold:.2f} AND latest_threshold_gap > 0",
        prob_rule & (latest_gap > 0),
    )

    base_positive = base_pred.astype(bool)
    add_deletion(
        "delete_recent_mhw_days_lt_1",
        "lstm_pred == 1 AND recent_mhw_days < 1",
        base_positive & (recent_mhw < 1),
    )
    add_deletion(
        "delete_recent_exceed90_days_lt_1",
        "lstm_pred == 1 AND recent_exceed90_days < 1",
        base_positive & (recent_exceed90 < 1),
    )
    add_deletion(
        "delete_latest_threshold_gap_lt_0",
        "lstm_pred == 1 AND latest_threshold_gap < 0",
        base_positive & (latest_gap < 0),
    )
    add_deletion(
        "delete_threshold_gap_trend_lt_0",
        "lstm_pred == 1 AND threshold_gap_trend < 0",
        base_positive & (gap_trend < 0),
    )

    columns = [
        "rule_type",
        "rule_name",
        "rule",
        "support",
        "removed_points",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "delta_f1",
    ]
    return pd.DataFrame(rows)[columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate physics point LSTM and rule corrections.")
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    cfg.ensure_dirs()
    if not cfg.POINT_LSTM_PHYSICS_MODEL.exists():
        raise SystemExit(f"[MISSING] {cfg.POINT_LSTM_PHYSICS_MODEL}")

    torch, nn, DataLoader, TensorDataset = require_torch()
    data = load_dataset()
    feature_names = [str(x) for x in data["feature_names"].tolist()]
    x_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.uint8)
    x_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.uint8)

    checkpoint = torch.load(cfg.POINT_LSTM_PHYSICS_MODEL, map_location="cpu")

    class PointLSTMPhysics(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            lstm_dropout = dropout if num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=lstm_dropout,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (hidden, _) = self.lstm(x)
            return self.head(hidden[-1]).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointLSTMPhysics(
        int(checkpoint["input_size"]),
        int(checkpoint["hidden_size"]),
        int(checkpoint["num_layers"]),
        float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def predict_prob(x: np.ndarray, split_name: str) -> np.ndarray:
        loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=args.batch_size, shuffle=False)
        probs = []
        with torch.no_grad():
            for batch_id, (xb,) in enumerate(loader, start=1):
                logits = model(xb.to(device))
                probs.append(torch.sigmoid(logits).cpu().numpy())
                if batch_id == 1 or batch_id == len(loader):
                    print(f"[PREDICT {split_name}] batch={batch_id}/{len(loader)}")
        return np.concatenate(probs).astype(np.float32)

    val_prob = predict_prob(x_val, "VAL")
    test_prob = predict_prob(x_test, "TEST")

    sweep_rows = []
    for threshold in np.arange(0.05, 1.00, 0.05):
        metrics = metrics_at_threshold(y_val, val_prob, float(threshold))
        sweep_rows.append({"threshold": float(round(threshold, 2)), **metrics})
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(cfg.POINT_LSTM_PHYSICS_VAL_SWEEP_FILE, index=False)
    best = sweep.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
    best_threshold = float(best["threshold"])

    test_pred = (test_prob >= best_threshold).astype(np.uint8)
    test_metrics = metrics_at_threshold(y_test, test_prob, best_threshold)
    roc_auc, pr_auc = try_auc_metrics(y_test, test_prob)
    test_metrics["roc_auc"] = roc_auc
    test_metrics["pr_auc"] = pr_auc

    features = feature_matrix(x_test, feature_names)
    rules = build_rule_rows(
        y_true=y_test,
        y_prob=test_prob,
        base_pred=test_pred,
        base_f1=float(test_metrics["f1"]),
        features=features,
        best_threshold=best_threshold,
    )
    rules.to_csv(cfg.POINT_LSTM_PHYSICS_RULE_FILE, index=False)

    np.savez_compressed(
        cfg.POINT_LSTM_PHYSICS_PRED_FILE,
        y_true=y_test,
        y_prob=test_prob,
        y_pred=test_pred,
        best_val_threshold=np.array(best_threshold, dtype=np.float32),
        feature_names=np.array(feature_names, dtype=object),
    )

    summary = {
        "best_val_threshold": best_threshold,
        "val_metrics_at_best_threshold": {
            key: float(value) for key, value in best.drop(labels=["threshold"]).to_dict().items()
        },
        "test_metrics_at_fixed_val_threshold": {
            key: (int(value) if isinstance(value, (np.integer,)) else float(value))
            for key, value in test_metrics.items()
        },
        "dataset": str(cfg.POINT_PHYSICS_DATASET_FILE),
        "model": str(cfg.POINT_LSTM_PHYSICS_MODEL),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "best_val_f1_from_training": float(checkpoint.get("best_val_f1", float("nan"))),
    }

    if cfg.POINT_THRESHOLD_SWEEP_FILE.exists():
        try:
            old = pd.read_csv(cfg.POINT_THRESHOLD_SWEEP_FILE)
            old_best = old.sort_values("f1", ascending=False).iloc[0].to_dict()
            summary["old_point_lstm_reference_from_existing_test_sweep"] = {
                key: (float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value)
                for key, value in old_best.items()
            }
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"[WARN] Could not read old point LSTM sweep: {exc}")

    with cfg.POINT_LSTM_PHYSICS_SUMMARY_JSON.open("w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(
        [
            {"split": "val", "threshold": best_threshold, **summary["val_metrics_at_best_threshold"]},
            {"split": "test", "threshold": best_threshold, **summary["test_metrics_at_fixed_val_threshold"]},
        ]
    ).to_csv(cfg.POINT_LSTM_PHYSICS_SUMMARY_CSV, index=False)

    print(f"[SAVED] {cfg.POINT_LSTM_PHYSICS_VAL_SWEEP_FILE}")
    print(f"[SAVED] {cfg.POINT_LSTM_PHYSICS_PRED_FILE}")
    print(f"[SAVED] {cfg.POINT_LSTM_PHYSICS_RULE_FILE}")
    print(f"[SAVED] {cfg.POINT_LSTM_PHYSICS_SUMMARY_JSON}")
    print(f"[SAVED] {cfg.POINT_LSTM_PHYSICS_SUMMARY_CSV}")
    print("[CALIBRATED LSTM TEST METRICS]")
    for key, value in test_metrics.items():
        print(f"{key}: {float(value):.6f}")
    print(f"best_val_threshold: {best_threshold:.2f}")

    print("[TOP POINT RULES BY F1]")
    print(rules.sort_values(["f1", "precision", "support"], ascending=False).head(10).to_string(index=False))
    print("[TOP CORRECTION RULES BY DELTA_F1]")
    print(
        rules[rules["rule_type"] == "deletion_correction"]
        .sort_values(["delta_f1", "precision", "support"], ascending=False)
        .head(10)
        .to_string(index=False)
    )
    if "old_point_lstm_reference_from_existing_test_sweep" in summary:
        print("[OLD POINT LSTM REFERENCE FROM EXISTING FILE; NOT USED FOR THRESHOLD SELECTION]")
        print(summary["old_point_lstm_reference_from_existing_test_sweep"])


if __name__ == "__main__":
    main()
