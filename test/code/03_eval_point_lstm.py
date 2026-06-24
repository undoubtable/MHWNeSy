#!/usr/bin/env python
"""Evaluate the point-wise LSTM on the test split."""

from __future__ import annotations

import argparse
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


cfg = SourceFileLoader("test_cfg", str(Path(__file__).with_name("00_test_config.py"))).load_module()


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[MISSING DEPENDENCY] torch is required to evaluate point_lstm.\n"
            "Install project dependencies first: pip install -r requirements.txt"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def try_auc_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Return ROC-AUC and PR-AUC, or nan if sklearn is unavailable."""

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ModuleNotFoundError:
        print("[WARN] scikit-learn not installed; ROC-AUC and PR-AUC set to nan.")
        return float("nan"), float("nan")

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, y_prob)), float(average_precision_score(y_true, y_prob))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = float(np.logical_and(y_true, y_pred).sum())
    fp = float(np.logical_and(~y_true, y_pred).sum())
    fn = float(np.logical_and(y_true, ~y_pred).sum())
    tn = float(np.logical_and(~y_true, ~y_pred).sum())

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate point-wise LSTM.")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg.ensure_dirs()
    if not cfg.POINT_DATASET_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.POINT_DATASET_FILE}")
    if not cfg.POINT_LSTM_MODEL_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.POINT_LSTM_MODEL_FILE}")

    torch, nn, DataLoader, TensorDataset = require_torch()
    data = np.load(cfg.POINT_DATASET_FILE, allow_pickle=True)
    x_test = data["X_test"].astype(np.float32)
    y_true = data["y_test"].astype(np.uint8)

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

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    probs = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    y_prob = np.concatenate(probs).astype(np.float32)
    y_pred = (y_prob >= args.threshold).astype(np.uint8)

    metrics = binary_metrics(y_true, y_pred)
    roc_auc, pr_auc = try_auc_metrics(y_true, y_prob)
    metrics["roc_auc"] = roc_auc
    metrics["pr_auc"] = pr_auc

    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    np.savez_compressed(
        cfg.POINT_LSTM_PRED_FILE,
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
    )
    print(f"[SAVED] {cfg.POINT_LSTM_PRED_FILE}")


if __name__ == "__main__":
    main()
