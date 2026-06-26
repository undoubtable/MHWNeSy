#!/usr/bin/env python
"""Train a physics-enhanced point-wise LSTM for MHW forecasting."""

from __future__ import annotations

import argparse
import logging
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
            "[MISSING DEPENDENCY] torch is required to train point_lstm_physics.\n"
            "Use the project environment, for example: conda run -n ybzcu121 python ..."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def setup_logger() -> logging.Logger:
    cfg.ensure_dirs()
    logger = logging.getLogger("point_lstm_physics_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(cfg.POINT_LSTM_PHYSICS_TRAIN_LOG, mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_dataset() -> dict[str, np.ndarray]:
    if not cfg.POINT_PHYSICS_DATASET_FILE.exists():
        raise SystemExit(
            f"[MISSING] {cfg.POINT_PHYSICS_DATASET_FILE}\n"
            "Build it first: python test/code/09_build_point_dataset_physics.py"
        )
    return dict(np.load(cfg.POINT_PHYSICS_DATASET_FILE, allow_pickle=True))


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
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1.0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train physics-enhanced point LSTM.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    torch, nn, DataLoader, TensorDataset = require_torch()
    logger = setup_logger()
    data = load_dataset()

    x_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    x_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)
    input_size = int(x_train.shape[-1])
    lstm_dropout = args.dropout if args.num_layers > 1 else 0.0

    class PointLSTMPhysics(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (hidden, _) = self.lstm(x)
            return self.head(hidden[-1]).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointLSTMPhysics(input_size, args.hidden_size, args.num_layers, lstm_dropout).to(device)

    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight_value = negatives / max(positives, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    logger.info("dataset=%s", cfg.POINT_PHYSICS_DATASET_FILE)
    logger.info("X_train=%s y_train=%s X_val=%s y_val=%s", x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    logger.info(
        "input_size=%d pos_weight=%.6f hidden_size=%d num_layers=%d dropout=%.3f device=%s",
        input_size,
        pos_weight_value,
        args.hidden_size,
        args.num_layers,
        lstm_dropout,
        device,
    )

    best_val_f1 = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            total_seen += len(yb)

        model.eval()
        val_loss_total = 0.0
        val_seen = 0
        val_probs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_total += float(loss.item()) * len(yb)
                val_seen += len(yb)
                val_probs.append(torch.sigmoid(logits).cpu().numpy())

        y_prob = np.concatenate(val_probs).astype(np.float32)
        y_pred = (y_prob >= 0.5).astype(np.uint8)
        metrics = binary_metrics(y_val, y_pred)
        train_loss = total_loss / max(total_seen, 1)
        val_loss = val_loss_total / max(val_seen, 1)
        logger.info(
            "epoch=%03d train_loss=%.6f val_loss=%.6f val_acc=%.6f "
            "val_precision=%.6f val_recall=%.6f val_f1=%.6f",
            epoch,
            train_loss,
            val_loss,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )

        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": input_size,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "dropout": lstm_dropout,
                    "history_days": cfg.HISTORY_DAYS,
                    "lead_days": cfg.LEAD_DAYS,
                    "feature_names": data["feature_names"].tolist(),
                    "pos_weight": pos_weight_value,
                    "best_epoch": best_epoch,
                    "best_val_f1": best_val_f1,
                },
                cfg.POINT_LSTM_PHYSICS_MODEL,
            )
            logger.info("saved_best_model=%s best_epoch=%d best_val_f1=%.6f", cfg.POINT_LSTM_PHYSICS_MODEL, best_epoch, best_val_f1)

    logger.info("done best_epoch=%d best_val_f1=%.6f log=%s", best_epoch, best_val_f1, cfg.POINT_LSTM_PHYSICS_TRAIN_LOG)


if __name__ == "__main__":
    main()
