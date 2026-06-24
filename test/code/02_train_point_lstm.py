#!/usr/bin/env python
"""Train a small point-wise LSTM baseline for future MHW occurrence.

This is a scaffold model for the hierarchical experiment. It is deliberately
simple: one LSTM encoder followed by a linear binary classifier.
"""

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
            "[MISSING DEPENDENCY] torch is required to train point_lstm.\n"
            "Install project dependencies first: pip install -r requirements.txt"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def setup_logger() -> logging.Logger:
    cfg.ensure_dirs()
    logger = logging.getLogger("point_lstm_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(cfg.POINT_LSTM_TRAIN_LOG, mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_dataset() -> dict[str, np.ndarray]:
    if not cfg.POINT_DATASET_FILE.exists():
        raise SystemExit(
            f"[MISSING] {cfg.POINT_DATASET_FILE}\n"
            "Build it first: python test/code/01_build_point_dataset.py"
        )
    return dict(np.load(cfg.POINT_DATASET_FILE, allow_pickle=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train point-wise LSTM MHW baseline.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch, nn, DataLoader, TensorDataset = require_torch()
    logger = setup_logger()
    data = load_dataset()

    x_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    x_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)
    input_size = int(x_train.shape[-1])

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
    model = PointLSTM(input_size, args.hidden_size, args.num_layers).to(device)

    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight_value = negatives / max(positives, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    logger.info("dataset=%s", cfg.POINT_DATASET_FILE)
    logger.info("X_train=%s y_train=%s X_val=%s y_val=%s", x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    logger.info("input_size=%d pos_weight=%.6f device=%s", input_size, pos_weight_value, device)

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
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = criterion(val_logits, val_y)
            val_pred = (torch.sigmoid(val_logits) >= 0.5).float()
            val_acc = (val_pred == val_y).float().mean()

        logger.info(
            "epoch=%03d train_loss=%.6f val_loss=%.6f val_acc=%.6f",
            epoch,
            total_loss / max(total_seen, 1),
            float(val_loss.item()),
            float(val_acc.item()),
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "history_days": cfg.HISTORY_DAYS,
            "lead_days": cfg.LEAD_DAYS,
            "feature_names": data.get("feature_names", np.array(cfg.INPUT_VARIABLES)).tolist(),
        },
        cfg.POINT_LSTM_MODEL_FILE,
    )
    logger.info("saved_model=%s", cfg.POINT_LSTM_MODEL_FILE)
    logger.info("saved_log=%s", cfg.POINT_LSTM_TRAIN_LOG)


if __name__ == "__main__":
    main()
