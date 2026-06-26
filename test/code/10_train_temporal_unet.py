#!/usr/bin/env python
"""Train a lightweight Temporal U-Net full-grid MHW forecasting baseline."""

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
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[MISSING DEPENDENCY] torch is required to train Temporal U-Net.\n"
            "Use the project environment, for example: conda run -n ybzcu121 python ..."
        ) from exc
    return torch, nn, F, DataLoader, Dataset


def setup_logger() -> logging.Logger:
    cfg.ensure_dirs()
    logger = logging.getLogger("temporal_unet_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(cfg.TEMPORAL_UNET_TRAIN_LOG, mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def sidecar_files_exist() -> bool:
    return all(
        path.exists()
        for path in (
            cfg.TEMPORAL_UNET_X_TRAIN_FILE,
            cfg.TEMPORAL_UNET_Y_TRAIN_FILE,
            cfg.TEMPORAL_UNET_X_VAL_FILE,
            cfg.TEMPORAL_UNET_Y_VAL_FILE,
        )
    )


def load_dataset() -> dict[str, np.ndarray]:
    if not cfg.TEMPORAL_UNET_DATASET_FILE.exists():
        raise SystemExit(
            f"[MISSING] {cfg.TEMPORAL_UNET_DATASET_FILE}\n"
            "Build it first: python test/code/09_build_temporal_unet_dataset.py"
        )
    meta = np.load(cfg.TEMPORAL_UNET_DATASET_FILE, allow_pickle=True)
    if sidecar_files_exist():
        print("[LOAD] using mmap sidecar .npy arrays for Temporal U-Net training")
        return {
            "X_train": np.load(cfg.TEMPORAL_UNET_X_TRAIN_FILE, mmap_mode="r"),
            "y_train": np.load(cfg.TEMPORAL_UNET_Y_TRAIN_FILE, mmap_mode="r"),
            "X_val": np.load(cfg.TEMPORAL_UNET_X_VAL_FILE, mmap_mode="r"),
            "y_val": np.load(cfg.TEMPORAL_UNET_Y_VAL_FILE, mmap_mode="r"),
            "ocean_mask": meta["ocean_mask"],
            "feature_names": meta["feature_names"] if "feature_names" in meta.files else np.array([]),
        }
    print("[LOAD] sidecar .npy arrays not found; loading arrays from .npz")
    return {
        "X_train": meta["X_train"],
        "y_train": meta["y_train"],
        "X_val": meta["X_val"],
        "y_val": meta["y_val"],
        "ocean_mask": meta["ocean_mask"],
        "feature_names": meta["feature_names"] if "feature_names" in meta.files else np.array([]),
    }


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = float(np.logical_and(y_true, y_pred).sum())
    fp = float(np.logical_and(~y_true, y_pred).sum())
    fn = float(np.logical_and(y_true, ~y_pred).sum())
    tn = float(np.logical_and(~y_true, ~y_pred).sum())
    return tp, fp, fn, tn


def metrics_from_counts(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1.0)
    iou = tp / max(tp + fp + fn, 1.0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_csi": iou,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Temporal U-Net MHW baseline.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dice_weight", type=float, default=0.0)
    parser.add_argument("--log_every_batches", type=int, default=100)
    args = parser.parse_args()

    torch, nn, F, DataLoader, Dataset = require_torch()
    logger = setup_logger()
    data = load_dataset()

    x_train = data["X_train"]
    y_train = data["y_train"].astype(np.float32)
    x_val = data["X_val"]
    y_val = data["y_val"].astype(np.float32)
    ocean_mask_np = data["ocean_mask"].astype(bool)
    in_channels = int(x_train.shape[1])

    class GridDataset(Dataset):
        """Lazy numpy-backed dataset; works with normal arrays and memmaps."""

        def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
            self.x = x
            self.y = y

        def __len__(self) -> int:
            return int(self.y.shape[0])

        def __getitem__(self, idx: int):
            # asarray materializes only the requested sample when x/y are memmaps.
            xb = torch.from_numpy(np.asarray(self.x[idx]))
            yb = torch.from_numpy(np.asarray(self.y[idx], dtype=np.float32))
            return xb, yb

    class ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class TemporalUNet(nn.Module):
        def __init__(self, in_channels: int) -> None:
            super().__init__()
            self.enc1 = ConvBlock(in_channels, 32)
            self.enc2 = ConvBlock(32, 64)
            self.enc3 = ConvBlock(64, 128)
            self.bottleneck = ConvBlock(128, 256)
            self.dec3 = ConvBlock(256 + 128, 128)
            self.dec2 = ConvBlock(128 + 64, 64)
            self.dec1 = ConvBlock(64 + 32, 32)
            self.out = nn.Conv2d(32, 1, kernel_size=1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(F.max_pool2d(e1, 2))
            e3 = self.enc3(F.max_pool2d(e2, 2))
            b = self.bottleneck(F.max_pool2d(e3, 2))
            d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            return self.out(d1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalUNet(in_channels).to(device)
    ocean_mask = torch.from_numpy(ocean_mask_np.astype(np.float32)).to(device)

    y_train_ocean = y_train[:, ocean_mask_np]
    positives = float(y_train_ocean.sum())
    total = float(y_train_ocean.size)
    negatives = total - positives
    pos_weight_value = negatives / max(positives, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        GridDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        GridDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
    )

    def masked_loss(logits, target):
        logits_2d = logits.squeeze(1)
        loss_map = criterion(logits_2d, target)
        mask = ocean_mask.unsqueeze(0).expand_as(target)
        bce = (loss_map * mask).sum() / mask.sum().clamp_min(1.0)
        if args.dice_weight <= 0:
            return bce
        prob = torch.sigmoid(logits_2d)
        intersection = (prob * target * mask).sum()
        denom = (prob * mask).sum() + (target * mask).sum()
        dice = 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)
        return bce + args.dice_weight * dice

    def evaluate():
        model.eval()
        total_loss = 0.0
        total_seen = 0
        tp = fp = fn = tn = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.float32)
                logits = model(xb)
                loss = masked_loss(logits, yb)
                total_loss += float(loss.item()) * len(yb)
                total_seen += len(yb)
                pred = (torch.sigmoid(logits.squeeze(1)) >= 0.5).detach().cpu().numpy()
                true = yb.detach().cpu().numpy().astype(bool)
                mask = ocean_mask_np[None, :, :]
                ctp, cfp, cfn, ctn = binary_counts(true[mask.repeat(len(true), axis=0)], pred[mask.repeat(len(pred), axis=0)])
                tp += ctp
                fp += cfp
                fn += cfn
                tn += ctn
        metrics = metrics_from_counts(tp, fp, fn, tn)
        return total_loss / max(total_seen, 1), metrics

    logger.info("dataset=%s", cfg.TEMPORAL_UNET_DATASET_FILE)
    logger.info(
        "X_train=%s y_train=%s X_val=%s y_val=%s ocean_pixels=%d",
        x_train.shape,
        y_train.shape,
        x_val.shape,
        y_val.shape,
        int(ocean_mask_np.sum()),
    )
    logger.info("in_channels=%d pos_weight=%.6f device=%s dice_weight=%.3f", in_channels, pos_weight_value, device, args.dice_weight)

    best_val_f1 = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for batch_id, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = masked_loss(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            total_seen += len(yb)
            if args.log_every_batches > 0 and (
                batch_id == 1 or batch_id % args.log_every_batches == 0 or batch_id == len(train_loader)
            ):
                logger.info(
                    "epoch=%03d batch=%04d/%04d train_loss_running=%.6f",
                    epoch,
                    batch_id,
                    len(train_loader),
                    total_loss / max(total_seen, 1),
                )

        train_loss = total_loss / max(total_seen, 1)
        val_loss, val_metrics = evaluate()
        logger.info(
            "epoch=%03d train_loss=%.6f val_loss=%.6f val_accuracy=%.6f "
            "val_precision=%.6f val_recall=%.6f val_f1=%.6f val_iou_csi=%.6f",
            epoch,
            train_loss,
            val_loss,
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
            val_metrics["iou_csi"],
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "history_days": cfg.HISTORY_DAYS,
                    "lead_days": cfg.LEAD_DAYS,
                    "feature_names": data.get("feature_names", np.array([])).tolist(),
                    "pos_weight": pos_weight_value,
                    "best_epoch": best_epoch,
                    "best_val_f1": best_val_f1,
                },
                cfg.TEMPORAL_UNET_MODEL_FILE,
            )
            logger.info("saved_best_model=%s best_epoch=%d best_val_f1=%.6f", cfg.TEMPORAL_UNET_MODEL_FILE, best_epoch, best_val_f1)

    logger.info("done best_epoch=%d best_val_f1=%.6f log=%s", best_epoch, best_val_f1, cfg.TEMPORAL_UNET_TRAIN_LOG)


if __name__ == "__main__":
    main()
