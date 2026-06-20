# -*- coding: utf-8 -*-
"""
Train a simple U-Net baseline for MHW mask forecasting.

Input:
    /ybz/ybz/2026/MHWNeurRL/data/forecast_h10_l5/

Output:
    /ybz/ybz/2026/MHWNeurRL/runs/04_unet_baseline_h10_l5/
        best_model.pt
        train_log.csv
        config.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


class NpySegDataset(Dataset):
    def __init__(self, data_dir: Path, split: str):
        self.X = np.load(data_dir / f"X_{split}.npy", mmap_mode="r")
        self.y = np.load(data_dir / f"y_{split}.npy", mmap_mode="r")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        y = torch.from_numpy(np.array(self.y[idx], dtype=np.float32))[None, ...]
        return x, y


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=10, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.dec3 = DoubleConv(base * 8 + base * 4, base * 4)
        self.dec2 = DoubleConv(base * 4 + base * 2, base * 2)
        self.dec1 = DoubleConv(base * 2 + base, base)

        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))

        x = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, e3], dim=1))

        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, e2], dim=1))

        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, e1], dim=1))

        return self.out(x)


def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dims)
    denom = probs.sum(dims) + targets.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    tp = fp = fn = tn = 0

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = (torch.sigmoid(logits) >= threshold).bool()
        yy = y.bool()

        tp += (pred & yy).sum().item()
        fp += (pred & ~yy).sum().item()
        fn += (~pred & yy).sum().item()
        tn += (~pred & ~yy).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_iou": iou,
        "pixel_acc": acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(cfg.FORECAST_DIR))
    parser.add_argument("--run_dir", type=str, default=str(cfg.UNET_RUN_DIR))
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    train_ds = NpySegDataset(data_dir, "train")
    val_ds = NpySegDataset(data_dir, "val")

    y_train = np.load(data_dir / "y_train.npy", mmap_mode="r")
    pos = float(y_train.sum())
    total = float(np.prod(y_train.shape))
    neg = total - pos
    pos_weight_value = min(50.0, max(1.0, neg / (pos + 1e-8)))
    print("[POS WEIGHT]", pos_weight_value)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = UNetSmall(in_ch=args.history, base=args.base).to(device)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    config = vars(args)
    config["device"] = str(device)
    config["pos_weight"] = pos_weight_value
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    log_path = run_dir / "train_log.csv"
    best_iou = -1.0

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "train_loss",
                "val_pixel_precision", "val_pixel_recall",
                "val_pixel_f1", "val_pixel_iou", "val_pixel_acc",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []

            for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                loss = bce(logits, y) + dice_loss_from_logits(logits, y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                losses.append(loss.item())

            train_loss = float(np.mean(losses))
            metrics = evaluate(model, val_loader, device, threshold=args.threshold)

            row = {"epoch": epoch, "train_loss": train_loss}
            row.update({f"val_{k}": v for k, v in metrics.items()})
            writer.writerow(row)
            f.flush()

            print(
                f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                f"val_iou={metrics['pixel_iou']:.4f} | "
                f"val_f1={metrics['pixel_f1']:.4f}"
            )

            if metrics["pixel_iou"] > best_iou:
                best_iou = metrics["pixel_iou"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "best_val_iou": best_iou,
                    },
                    run_dir / "best_model.pt",
                )
                print("[SAVE BEST]", run_dir / "best_model.pt")

    print("[DONE]", run_dir)


if __name__ == "__main__":
    main()
