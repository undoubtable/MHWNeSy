# -*- coding: utf-8 -*-
"""
Evaluate U-Net baseline and save predictions for NeurRL event verifier.

Outputs:
    outputs/04_unet_baseline_h10_l5/
        eval_metrics.csv
        pred_prob_train.npy, pred_mask_train.npy
        pred_prob_val.npy,   pred_mask_val.npy
        pred_prob_test.npy,  pred_mask_test.npy
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()

# Reuse model definitions.
train_mod = SourceFileLoader("train_unet", str(Path(__file__).with_name("04_train_unet_baseline.py"))).load_module()
UNetSmall = train_mod.UNetSmall


class NpySegDataset(Dataset):
    def __init__(self, data_dir: Path, split: str):
        self.X = np.load(data_dir / f"X_{split}.npy", mmap_mode="r")
        self.y = np.load(data_dir / f"y_{split}.npy", mmap_mode="r")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        y = torch.from_numpy(np.array(self.y[idx], dtype=np.uint8))
        return x, y


def compute_metrics(pred_mask, y):
    pred = pred_mask.astype(bool)
    yy = y.astype(bool)

    tp = np.logical_and(pred, yy).sum()
    fp = np.logical_and(pred, ~yy).sum()
    fn = np.logical_and(~pred, yy).sum()
    tn = np.logical_and(~pred, ~yy).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "pixel_f1": float(f1),
        "pixel_iou": float(iou),
        "pixel_acc": float(acc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


@torch.no_grad()
def predict_split(model, data_dir, run_dir, split, device, batch_size, num_workers, threshold):
    ds = NpySegDataset(data_dir, split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n, H, W = ds.y.shape
    prob_path = run_dir / f"pred_prob_{split}.npy"
    mask_path = run_dir / f"pred_mask_{split}.npy"

    prob_mm = np.lib.format.open_memmap(prob_path, mode="w+", dtype=np.float16, shape=(n, H, W))
    mask_mm = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.uint8, shape=(n, H, W))

    model.eval()
    start = 0

    for x, y in tqdm(loader, desc=f"Predict {split}"):
        bs = x.shape[0]
        x = x.to(device)
        prob = torch.sigmoid(model(x)).squeeze(1).cpu().numpy().astype(np.float32)
        mask = (prob >= threshold).astype(np.uint8)

        prob_mm[start:start + bs] = prob.astype(np.float16)
        mask_mm[start:start + bs] = mask
        start += bs

    prob_mm.flush()
    mask_mm.flush()

    y_all = np.load(data_dir / f"y_{split}.npy", mmap_mode="r")
    pred_all = np.load(mask_path, mmap_mode="r")
    metrics = compute_metrics(pred_all, y_all)
    metrics["split"] = split
    metrics["n_samples"] = int(n)

    print("[METRICS]", split, metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(cfg.FORECAST_DIR))
    parser.add_argument("--run_dir", type=str, default=str(cfg.UNET_RUN_DIR))
    parser.add_argument("--ckpt", type=str, default=str(cfg.UNET_BEST))
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--splits", type=str, default="train,val,test")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    ckpt_path = Path(args.ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = UNetSmall(in_ch=args.history, base=args.base).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    rows = []
    for split in args.splits.split(","):
        split = split.strip()
        if not split:
            continue
        rows.append(
            predict_split(
                model=model,
                data_dir=data_dir,
                run_dir=run_dir,
                split=split,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                threshold=args.threshold,
            )
        )

    out_csv = run_dir / "eval_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[SAVE]", out_csv)


if __name__ == "__main__":
    main()
