# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


class EventPatchDataset(Dataset):
    def __init__(self, npz_path, mean=None, std=None):
        z = np.load(npz_path, allow_pickle=True)
        self.X = z["X"]
        self.y = z["y_valid"].astype(np.float32)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)

        # 前 10 个通道是 SSTA，最后 1 个通道是预测候选 mask
        if self.mean is not None and self.std is not None:
            x[:10] = (x[:10] - self.mean) / (self.std + 1e-6)

        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class SmallEventCNN(nn.Module):
    def __init__(self, in_ch=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 16 -> 8

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.net(x)).squeeze(1)


def binary_metrics(y_true, prob, thr=0.5):
    y_true = y_true.astype(np.uint8)
    pred = (prob >= thr).astype(np.uint8)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "threshold": float(thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_pos_ratio": float(pred.mean()),
        "true_pos_ratio": float(y_true.mean()),
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in tqdm(loader, desc="Predict", leave=False):
        x = x.to(device, non_blocking=True)
        logit = model(x)
        prob = torch.sigmoid(logit).detach().cpu().numpy()
        probs.append(prob)
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def find_best_threshold(y_true, prob):
    best = None
    for thr in np.arange(0.05, 0.96, 0.05):
        m = binary_metrics(y_true, prob, thr=float(thr))
        if best is None or m["f1"] > best["f1"]:
            best = m
    return best


def compute_train_mean_std(train_npz):
    z = np.load(train_npz, allow_pickle=True)
    X = z["X"]
    vals = X[:, :10].astype(np.float32)
    mean = float(vals.mean())
    std = float(vals.std())
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    data_dir = cfg.EVENT_DIR
    out_dir = data_dir / "08_figure_event_verifier_cnn"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_npz = data_dir / "figure_event_train.npz"
    val_npz = data_dir / "figure_event_val.npz"
    test_npz = data_dir / "figure_event_test.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    mean, std = compute_train_mean_std(train_npz)
    print("[NORMALIZE] SSTA mean/std:", mean, std)

    train_ds = EventPatchDataset(train_npz, mean=mean, std=std)
    val_ds = EventPatchDataset(val_npz, mean=mean, std=std)
    test_ds = EventPatchDataset(test_npz, mean=mean, std=std)

    n_pos = float(train_ds.y.sum())
    n_neg = float(len(train_ds.y) - n_pos)
    pos_weight = n_neg / (n_pos + 1e-8)
    print("[TRAIN COUNTS] pos=", int(n_pos), "neg=", int(n_neg), "pos_weight=", pos_weight)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = SmallEventCNN(in_ch=11).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1 = -1.0
    best_thr = 0.5
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logit = model(x)
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        y_val, p_val = predict(model, val_loader, device)
        best_val_m = find_best_threshold(y_val, p_val)
        val_m_05 = binary_metrics(y_val, p_val, thr=0.5)

        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_f1_bestthr": best_val_m["f1"],
            "val_thr": best_val_m["threshold"],
            "val_precision_bestthr": best_val_m["precision"],
            "val_recall_bestthr": best_val_m["recall"],
            "val_f1_thr05": val_m_05["f1"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | loss={row['loss']:.4f} | "
            f"val_f1={row['val_f1_bestthr']:.4f} | "
            f"thr={row['val_thr']:.2f} | "
            f"P={row['val_precision_bestthr']:.4f} | "
            f"R={row['val_recall_bestthr']:.4f}"
        )

        if best_val_m["f1"] > best_f1:
            best_f1 = best_val_m["f1"]
            best_thr = best_val_m["threshold"]
            torch.save({
                "model": model.state_dict(),
                "mean": mean,
                "std": std,
                "best_thr": best_thr,
                "epoch": epoch,
                "best_val_metrics": best_val_m,
            }, out_dir / "best_model.pt")
            print("[SAVE BEST]", out_dir / "best_model.pt")

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)

    # reload best and evaluate all splits
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    best_thr = float(ckpt["best_thr"])
    print("[BEST THR]", best_thr)

    rows = []
    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        y_true, prob = predict(model, loader, device)
        pred = (prob >= best_thr).astype(np.uint8)

        np.save(out_dir / f"event_score_{split}.npy", prob.astype(np.float32))
        np.save(out_dir / f"event_pred_{split}.npy", pred.astype(np.uint8))

        m = binary_metrics(y_true, prob, thr=best_thr)
        m["split"] = split
        m["n_samples"] = int(len(y_true))
        rows.append(m)
        print("[METRICS]", split, m)

    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    print("[DONE]", out_dir)


if __name__ == "__main__":
    main()
