#!/usr/bin/env python
"""Evaluate Temporal U-Net with validation-selected threshold."""

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
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[MISSING DEPENDENCY] torch is required to evaluate Temporal U-Net.\n"
            "Use the project environment, for example: conda run -n ybzcu121 python ..."
        ) from exc
    return torch, nn, F, DataLoader, Dataset


def try_auc_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Return ROC-AUC and PR-AUC, with NaN for unavailable or single-class cases."""

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


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_prob >= threshold
    tp = float(np.logical_and(y_true_bool, y_pred_bool).sum())
    fp = float(np.logical_and(~y_true_bool, y_pred_bool).sum())
    fn = float(np.logical_and(y_true_bool, ~y_pred_bool).sum())
    tn = float(np.logical_and(~y_true_bool, ~y_pred_bool).sum())

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
    parser = argparse.ArgumentParser(description="Evaluate Temporal U-Net MHW baseline.")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    cfg.ensure_dirs()
    if not cfg.TEMPORAL_UNET_DATASET_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.TEMPORAL_UNET_DATASET_FILE}")
    if not cfg.TEMPORAL_UNET_MODEL_FILE.exists():
        raise SystemExit(f"[MISSING] {cfg.TEMPORAL_UNET_MODEL_FILE}")

    torch, nn, F, DataLoader, Dataset = require_torch()

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

    data = np.load(cfg.TEMPORAL_UNET_DATASET_FILE, allow_pickle=True)
    if all(
        path.exists()
        for path in (
            cfg.TEMPORAL_UNET_X_VAL_FILE,
            cfg.TEMPORAL_UNET_Y_VAL_FILE,
            cfg.TEMPORAL_UNET_X_TEST_FILE,
            cfg.TEMPORAL_UNET_Y_TEST_FILE,
        )
    ):
        print("[LOAD] using mmap sidecar .npy arrays for Temporal U-Net evaluation")
        x_val = np.load(cfg.TEMPORAL_UNET_X_VAL_FILE, mmap_mode="r")
        y_val = np.load(cfg.TEMPORAL_UNET_Y_VAL_FILE, mmap_mode="r").astype(np.uint8)
        x_test = np.load(cfg.TEMPORAL_UNET_X_TEST_FILE, mmap_mode="r")
        y_test = np.load(cfg.TEMPORAL_UNET_Y_TEST_FILE, mmap_mode="r").astype(np.uint8)
    else:
        print("[LOAD] sidecar .npy arrays not found; loading arrays from .npz")
        x_val = data["X_val"]
        y_val = data["y_val"].astype(np.uint8)
        x_test = data["X_test"]
        y_test = data["y_test"].astype(np.uint8)
    ocean_mask = data["ocean_mask"].astype(bool)

    class GridOnlyDataset(Dataset):
        """Lazy numpy-backed dataset for prediction."""

        def __init__(self, x: np.ndarray) -> None:
            self.x = x

        def __len__(self) -> int:
            return int(self.x.shape[0])

        def __getitem__(self, idx: int):
            return torch.from_numpy(np.asarray(self.x[idx]))

    checkpoint = torch.load(cfg.TEMPORAL_UNET_MODEL_FILE, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalUNet(int(checkpoint["in_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def predict_probs(x: np.ndarray, split_name: str) -> np.ndarray:
        loader = DataLoader(GridOnlyDataset(x), batch_size=args.batch_size, shuffle=False)
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_id, xb in enumerate(loader, start=1):
                logits = model(xb.to(device=device, dtype=torch.float32))
                probs.append(torch.sigmoid(logits.squeeze(1)).cpu().numpy().astype(np.float32))
                if batch_id == 1 or batch_id % 50 == 0 or batch_id == len(loader):
                    print(f"[PREDICT {split_name}] batch={batch_id}/{len(loader)}")
        return np.concatenate(probs, axis=0)

    print(f"[LOAD] dataset={cfg.TEMPORAL_UNET_DATASET_FILE}")
    print(f"[LOAD] model={cfg.TEMPORAL_UNET_MODEL_FILE}")
    print(f"[SHAPE] X_val={x_val.shape} y_val={y_val.shape} X_test={x_test.shape} y_test={y_test.shape}")

    val_prob = predict_probs(x_val, "VAL")
    val_true = y_val[:, ocean_mask].reshape(-1)
    val_score = val_prob[:, ocean_mask].reshape(-1)

    rows = []
    for threshold in np.arange(0.05, 1.00, 0.05):
        metrics = binary_metrics(val_true, val_score, float(threshold))
        rows.append({"threshold": float(round(threshold, 2)), **metrics})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(cfg.TEMPORAL_UNET_VAL_SWEEP_FILE, index=False)
    best = sweep.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
    best_threshold = float(best["threshold"])

    test_prob = predict_probs(x_test, "TEST")
    test_pred = (test_prob >= best_threshold).astype(np.uint8)
    test_true = y_test[:, ocean_mask].reshape(-1)
    test_score = test_prob[:, ocean_mask].reshape(-1)
    test_metrics = binary_metrics(test_true, test_score, best_threshold)
    roc_auc, pr_auc = try_auc_metrics(test_true, test_score)
    test_metrics["roc_auc"] = roc_auc
    test_metrics["pr_auc"] = pr_auc

    val_metrics = best.drop(labels=["threshold"]).to_dict()
    summary = {
        "best_val_threshold": best_threshold,
        "val_metrics_at_best_threshold": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics_at_fixed_val_threshold": {k: float(v) for k, v in test_metrics.items()},
        "dataset": str(cfg.TEMPORAL_UNET_DATASET_FILE),
        "model": str(cfg.TEMPORAL_UNET_MODEL_FILE),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "best_val_f1_from_training": float(checkpoint.get("best_val_f1", float("nan"))),
    }

    point_lstm_reference = None
    if cfg.POINT_THRESHOLD_SWEEP_FILE.exists():
        try:
            point_sweep = pd.read_csv(cfg.POINT_THRESHOLD_SWEEP_FILE)
            if "f1" in point_sweep.columns:
                point_best = point_sweep.sort_values("f1", ascending=False).iloc[0].to_dict()
                point_lstm_reference = {
                    key: (float(value) if isinstance(value, (int, float, np.floating, np.integer)) else value)
                    for key, value in point_best.items()
                }
                summary["point_lstm_reference_from_existing_sweep"] = point_lstm_reference
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"[WARN] Could not read point LSTM sweep reference: {exc}")

    np.save(cfg.TEMPORAL_UNET_TEST_PROB_FILE, test_prob.astype(np.float32))
    np.save(cfg.TEMPORAL_UNET_TEST_PRED_FILE, test_pred)
    with cfg.TEMPORAL_UNET_EVAL_SUMMARY_JSON.open("w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(
        [
            {"split": "val", "threshold": best_threshold, **summary["val_metrics_at_best_threshold"]},
            {"split": "test", "threshold": best_threshold, **summary["test_metrics_at_fixed_val_threshold"]},
        ]
    ).to_csv(cfg.TEMPORAL_UNET_EVAL_SUMMARY_CSV, index=False)

    print(f"[SAVED] {cfg.TEMPORAL_UNET_VAL_SWEEP_FILE}")
    print(f"[SAVED] {cfg.TEMPORAL_UNET_TEST_PROB_FILE}")
    print(f"[SAVED] {cfg.TEMPORAL_UNET_TEST_PRED_FILE}")
    print(f"[SAVED] {cfg.TEMPORAL_UNET_EVAL_SUMMARY_JSON}")
    print(f"[SAVED] {cfg.TEMPORAL_UNET_EVAL_SUMMARY_CSV}")
    print(f"best_val_threshold: {best_threshold:.2f}")
    print("val metrics at best threshold:")
    for key, value in summary["val_metrics_at_best_threshold"].items():
        print(f"  {key}: {value:.6f}")
    print("test metrics at fixed val threshold:")
    for key, value in summary["test_metrics_at_fixed_val_threshold"].items():
        print(f"  {key}: {value:.6f}")
    if point_lstm_reference is not None:
        print("point LSTM reference from existing point_threshold_sweep.csv; not a strict identical-protocol comparison:")
        print(point_lstm_reference)


if __name__ == "__main__":
    main()
