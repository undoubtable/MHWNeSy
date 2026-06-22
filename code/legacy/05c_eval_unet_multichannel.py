# -*- coding: utf-8 -*-
"""
Evaluate multichannel U-Net and select a probability threshold.

Outputs:
    outputs/04c_unet_multichannel_h10_l5/
        pred_prob_train.npy, pred_prob_val.npy, pred_prob_test.npy
        eval_metrics.csv
        threshold_sweep.csv
        best_threshold_comparison.csv
        selected_threshold.json
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

train_mod = SourceFileLoader(
    "train_unet_multichannel",
    str(Path(__file__).with_name("04c_train_unet_multichannel.py")),
).load_module()
UNetSmall = train_mod.UNetSmall


METRIC_COLUMNS = [
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pixel_iou",
    "pixel_acc",
    "pred_pos_ratio",
    "true_pos_ratio",
]


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


def parse_splits(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_thresholds(args):
    if args.thresholds:
        return [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    n = int(round((args.threshold_max - args.threshold_min) / args.threshold_step)) + 1
    return [round(args.threshold_min + i * args.threshold_step, 10) for i in range(n)]


def counts_to_metrics(counts):
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    total = tp + fp + fn + tn

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    acc = (tp + tn) / (total + 1e-8)

    return {
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "pixel_f1": float(f1),
        "pixel_iou": float(iou),
        "pixel_acc": float(acc),
        "pred_pos_ratio": float((tp + fp) / (total + 1e-8)),
        "true_pos_ratio": float((tp + fn) / (total + 1e-8)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


@torch.no_grad()
def predict_split(model, data_dir, run_dir, split, device, batch_size, num_workers):
    ds = NpySegDataset(data_dir, split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n, H, W = ds.y.shape
    prob_path = run_dir / f"pred_prob_{split}.npy"
    prob_mm = np.lib.format.open_memmap(prob_path, mode="w+", dtype=np.float16, shape=(n, H, W))

    model.eval()
    start = 0
    for x, y in tqdm(loader, desc=f"Predict {split}"):
        bs = x.shape[0]
        x = x.to(device)
        prob = torch.sigmoid(model(x)).squeeze(1).cpu().numpy().astype(np.float32)
        prob_mm[start:start + bs] = prob.astype(np.float16)
        start += bs

    prob_mm.flush()
    print("[SAVE]", prob_path)


def evaluate_split_thresholds(split, data_dir, run_dir, thresholds, chunk_size):
    prob = np.load(run_dir / f"pred_prob_{split}.npy", mmap_mode="r")
    y = np.load(data_dir / f"y_{split}.npy", mmap_mode="r")

    if prob.shape != y.shape:
        raise ValueError(f"{split}: probability shape {prob.shape} != target shape {y.shape}")

    counts_by_thr = {
        thr: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for thr in thresholds
    }

    for start in tqdm(range(0, y.shape[0], chunk_size), desc=f"Sweep {split}"):
        end = min(start + chunk_size, y.shape[0])
        prob_chunk = np.array(prob[start:end], dtype=np.float32)
        y_chunk = np.array(y[start:end], dtype=np.uint8).astype(bool)

        for thr in thresholds:
            pred = prob_chunk >= thr
            counts = counts_by_thr[thr]
            counts["tp"] += int(np.logical_and(pred, y_chunk).sum())
            counts["fp"] += int(np.logical_and(pred, ~y_chunk).sum())
            counts["fn"] += int(np.logical_and(~pred, y_chunk).sum())
            counts["tn"] += int(np.logical_and(~pred, ~y_chunk).sum())

    rows = []
    for thr in thresholds:
        row = counts_to_metrics(counts_by_thr[thr])
        row.update({
            "split": split,
            "threshold": float(thr),
            "n_samples": int(y.shape[0]),
        })
        rows.append(row)

    return rows


def choose_best_threshold(rows, select_split, metric):
    sub = [r for r in rows if r["split"] == select_split]
    if not sub:
        raise ValueError(f"no rows found for select split: {select_split}")
    if metric not in sub[0]:
        raise ValueError(f"unknown metric: {metric}")

    return max(sub, key=lambda r: (float(r[metric]), -float(r["threshold"])))


def find_row(rows, split, threshold):
    for row in rows:
        if row["split"] == split and abs(float(row["threshold"]) - float(threshold)) < 1e-9:
            return row
    raise KeyError(f"row not found: split={split}, threshold={threshold}")


def build_comparison_rows(rows, splits, selected_threshold, default_threshold):
    comparison = []
    selected_metrics = []

    for split in splits:
        default_row = find_row(rows, split, default_threshold)
        selected_row = find_row(rows, split, selected_threshold)

        selected_metric = {
            "split": split,
            "threshold": float(selected_threshold),
            "n_samples": int(selected_row["n_samples"]),
        }
        for col in METRIC_COLUMNS:
            selected_metric[col] = selected_row[col]
        selected_metrics.append(selected_metric)

        base = {
            "split": split,
            "method": f"Multichannel U-Net threshold {default_threshold:.2f}",
            "threshold": float(default_threshold),
        }
        selected = {
            "split": split,
            "method": "Multichannel U-Net selected threshold",
            "threshold": float(selected_threshold),
        }
        delta = {
            "split": split,
            "method": "Delta selected - default",
            "threshold": float(selected_threshold),
        }

        for col in METRIC_COLUMNS:
            base[col] = default_row[col]
            selected[col] = selected_row[col]
            delta[col] = float(selected_row[col]) - float(default_row[col])

        comparison.extend([base, selected, delta])

    return selected_metrics, comparison


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "03b_forecast_dataset_multichannel_h10_l5"),
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5"),
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5" / "best_model.pt"),
    )
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--n_features", type=int, default=4)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--metric", type=str, default="pixel_f1")
    parser.add_argument("--select_split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--thresholds", type=str, default="")
    parser.add_argument("--default_threshold", type=float, default=0.50)
    parser.add_argument("--chunk_size", type=int, default=256)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    ckpt_path = Path(args.ckpt)
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = parse_splits(args.splits)
    thresholds = parse_thresholds(args)
    if args.default_threshold not in thresholds:
        thresholds = sorted(thresholds + [args.default_threshold])

    in_ch = args.history * args.n_features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)
    print("[DATA]", data_dir)
    print("[RUN]", run_dir)
    print("[CKPT]", ckpt_path)
    print("[THRESHOLDS]", thresholds)

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    model = UNetSmall(
        in_ch=int(config.get("in_ch", in_ch)),
        base=int(config.get("base", args.base)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    for split in splits:
        predict_split(
            model=model,
            data_dir=data_dir,
            run_dir=run_dir,
            split=split,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    rows = []
    for split in splits:
        rows.extend(
            evaluate_split_thresholds(
                split=split,
                data_dir=data_dir,
                run_dir=run_dir,
                thresholds=thresholds,
                chunk_size=args.chunk_size,
            )
        )

    threshold_sweep_csv = run_dir / "threshold_sweep.csv"
    write_csv(threshold_sweep_csv, rows)

    best_row = choose_best_threshold(rows, args.select_split, args.metric)
    selected_threshold = float(best_row["threshold"])
    eval_rows, comparison_rows = build_comparison_rows(
        rows=rows,
        splits=splits,
        selected_threshold=selected_threshold,
        default_threshold=args.default_threshold,
    )

    eval_csv = run_dir / "eval_metrics.csv"
    comparison_csv = run_dir / "best_threshold_comparison.csv"
    write_csv(eval_csv, eval_rows)
    write_csv(comparison_csv, comparison_rows)

    selected = {
        "selected_threshold": selected_threshold,
        "select_split": args.select_split,
        "select_metric": args.metric,
        "default_threshold": float(args.default_threshold),
        "thresholds": [float(t) for t in thresholds],
        "selected_val_row": best_row,
        "note": "Threshold is selected by maximizing the requested metric on the validation split.",
    }
    selected_json = run_dir / "selected_threshold.json"
    selected_json.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    print("[SELECTED THRESHOLD]", selected_threshold)
    print("[SAVE]", eval_csv)
    print("[SAVE]", threshold_sweep_csv)
    print("[SAVE]", comparison_csv)
    print("[SAVE]", selected_json)


if __name__ == "__main__":
    main()
