# -*- coding: utf-8 -*-
"""
Apply event-level verifier correction to multichannel U-Net predictions.

Outputs:
    outputs/04c_unet_multichannel_h10_l5/09c_unet_plus_figure_verifier/
        corrected_mask_{split}_thrXX.npy
        correction_metrics.csv
        correction_summary.json
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def parse_thresholds(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def thr_tag(thr):
    return f"{thr:.2f}".replace(".", "p")


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
        "pred_pos_ratio": float(pred.mean()),
        "true_pos_ratio": float(yy.mean()),
    }


def build_remove_dict(meta, scores, threshold):
    remove = defaultdict(list)
    keep_count = 0
    remove_count = 0

    for k in range(len(scores)):
        sample_index = int(meta[k, 0])
        component_id = int(meta[k, 1])
        score = float(scores[k])

        if score < threshold:
            remove[sample_index].append(component_id)
            remove_count += 1
        else:
            keep_count += 1

    return remove, keep_count, remove_count


def make_pred_mask_from_prob(prob_path, mask_path, selected_threshold, chunk_size):
    prob = np.load(prob_path, mmap_mode="r")
    N, H, W = prob.shape
    mask = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.uint8, shape=(N, H, W))
    for start in tqdm(range(0, N, chunk_size), desc=f"Build mask {prob_path.name}"):
        end = min(start + chunk_size, N)
        mask[start:end] = (np.array(prob[start:end], dtype=np.float32) >= selected_threshold).astype(np.uint8)
    mask.flush()


def load_selected_threshold(path):
    selected = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(selected["selected_threshold"])


def correct_split(
    split, threshold, data_dir, pred_dir, event_dir, verifier_dir, out_dir,
    selected_threshold, mask_chunk_size, save_mask=True,
):
    y_path = data_dir / f"y_{split}.npy"
    pred_path = pred_dir / f"pred_mask_{split}.npy"
    prob_path = pred_dir / f"pred_prob_{split}.npy"
    event_path = event_dir / f"figure_event_{split}.npz"
    score_path = verifier_dir / f"event_score_{split}.npy"

    if not pred_path.exists():
        make_pred_mask_from_prob(prob_path, pred_path, selected_threshold, mask_chunk_size)

    y = np.load(y_path, mmap_mode="r")
    pred = np.load(pred_path, mmap_mode="r")
    z = np.load(event_path, allow_pickle=True)
    meta = z["meta"]
    scores = np.load(score_path)

    assert len(meta) == len(scores), f"{split}: meta/scores length mismatch"

    N, H, W = pred.shape
    remove_dict, keep_count, remove_count = build_remove_dict(meta, scores, threshold)

    if save_mask:
        out_mask_path = out_dir / f"corrected_mask_{split}_thr{thr_tag(threshold)}.npy"
        corrected = np.lib.format.open_memmap(
            out_mask_path, mode="w+", dtype=np.uint8, shape=(N, H, W)
        )
    else:
        corrected = np.empty((N, H, W), dtype=np.uint8)
        out_mask_path = None

    for i in tqdm(range(N), desc=f"Correct {split} thr={threshold:.2f}"):
        m = np.array(pred[i], dtype=np.uint8)

        if i in remove_dict:
            lab, _ = ndimage.label(m.astype(bool))
            for cid in remove_dict[i]:
                m[lab == cid] = 0

        corrected[i] = m

    if save_mask:
        corrected.flush()
        corrected_for_eval = np.load(out_mask_path, mmap_mode="r")
    else:
        corrected_for_eval = corrected

    metrics = compute_metrics(corrected_for_eval, y)
    metrics.update({
        "split": split,
        "threshold": float(threshold),
        "base_selected_threshold": float(selected_threshold),
        "n_samples": int(N),
        "n_events": int(len(scores)),
        "kept_events": int(keep_count),
        "removed_events": int(remove_count),
        "removed_event_ratio": float(remove_count / (len(scores) + 1e-8)),
        "mask_file": str(out_mask_path) if out_mask_path is not None else "",
    })
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--thresholds", type=str, default="0.30,0.40,0.50,0.60,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "03b_forecast_dataset_multichannel_h10_l5"),
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5"),
    )
    parser.add_argument(
        "--event_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "06c_neurrl_event_dataset_from_multichannel_h10_l5"),
    )
    parser.add_argument(
        "--verifier_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "06c_neurrl_event_dataset_from_multichannel_h10_l5" / "08c_figure_event_verifier_cnn"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5" / "09c_unet_plus_figure_verifier"),
    )
    parser.add_argument(
        "--selected_threshold_json",
        type=str,
        default=str(cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5" / "selected_threshold.json"),
    )
    parser.add_argument("--mask_chunk_size", type=int, default=256)
    parser.add_argument("--no_save_masks", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pred_dir = Path(args.pred_dir)
    event_dir = Path(args.event_dir)
    verifier_dir = Path(args.verifier_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_thresholds(args.thresholds)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    selected_threshold = load_selected_threshold(args.selected_threshold_json)

    print("[DATA]", data_dir)
    print("[PRED]", pred_dir)
    print("[EVENT]", event_dir)
    print("[VERIFIER]", verifier_dir)
    print("[OUT]", out_dir)
    print("[BASE SELECTED THRESHOLD]", selected_threshold)
    print("[VERIFIER THRESHOLDS]", thresholds)

    rows = []
    for thr in thresholds:
        for split in splits:
            m = correct_split(
                split=split,
                threshold=thr,
                data_dir=data_dir,
                pred_dir=pred_dir,
                event_dir=event_dir,
                verifier_dir=verifier_dir,
                out_dir=out_dir,
                selected_threshold=selected_threshold,
                mask_chunk_size=args.mask_chunk_size,
                save_mask=(not args.no_save_masks),
            )
            rows.append(m)
            print("[METRICS]", split, "thr", thr, m)

    out_csv = out_dir / "correction_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "thresholds": thresholds,
        "splits": splits,
        "base_selected_threshold": selected_threshold,
        "note": "Predicted connected components with verifier score below threshold are removed.",
    }
    (out_dir / "correction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[SAVE]", out_csv)
    print("[DONE]", out_dir)


if __name__ == "__main__":
    main()
