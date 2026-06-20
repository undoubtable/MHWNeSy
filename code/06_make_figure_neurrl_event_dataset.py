# -*- coding: utf-8 -*-
"""
Make object/event-level candidate dataset for Figure-NeurRL.

Idea:
    U-Net predicts MHW mask.
    Connected components are candidate MHW events.
    Each candidate is cropped as a spatiotemporal patch.
    Label candidate as valid if it overlaps a true MHW component.

Outputs:
    /ybz/ybz/2026/MHWNeurRL/data/event_h10_l5/
        figure_event_train.npz
        figure_event_val.npz
        figure_event_test.npz

Each .npz contains:
    X           float16 [N, history + 1, crop, crop]
                first history channels: normalized SSTA sequence
                last channel: candidate component mask
    y_valid     uint8 [N], 1 = valid MHW candidate, 0 = false alarm
    meta        float32 [N, 5], columns:
                sample_index, component_id, area_px, best_iou, overlap_ratio
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from scipy import ndimage

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def crop_with_padding(arr, center_y, center_x, crop_size, fill=0):
    """
    arr shape: [C, H, W] or [H, W]
    """
    is_2d = arr.ndim == 2
    if is_2d:
        arr = arr[None, ...]

    C, H, W = arr.shape
    half = crop_size // 2

    y0 = center_y - half
    y1 = y0 + crop_size
    x0 = center_x - half
    x1 = x0 + crop_size

    out = np.full((C, crop_size, crop_size), fill, dtype=arr.dtype)

    sy0 = max(0, y0)
    sy1 = min(H, y1)
    sx0 = max(0, x0)
    sx1 = min(W, x1)

    dy0 = sy0 - y0
    dy1 = dy0 + (sy1 - sy0)
    dx0 = sx0 - x0
    dx1 = dx0 + (sx1 - sx0)

    out[:, dy0:dy1, dx0:dx1] = arr[:, sy0:sy1, sx0:sx1]

    if is_2d:
        return out[0]
    return out


def component_best_match(comp_mask, true_mask):
    """
    Match candidate component to true connected components.
    Return best IoU and overlap ratio.
    """
    true_lab, true_n = ndimage.label(true_mask.astype(bool))
    comp_area = comp_mask.sum()

    if comp_area == 0 or true_n == 0:
        return 0.0, 0.0

    best_iou = 0.0
    best_overlap = 0.0

    for k in range(1, true_n + 1):
        gt = true_lab == k
        inter = np.logical_and(comp_mask, gt).sum()
        if inter == 0:
            continue
        union = np.logical_or(comp_mask, gt).sum()
        iou = inter / (union + 1e-8)
        overlap = inter / (comp_area + 1e-8)
        best_iou = max(best_iou, iou)
        best_overlap = max(best_overlap, overlap)

    return float(best_iou), float(best_overlap)


def process_split(args, split):
    data_dir = Path(args.data_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_all = np.load(data_dir / f"X_{split}.npy", mmap_mode="r")
    y_all = np.load(data_dir / f"y_{split}.npy", mmap_mode="r")
    pred_all = np.load(pred_dir / f"pred_mask_{split}.npy", mmap_mode="r")

    N, history, H, W = X_all.shape
    print(f"[{split}] X={X_all.shape}, y={y_all.shape}, pred={pred_all.shape}")

    X_events = []
    y_valid = []
    meta = []

    cand_count = 0

    for i in tqdm(range(N), desc=f"Events {split}"):
        pred = pred_all[i].astype(bool)
        true = y_all[i].astype(bool)

        lab, n_comp = ndimage.label(pred)
        if n_comp == 0:
            continue

        for cid in range(1, n_comp + 1):
            comp = lab == cid
            area = int(comp.sum())
            if area < args.min_area:
                continue

            ys, xs = np.where(comp)
            cy = int(np.round(ys.mean()))
            cx = int(np.round(xs.mean()))

            best_iou, overlap = component_best_match(comp, true)
            valid = int((best_iou >= args.iou_thr) or (overlap >= args.overlap_thr))

            x_patch = crop_with_padding(X_all[i], cy, cx, args.crop_size, fill=0).astype(np.float16)
            m_patch = crop_with_padding(comp.astype(np.float32), cy, cx, args.crop_size, fill=0).astype(np.float16)
            x_event = np.concatenate([x_patch, m_patch[None, ...]], axis=0)

            X_events.append(x_event)
            y_valid.append(valid)
            meta.append([i, cid, area, best_iou, overlap])

            cand_count += 1
            if args.max_candidates > 0 and cand_count >= args.max_candidates:
                break

        if args.max_candidates > 0 and cand_count >= args.max_candidates:
            break

    if not X_events:
        raise RuntimeError(f"No event candidates found for split={split}.")

    X_events = np.stack(X_events).astype(np.float16)
    y_valid = np.array(y_valid, dtype=np.uint8)
    meta = np.array(meta, dtype=np.float32)

    # Optional simple balancing for train only.
    if split == "train" and args.balance_train:
        rng = np.random.default_rng(args.seed)
        pos_idx = np.where(y_valid == 1)[0]
        neg_idx = np.where(y_valid == 0)[0]

        max_neg = min(len(neg_idx), max(len(pos_idx) * args.neg_pos_ratio, len(pos_idx)))
        if len(pos_idx) > 0 and len(neg_idx) > max_neg:
            neg_keep = rng.choice(neg_idx, size=max_neg, replace=False)
            keep = np.concatenate([pos_idx, neg_keep])
            rng.shuffle(keep)

            X_events = X_events[keep]
            y_valid = y_valid[keep]
            meta = meta[keep]

    out_file = out_dir / f"figure_event_{split}.npz"
    np.savez_compressed(
        out_file,
        X=X_events,
        y_valid=y_valid,
        meta=meta,
        meta_columns=np.array(["sample_index", "component_id", "area_px", "best_iou", "overlap_ratio"]),
        description=np.array([
            "X shape [N, history+1, crop, crop]. Last channel is predicted component mask."
        ]),
    )

    print("[SAVE]", out_file)
    print("[STATS]", split, "N=", len(y_valid), "valid=", int(y_valid.sum()), "invalid=", int((1 - y_valid).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(cfg.FORECAST_DIR))
    parser.add_argument("--pred_dir", type=str, default=str(cfg.UNET_RUN_DIR))
    parser.add_argument("--out_dir", type=str, default=str(cfg.EVENT_DIR))
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--min_area", type=int, default=8)
    parser.add_argument("--iou_thr", type=float, default=0.10)
    parser.add_argument("--overlap_thr", type=float, default=0.30)
    parser.add_argument("--max_candidates", type=int, default=0)
    parser.add_argument("--balance_train", action="store_true")
    parser.add_argument("--neg_pos_ratio", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for split in args.splits.split(","):
        split = split.strip()
        if split:
            process_split(args, split)


if __name__ == "__main__":
    main()
