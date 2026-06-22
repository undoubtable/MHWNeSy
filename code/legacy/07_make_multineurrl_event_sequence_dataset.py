# -*- coding: utf-8 -*-
"""
Convert Figure-NeurRL event patches into Multi-NeurRL-style multivariate sequences.

Input:
    figure_event_train/val/test.npz

Output:
    multi_event_train/val/test.npz

Each output:
    X_seq   float32 [N, V, T]
    y       uint8   [N]

Variables V:
    0 mean_ssta_inside_candidate
    1 max_ssta_inside_candidate
    2 warm_fraction_inside_candidate, normalized-SSTA > 0
    3 mean_ssta_context
    4 max_ssta_context
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def convert_one(in_file: Path, out_file: Path):
    z = np.load(in_file)
    X = z["X"].astype(np.float32)
    y = z["y_valid"].astype(np.uint8)

    # X: [N, history+1, crop, crop], last channel = candidate mask.
    ssta = X[:, :-1]
    mask = X[:, -1] > 0.5

    N, T, H, W = ssta.shape
    V = 5
    out = np.zeros((N, V, T), dtype=np.float32)

    mask_area = mask.reshape(N, -1).sum(axis=1).astype(np.float32)
    mask_area = np.maximum(mask_area, 1.0)

    for i in tqdm(range(N), desc=f"Convert {in_file.name}"):
        m = mask[i]
        context = ~m

        for t in range(T):
            a = ssta[i, t]
            inside = a[m]
            ctx = a[context]

            if inside.size == 0:
                continue

            out[i, 0, t] = float(np.mean(inside))
            out[i, 1, t] = float(np.max(inside))
            out[i, 2, t] = float(np.mean(inside > 0.0))
            out[i, 3, t] = float(np.mean(ctx)) if ctx.size else 0.0
            out[i, 4, t] = float(np.max(ctx)) if ctx.size else 0.0

    np.savez_compressed(
        out_file,
        X_seq=out,
        y=y,
        variable_names=np.array([
            "mean_ssta_inside_candidate",
            "max_ssta_inside_candidate",
            "warm_fraction_inside_candidate",
            "mean_ssta_context",
            "max_ssta_context",
        ]),
    )
    print("[SAVE]", out_file, out.shape, y.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_dir", type=str, default=str(cfg.EVENT_DIR))
    parser.add_argument("--splits", type=str, default="train,val,test")
    args = parser.parse_args()

    event_dir = Path(args.event_dir)

    for split in args.splits.split(","):
        split = split.strip()
        if not split:
            continue

        in_file = event_dir / f"figure_event_{split}.npz"
        out_file = event_dir / f"multi_event_{split}.npz"
        convert_one(in_file, out_file)


if __name__ == "__main__":
    main()
