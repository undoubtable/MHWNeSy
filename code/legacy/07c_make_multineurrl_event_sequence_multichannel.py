# -*- coding: utf-8 -*-
"""
Convert multichannel Figure-NeurRL event patches into event sequences.

Input:
    outputs/06c_neurrl_event_dataset_from_multichannel_h10_l5/
        figure_event_train/val/test.npz

Output:
    outputs/06c_neurrl_event_dataset_from_multichannel_h10_l5/
        multi_event_train/val/test.npz
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


VARIABLE_NAMES = np.array([
    "mean_ssta_inside_candidate",
    "max_ssta_inside_candidate",
    "mean_threshold_gap_inside_candidate",
    "max_threshold_gap_inside_candidate",
    "mhw_fraction_inside_candidate",
    "exceed90_fraction_inside_candidate",
    "mean_ssta_context",
    "component_area_fraction",
])


def convert_one(in_file: Path, out_file: Path, history: int):
    z = np.load(in_file, allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y_valid"].astype(np.uint8)
    meta = z["meta"].astype(np.float32)

    # X: [N, 4 * history + 1, crop, crop], last channel = candidate mask.
    ssta = X[:, 0:history]
    mhw = X[:, history:2 * history]
    exceed90 = X[:, 2 * history:3 * history]
    threshold_gap = X[:, 3 * history:4 * history]
    mask = X[:, -1] > 0.5

    N, T, H, W = ssta.shape
    out = np.zeros((N, len(VARIABLE_NAMES), T), dtype=np.float32)

    for i in tqdm(range(N), desc=f"Convert {in_file.name}"):
        m = mask[i]
        context = ~m
        area_fraction = float(m.mean())

        for t in range(T):
            inside_ssta = ssta[i, t][m]
            inside_gap = threshold_gap[i, t][m]
            inside_mhw = mhw[i, t][m]
            inside_exceed = exceed90[i, t][m]
            ctx_ssta = ssta[i, t][context]

            if inside_ssta.size == 0:
                continue

            out[i, 0, t] = float(np.mean(inside_ssta))
            out[i, 1, t] = float(np.max(inside_ssta))
            out[i, 2, t] = float(np.mean(inside_gap))
            out[i, 3, t] = float(np.max(inside_gap))
            out[i, 4, t] = float(np.mean(inside_mhw > 0.5))
            out[i, 5, t] = float(np.mean(inside_exceed > 0.5))
            out[i, 6, t] = float(np.mean(ctx_ssta)) if ctx_ssta.size else 0.0
            out[i, 7, t] = area_fraction

    np.savez_compressed(
        out_file,
        X_seq=out,
        y=y,
        meta=meta,
        variable_names=VARIABLE_NAMES,
        description=np.array([
            "X_seq shape [N, V, history]. Variables summarize multichannel event patches inside the predicted component mask."
        ]),
    )
    print("[SAVE]", out_file, out.shape, y.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "06c_neurrl_event_dataset_from_multichannel_h10_l5"),
    )
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--history", type=int, default=10)
    args = parser.parse_args()

    event_dir = Path(args.event_dir)
    for split in args.splits.split(","):
        split = split.strip()
        if not split:
            continue
        convert_one(
            event_dir / f"figure_event_{split}.npz",
            event_dir / f"multi_event_{split}.npz",
            history=args.history,
        )


if __name__ == "__main__":
    main()
