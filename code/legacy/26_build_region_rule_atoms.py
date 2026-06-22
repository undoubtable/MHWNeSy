#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build region-level boolean atoms from multichannel event patches.

Outputs:
    outputs/26_region_rule_learning/region_atoms_train.csv
    outputs/26_region_rule_learning/region_atoms_val.csv
    outputs/26_region_rule_learning/region_atoms_test.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def parse_splits(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def region_slices(size, grid):
    edges = np.linspace(0, size, grid + 1, dtype=int)
    return [(edges[i], edges[i + 1]) for i in range(grid)]


def build_split(event_dir, out_dir, split, grid):
    z = np.load(event_dir / f"figure_event_{split}.npz", allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y_valid"].astype(np.uint8)
    meta = z["meta"].astype(np.float32)

    H, W = X.shape[-2:]
    rs = region_slices(H, grid)
    cs = region_slices(W, grid)
    rows = []

    for i in range(X.shape[0]):
        row = {
            "split": split,
            "event_index": int(i),
            "y_valid": int(y[i]),
            "sample_index": int(meta[i, 0]),
            "component_id": int(meta[i, 1]),
            "area_px": float(meta[i, 2]),
            "best_iou": float(meta[i, 3]),
            "overlap_ratio": float(meta[i, 4]),
        }
        recent = {
            "SSTA": X[i, 9],
            "TGAP": X[i, 39],
            "MHW": X[i, 19],
            "EXC": X[i, 29],
            "COMP": X[i, 40],
        }
        for r, (r0, r1) in enumerate(rs, start=1):
            for c, (c0, c1) in enumerate(cs, start=1):
                tag = f"R{r}C{c}"
                comp = recent["COMP"][r0:r1, c0:c1] > 0.5
                comp_occ = float(comp.mean())
                row[f"COMP_{tag}_occ"] = comp_occ
                row[f"COMP_{tag}_ACTIVE"] = int(comp_occ >= 0.05)

                gap = recent["TGAP"][r0:r1, c0:c1]
                row[f"TGAP_{tag}_mean"] = float(np.mean(gap))
                row[f"TGAP_{tag}_max"] = float(np.max(gap))
                row[f"TGAP_{tag}_HIGH"] = int(np.mean(gap) > 0.5)
                row[f"TGAP_{tag}_LOW"] = int(np.mean(gap) <= 0.0)

                ssta = recent["SSTA"][r0:r1, c0:c1]
                row[f"SSTA_{tag}_mean"] = float(np.mean(ssta))
                row[f"SSTA_{tag}_HIGH"] = int(np.mean(ssta) > 0.5)

                mhw_occ = float((recent["MHW"][r0:r1, c0:c1] > 0.5).mean())
                exc_occ = float((recent["EXC"][r0:r1, c0:c1] > 0.5).mean())
                row[f"MHW_{tag}_occ"] = mhw_occ
                row[f"EXC_{tag}_occ"] = exc_occ
                row[f"MHW_{tag}_ACTIVE"] = int(mhw_occ >= 0.05)
                row[f"EXCEED90_{tag}_ACTIVE"] = int(exc_occ >= 0.05)
        rows.append(row)

    out_csv = out_dir / f"region_atoms_{split}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("[SAVE]", out_csv, "rows=", len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_dir", type=str, default=str(cfg.OUTPUT_DIR / "06c_neurrl_event_dataset_from_multichannel_h10_l5"))
    parser.add_argument("--out_dir", type=str, default=str(cfg.OUTPUT_DIR / "26_region_rule_learning"))
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--grid", type=int, default=4)
    args = parser.parse_args()

    event_dir = Path(args.event_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in parse_splits(args.splits):
        build_split(event_dir, out_dir, split, args.grid)


if __name__ == "__main__":
    main()
