# -*- coding: utf-8 -*-
"""
Compare multichannel U-Net baseline and event verifier correction.

Select verifier threshold by validation split, then report train/val/test.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from importlib.machinery import SourceFileLoader
cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select_split", type=str, default="val")
    parser.add_argument("--select_metric", type=str, default="pixel_f1")
    parser.add_argument(
        "--baseline_csv",
        type=str,
        default=str(cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5" / "eval_metrics.csv"),
    )
    parser.add_argument(
        "--correction_csv",
        type=str,
        default=str(cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5" / "09c_unet_plus_figure_verifier" / "correction_metrics.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5" / "10c_compare_baseline_vs_verifier"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.baseline_csv)
    corr = pd.read_csv(args.correction_csv)

    sub = corr[corr["split"] == args.select_split].copy()
    if len(sub) == 0:
        raise ValueError(f"No correction rows for select split: {args.select_split}")

    best_row = sub.loc[sub[args.select_metric].idxmax()]
    best_thr = float(best_row["threshold"])

    print("[SELECTED VERIFIER THRESHOLD]", best_thr)
    print("[SELECTED BY]", args.select_split, args.select_metric)

    rows = []
    for split in ["train", "val", "test"]:
        b = base[base["split"] == split].iloc[0].to_dict()
        c = corr[(corr["split"] == split) & (corr["threshold"] == best_thr)].iloc[0].to_dict()

        row_base = {
            "split": split,
            "method": "Multichannel U-Net selected threshold",
            "threshold": b.get("threshold", ""),
            "verifier_threshold": "",
            "pixel_precision": b["pixel_precision"],
            "pixel_recall": b["pixel_recall"],
            "pixel_f1": b["pixel_f1"],
            "pixel_iou": b["pixel_iou"],
            "pixel_acc": b["pixel_acc"],
            "pred_pos_ratio": b.get("pred_pos_ratio", ""),
            "true_pos_ratio": b.get("true_pos_ratio", ""),
            "removed_event_ratio": "",
        }

        row_corr = {
            "split": split,
            "method": "Multichannel U-Net + event verifier",
            "threshold": c.get("base_selected_threshold", b.get("threshold", "")),
            "verifier_threshold": best_thr,
            "pixel_precision": c["pixel_precision"],
            "pixel_recall": c["pixel_recall"],
            "pixel_f1": c["pixel_f1"],
            "pixel_iou": c["pixel_iou"],
            "pixel_acc": c["pixel_acc"],
            "pred_pos_ratio": c["pred_pos_ratio"],
            "true_pos_ratio": c["true_pos_ratio"],
            "removed_event_ratio": c["removed_event_ratio"],
        }

        row_delta = {
            "split": split,
            "method": "Delta",
            "threshold": c.get("base_selected_threshold", b.get("threshold", "")),
            "verifier_threshold": best_thr,
            "pixel_precision": c["pixel_precision"] - b["pixel_precision"],
            "pixel_recall": c["pixel_recall"] - b["pixel_recall"],
            "pixel_f1": c["pixel_f1"] - b["pixel_f1"],
            "pixel_iou": c["pixel_iou"] - b["pixel_iou"],
            "pixel_acc": c["pixel_acc"] - b["pixel_acc"],
            "pred_pos_ratio": c["pred_pos_ratio"] - b["pred_pos_ratio"],
            "true_pos_ratio": c["true_pos_ratio"] - b["true_pos_ratio"],
            "removed_event_ratio": "",
        }

        rows.extend([row_base, row_corr, row_delta])

    out = pd.DataFrame(rows)
    out_csv = out_dir / "final_comparison.csv"
    out.to_csv(out_csv, index=False)

    selected = {
        "selected_verifier_threshold": best_thr,
        "select_split": args.select_split,
        "select_metric": args.select_metric,
        "selected_val_row": best_row.to_dict(),
    }
    selected_json = out_dir / "selected_verifier_threshold.json"
    selected_json.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    print("\n[FINAL COMPARISON]")
    print(out.to_string(index=False))
    print("[SAVE]", out_csv)
    print("[SAVE]", selected_json)


if __name__ == "__main__":
    main()
