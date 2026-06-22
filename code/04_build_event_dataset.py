#!/usr/bin/env python
"""GitHub-facing entrypoint for candidate event dataset construction."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()


def run_legacy(script: str, args: list[str] | None = None) -> None:
    args = args or []
    env = os.environ.copy()
    env["MHWNEURRL_ROOT"] = str(cfg.ROOT)
    print("\n" + "=" * 72)
    print(f"[RUN] {script} {' '.join(args)}")
    print("=" * 72)
    subprocess.run([sys.executable, str(cfg.LEGACY_CODE_DIR / script), *args], cwd=str(cfg.ROOT), env=env, check=True)


def require_file(path: Path, hint: str) -> None:
    if not path.exists():
        print(f"[MISSING] {path}")
        print(f"[HINT] {hint}")
        raise SystemExit(2)


def build_ssta() -> None:
    require_file(cfg.OUTPUT_DIR / "04_unet_baseline_h10_l5" / "pred_prob_train.npy",
                 "Run: python code/03_train_eval_unet.py --model ssta --train --eval")
    run_legacy("06_make_figure_neurrl_event_dataset.py", [
        "--splits", "train,val,test",
        "--crop_size", "64",
        "--min_area", "8",
        "--iou_thr", "0.10",
        "--overlap_thr", "0.30",
        "--balance_train",
    ])
    run_legacy("07_make_multineurrl_event_sequence_dataset.py", ["--splits", "train,val,test"])


def build_multichannel() -> None:
    pred_dir = cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5"
    require_file(pred_dir / "pred_prob_train.npy",
                 "Run: python code/03_train_eval_unet.py --model multichannel --train --eval")
    require_file(pred_dir / "selected_threshold.json",
                 "Run: python code/05c_eval_unet_multichannel.py through the main U-Net wrapper.")
    run_legacy("06c_make_figure_event_dataset_multichannel.py", [
        "--splits", "train,val,test",
        "--crop_size", "64",
        "--min_area", "8",
        "--iou_thr", "0.10",
        "--overlap_thr", "0.30",
    ])
    run_legacy("07c_make_multineurrl_event_sequence_multichannel.py", ["--splits", "train,val,test"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate event datasets from U-Net predictions.")
    parser.add_argument("--source", choices=["ssta", "multichannel", "all"], default="multichannel")
    args = parser.parse_args()

    if args.source in {"ssta", "all"}:
        build_ssta()
    if args.source in {"multichannel", "all"}:
        build_multichannel()

    print("\n[DONE] Event dataset stage complete.")


if __name__ == "__main__":
    main()
