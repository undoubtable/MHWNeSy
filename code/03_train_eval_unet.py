#!/usr/bin/env python
"""GitHub-facing entrypoint for U-Net training and evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()

SSTA_DATA = cfg.OUTPUT_DIR / "03_forecast_dataset_h10_l5"
MULTI_DATA = cfg.OUTPUT_DIR / "03b_forecast_dataset_multichannel_h10_l5"
SSTA_RUN = cfg.OUTPUT_DIR / "04_unet_baseline_h10_l5"
MULTI_RUN = cfg.OUTPUT_DIR / "04c_unet_multichannel_h10_l5"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and/or evaluate SSTA-only or multichannel U-Net.")
    parser.add_argument("--model", choices=["ssta", "multichannel"], default="multichannel")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if not args.train and not args.eval:
        args.train = True
        args.eval = True

    if args.model == "ssta":
        data_dir = SSTA_DATA
        run_dir = SSTA_RUN
        train_script = "04_train_unet_baseline.py"
        eval_script = "05_eval_unet_baseline.py"
        build_hint = "Run: python code/02_build_forecast_dataset.py --mode ssta"
    else:
        data_dir = MULTI_DATA
        run_dir = MULTI_RUN
        train_script = "04c_train_unet_multichannel.py"
        eval_script = "05c_eval_unet_multichannel.py"
        build_hint = "Run: python code/02_build_forecast_dataset.py --mode multichannel"

    if args.train:
        require_file(data_dir / "X_train.npy", build_hint)
        require_file(data_dir / "y_train.npy", build_hint)
        run_legacy(train_script)

    if args.eval:
        require_file(run_dir / "best_model.pt", f"Run: python code/03_train_eval_unet.py --model {args.model} --train")
        run_legacy(eval_script)

    print("\n[DONE] U-Net stage complete.")


if __name__ == "__main__":
    main()
