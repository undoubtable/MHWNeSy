#!/usr/bin/env python
"""GitHub-facing entrypoint for applying and comparing the symbolic rule verifier."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()
RULE_DIR = cfg.OUTPUT_DIR / "20_event_rule_learning"
COMPARE_CSV = cfg.OUTPUT_DIR / "23_rule_verifier_comparison" / "final_comparison.csv"


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


def print_comparison() -> None:
    if not COMPARE_CSV.exists():
        return
    rows = list(csv.DictReader(COMPARE_CSV.open()))
    if not rows:
        return
    print("\n[SUMMARY] Symbolic rule verifier comparison")
    for row in rows:
        name = row.get("method", row.get("model", "method"))
        print(f"- {name}")
        for key in ["pixel_precision", "pixel_recall", "pixel_f1", "pixel_iou", "pixel_acc"]:
            if key in row:
                print(f"  {key}: {row[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply and compare symbolic event-rule verifier correction.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not any([args.apply, args.compare, args.all]):
        args.all = True

    if args.all or args.apply:
        require_file(RULE_DIR / "event_rule_predictions_train.csv", "Run: python code/05_learn_rules.py --type event")
        run_legacy("22_apply_event_rule_verifier.py", ["--splits", "train,val,test"])

    if args.all or args.compare:
        require_file(RULE_DIR / "rule_verifier_correction" / "rule_correction_metrics.csv",
                     "Run: python code/06_apply_rule_verifier.py --apply")
        run_legacy("23_compare_multichannel_unet_vs_rule_verifier.py")
        print_comparison()

    print("\n[DONE] Rule verifier stage complete.")


if __name__ == "__main__":
    main()
