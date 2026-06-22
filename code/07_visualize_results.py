#!/usr/bin/env python
"""GitHub-facing entrypoint for rule and correction visualizations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()
RULE_DIR = cfg.OUTPUT_DIR / "20_event_rule_learning"
REGION_DIR = cfg.OUTPUT_DIR / "26_region_rule_learning"


def run_legacy(script: str, args: list[str]) -> None:
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


def visualize_removed() -> None:
    require_file(RULE_DIR / "rule_verifier_correction" / "rule_corrected_mask_test.npy",
                 "Run: python code/06_apply_rule_verifier.py --all")
    run_legacy("24b_visualize_removed_rule_events_compact.py", [
        "--split", "test",
        "--top_invalid", "20",
        "--top_valid", "10",
    ])


def visualize_event_rules() -> None:
    require_file(RULE_DIR / "learned_valid_rules.txt", "Run: python code/05_learn_rules.py --type event")
    require_file(RULE_DIR / "event_rule_table_test.csv", "Run: python code/05_learn_rules.py --type event")
    run_legacy("25b_visualize_event_symbolic_rules.py", [
        "--split", "test",
        "--top_rules", "5",
        "--examples_per_rule", "3",
    ])


def visualize_region_rules() -> None:
    require_file(REGION_DIR / "learned_valid_region_rules.txt", "Run: python code/05_learn_rules.py --type region")
    run_legacy("28_visualize_region_rules.py", [
        "--split", "test",
        "--top_k", "5",
        "--examples_per_rule", "2",
        "--grid", "4",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create compact correction and rule visualization figures.")
    parser.add_argument("--type", choices=["removed", "event_rules", "region_rules", "all"], default="all")
    args = parser.parse_args()

    if args.type in {"removed", "all"}:
        visualize_removed()
    if args.type in {"event_rules", "all"}:
        visualize_event_rules()
    if args.type in {"region_rules", "all"}:
        visualize_region_rules()

    print("\n[DONE] Visualization stage complete.")


if __name__ == "__main__":
    main()
