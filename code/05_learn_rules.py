#!/usr/bin/env python
"""GitHub-facing entrypoint for event-level and region-level rule learning."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

cfg = SourceFileLoader("cfg", str(Path(__file__).with_name("00_config.py"))).load_module()
EVENT_DIR = cfg.OUTPUT_DIR / "06c_neurrl_event_dataset_from_multichannel_h10_l5"


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


def learn_event_rules() -> None:
    require_file(EVENT_DIR / "multi_event_train.npz", "Run: python code/04_build_event_dataset.py --source multichannel")
    run_legacy("20_build_event_rule_table.py", ["--splits", "train,val,test"])
    run_legacy("21_learn_event_rules.py", ["--splits", "train,val,test", "--max_terms", "3", "--top_k", "10"])


def learn_region_rules() -> None:
    require_file(EVENT_DIR / "figure_event_train.npz", "Run: python code/04_build_event_dataset.py --source multichannel")
    run_legacy("26_build_region_rule_atoms.py", ["--splits", "train,val,test", "--grid", "4"])
    run_legacy("27_learn_region_rules.py", ["--splits", "train,val,test", "--max_terms", "3", "--top_k", "10"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn event-level symbolic and/or region-level rules.")
    parser.add_argument("--type", choices=["event", "region", "all"], default="all")
    args = parser.parse_args()

    if args.type in {"event", "all"}:
        learn_event_rules()
    if args.type in {"region", "all"}:
        learn_region_rules()

    print("\n[DONE] Rule learning stage complete.")


if __name__ == "__main__":
    main()
