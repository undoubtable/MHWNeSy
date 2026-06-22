#!/usr/bin/env python
"""GitHub-facing entrypoint for MHW label construction and checks."""

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
    path = cfg.LEGACY_CODE_DIR / script
    env = os.environ.copy()
    env["MHWNEURRL_ROOT"] = str(cfg.ROOT)
    print("\n" + "=" * 72)
    print(f"[RUN] {script} {' '.join(args)}")
    print("=" * 72)
    subprocess.run([sys.executable, str(path), *args], cwd=str(cfg.ROOT), env=env, check=True)


def require_file(path: Path, hint: str) -> None:
    if not path.exists():
        print(f"[MISSING] {path}")
        print(f"[HINT] {hint}")
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, visualize, and validate MHW labels.")
    parser.add_argument("--make", action="store_true", help="Build strict Hobday-style MHW labels.")
    parser.add_argument("--visualize", action="store_true", help="Create label visualization figures.")
    parser.add_argument("--validate", action="store_true", help="Run label validation diagnostics.")
    parser.add_argument("--all", action="store_true", help="Run make, visualize, and validate.")
    args = parser.parse_args()

    if not any([args.make, args.visualize, args.validate, args.all]):
        args.all = True

    if args.all or args.make:
        require_file(cfg.RAW_NC, "Place NOAA OISST data at data/oisst_scs_1982_2023.nc.")
        run_legacy("01_make_mhw_labels.py")

    if args.all or args.visualize:
        require_file(cfg.LABEL_NC, "Run: python code/01_build_labels.py --make")
        run_legacy("02_visualize_mhw_labels.py")

    if args.all or args.validate:
        require_file(cfg.LABEL_NC, "Run: python code/01_build_labels.py --make")
        run_legacy("02b_validate_mhw_labels.py")

    print("\n[DONE] Label stage complete.")


if __name__ == "__main__":
    main()
