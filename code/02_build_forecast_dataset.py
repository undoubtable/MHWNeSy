#!/usr/bin/env python
"""GitHub-facing entrypoint for forecast dataset construction."""

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


def require_label() -> None:
    if not cfg.LABEL_NC.exists():
        print(f"[MISSING] {cfg.LABEL_NC}")
        print("[HINT] Run: python code/01_build_labels.py --make")
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SSTA-only and/or multichannel forecast datasets.")
    parser.add_argument("--mode", choices=["ssta", "multichannel", "all"], default="multichannel")
    args = parser.parse_args()

    require_label()

    if args.mode in {"ssta", "all"}:
        run_legacy("03_make_forecast_dataset.py")

    if args.mode in {"multichannel", "all"}:
        run_legacy("03b_make_forecast_dataset_multichannel.py")

    print("\n[DONE] Forecast dataset stage complete.")


if __name__ == "__main__":
    main()
