#!/usr/bin/env python
"""Shared configuration for Hierarchical Point-to-Region MHW-NeurRL tests.

This module is intentionally small and explicit. The scripts in ``test/code``
are experimental scaffolding and should keep all generated artifacts under
``test/outputs`` and ``test/logs``.
"""

from __future__ import annotations

from pathlib import Path


# Repository layout ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT_DIR / "test"
TEST_CODE_DIR = TEST_DIR / "code"
TEST_SCRIPT_DIR = TEST_DIR / "scripts"
TEST_OUTPUT_DIR = TEST_DIR / "outputs"
TEST_LOG_DIR = TEST_DIR / "logs"


# Stable label file from the existing 01-07 pipeline. Do not regenerate it in
# this experimental framework; build it with code/01_build_labels.py if needed.
LABEL_FILE = ROOT_DIR / "outputs" / "01_mhw_labels" / "mhw_labels_strict_hobday_1982_2023.nc"


# Forecast setup ------------------------------------------------------------

HISTORY_DAYS = 10
LEAD_DAYS = 5
INPUT_VARIABLES = ("ssta", "exceed90", "mhw")


# Temporal split uses the target day, not the last historical input day.
TRAIN_START = "1982-01-01"
TRAIN_END = "2011-12-31"
VAL_START = "2012-01-01"
VAL_END = "2016-12-31"
TEST_START = "2017-01-01"
TEST_END = "2023-12-31"


# Random point-time sampling keeps this scaffold lightweight. Increase these
# values later when running real experiments.
RANDOM_SEED = 2026
DEFAULT_TRAIN_SAMPLES = 50_000
DEFAULT_VAL_SAMPLES = 10_000
DEFAULT_TEST_SAMPLES = 10_000
DRY_RUN_SAMPLES = 256


# Output paths --------------------------------------------------------------

POINT_DATASET_FILE = TEST_OUTPUT_DIR / f"point_dataset_h{HISTORY_DAYS}_l{LEAD_DAYS}.npz"
POINT_LSTM_DIR = TEST_OUTPUT_DIR / "point_lstm"
POINT_LSTM_MODEL_FILE = POINT_LSTM_DIR / "point_lstm.pt"
POINT_LSTM_PRED_FILE = POINT_LSTM_DIR / "point_lstm_predictions.npz"
FULL_GRID_TEST_PROB_FILE = POINT_LSTM_DIR / "full_grid_test_prob.npy"
FULL_GRID_TEST_PRED_FILE = POINT_LSTM_DIR / "full_grid_test_pred.npy"
FULL_GRID_TEST_META_FILE = POINT_LSTM_DIR / "full_grid_test_meta.npz"
POINT_RULES_FILE = TEST_OUTPUT_DIR / "point_rules.csv"
POINT_THRESHOLD_SWEEP_FILE = TEST_OUTPUT_DIR / "point_threshold_sweep.csv"
REGION_DATASET_FILE = TEST_OUTPUT_DIR / "region_dataset_from_point.csv"
REGION_DATASET_FULL_GRID_FILE = TEST_OUTPUT_DIR / "region_dataset_full_grid.csv"
REGION_RULES_FILE = TEST_OUTPUT_DIR / "region_rules.csv"
REGION_REMOVAL_RULES_FILE = TEST_OUTPUT_DIR / "region_removal_rules.csv"
REGION_CORRECTION_SWEEP_FILE = TEST_OUTPUT_DIR / "region_correction_sweep.csv"
INTENSITY_REGION_DATASET_FILE = TEST_OUTPUT_DIR / "intensity_region_dataset.csv"
INTENSITY_RULES_FILE = TEST_OUTPUT_DIR / "intensity_rules.csv"
INTENSITY_RULE_SUMMARY_FILE = TEST_OUTPUT_DIR / "intensity_rule_summary.json"
REGION_RULE_CORRECTION_SUMMARY_CSV = TEST_OUTPUT_DIR / "region_rule_correction_summary.csv"
REGION_RULE_CORRECTION_SUMMARY_JSON = TEST_OUTPUT_DIR / "region_rule_correction_summary.json"
POINT_LSTM_TRAIN_LOG = TEST_LOG_DIR / "point_lstm_train.log"


def ensure_dirs() -> None:
    """Create all experimental output and log directories."""

    for path in (TEST_OUTPUT_DIR, TEST_LOG_DIR, POINT_LSTM_DIR):
        path.mkdir(parents=True, exist_ok=True)


def print_config() -> None:
    """Print a concise configuration summary for reproducibility checks."""

    rows = {
        "ROOT_DIR": ROOT_DIR,
        "LABEL_FILE": LABEL_FILE,
        "LABEL_FILE_EXISTS": LABEL_FILE.exists(),
        "TEST_OUTPUT_DIR": TEST_OUTPUT_DIR,
        "TEST_LOG_DIR": TEST_LOG_DIR,
        "HISTORY_DAYS": HISTORY_DAYS,
        "LEAD_DAYS": LEAD_DAYS,
        "INPUT_VARIABLES": ", ".join(INPUT_VARIABLES),
        "TRAIN_PERIOD": f"{TRAIN_START} to {TRAIN_END}",
        "VAL_PERIOD": f"{VAL_START} to {VAL_END}",
        "TEST_PERIOD": f"{TEST_START} to {TEST_END}",
        "POINT_DATASET_FILE": POINT_DATASET_FILE,
        "POINT_LSTM_MODEL_FILE": POINT_LSTM_MODEL_FILE,
        "POINT_LSTM_PRED_FILE": POINT_LSTM_PRED_FILE,
        "FULL_GRID_TEST_PROB_FILE": FULL_GRID_TEST_PROB_FILE,
        "FULL_GRID_TEST_PRED_FILE": FULL_GRID_TEST_PRED_FILE,
        "FULL_GRID_TEST_META_FILE": FULL_GRID_TEST_META_FILE,
        "POINT_RULES_FILE": POINT_RULES_FILE,
        "POINT_THRESHOLD_SWEEP_FILE": POINT_THRESHOLD_SWEEP_FILE,
        "REGION_DATASET_FILE": REGION_DATASET_FILE,
        "REGION_DATASET_FULL_GRID_FILE": REGION_DATASET_FULL_GRID_FILE,
        "REGION_RULES_FILE": REGION_RULES_FILE,
        "REGION_REMOVAL_RULES_FILE": REGION_REMOVAL_RULES_FILE,
        "REGION_CORRECTION_SWEEP_FILE": REGION_CORRECTION_SWEEP_FILE,
        "INTENSITY_REGION_DATASET_FILE": INTENSITY_REGION_DATASET_FILE,
        "INTENSITY_RULES_FILE": INTENSITY_RULES_FILE,
        "INTENSITY_RULE_SUMMARY_FILE": INTENSITY_RULE_SUMMARY_FILE,
        "REGION_RULE_CORRECTION_SUMMARY_CSV": REGION_RULE_CORRECTION_SUMMARY_CSV,
        "REGION_RULE_CORRECTION_SUMMARY_JSON": REGION_RULE_CORRECTION_SUMMARY_JSON,
    }

    print("[MHW-NeurRL test config]")
    for key, value in rows.items():
        print(f"{key}: {value}")


def main() -> None:
    ensure_dirs()
    print_config()
    if not LABEL_FILE.exists():
        raise SystemExit(
            f"[MISSING] {LABEL_FILE}\n"
            "Build labels first with: python code/01_build_labels.py --all"
        )
    print("[OK] LABEL_FILE exists.")


if __name__ == "__main__":
    main()
