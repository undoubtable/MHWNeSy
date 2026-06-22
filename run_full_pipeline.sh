#!/usr/bin/env bash
set -euo pipefail

# Compact GitHub-facing MHW-NeurRL pipeline runner.
# Usage:
#   bash run_full_pipeline.sh
# Optional:
#   MHWNEURRL_ROOT=/path/to/MHWNeurRL bash run_full_pipeline.sh

PROJECT_ROOT="${MHWNEURRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export MHWNEURRL_ROOT="$PROJECT_ROOT"
DATA_NC="$PROJECT_ROOT/data/oisst_scs_1982_2023.nc"

if [[ ! -f "$DATA_NC" ]]; then
  echo "[ERROR] Raw data not found: $DATA_NC"
  echo "Put oisst_scs_1982_2023.nc under data/ first."
  exit 1
fi

cd "$PROJECT_ROOT"

echo "[ROOT] $PROJECT_ROOT"
echo "[DATA] $DATA_NC"
echo

echo "[1/7] Build labels"
python code/01_build_labels.py --all

echo "[2/7] Build multichannel forecast dataset"
python code/02_build_forecast_dataset.py --mode multichannel

echo "[3/7] Train and evaluate multichannel U-Net"
python code/03_train_eval_unet.py --model multichannel --train --eval

echo "[4/7] Build multichannel event dataset"
python code/04_build_event_dataset.py --source multichannel

echo "[5/7] Learn event-level and region-level rules"
python code/05_learn_rules.py --type all

echo "[6/7] Apply symbolic rule verifier"
python code/06_apply_rule_verifier.py --all

echo "[7/7] Visualize results"
python code/07_visualize_results.py --type all

echo
echo "[DONE] Compact MHW-NeurRL pipeline finished."
