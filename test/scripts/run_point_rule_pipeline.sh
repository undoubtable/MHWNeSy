#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "================================================================"
echo "[1/7] Print test config and check label file"
echo "================================================================"
python test/code/00_test_config.py

echo "================================================================"
echo "[2/7] Build point-wise sampled dataset"
echo "================================================================"
python test/code/01_build_point_dataset.py

echo "================================================================"
echo "[3/7] Train point-wise LSTM"
echo "================================================================"
python test/code/02_train_point_lstm.py

echo "================================================================"
echo "[4/7] Evaluate point-wise LSTM"
echo "================================================================"
python test/code/03_eval_point_lstm.py

echo "================================================================"
echo "[5/7] Sweep point-wise probability thresholds"
echo "================================================================"
python test/code/04b_sweep_point_thresholds.py

echo "================================================================"
echo "[6/7] Learn point-level rule baselines"
echo "================================================================"
python test/code/04_learn_point_rules.py

echo "================================================================"
echo "[7/7] Build region candidates from point predictions"
echo "================================================================"
python test/code/05_build_region_from_point.py

echo "================================================================"
echo "[DONE] Point-to-region rule pipeline finished"
echo "================================================================"
