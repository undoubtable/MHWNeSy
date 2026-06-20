#!/usr/bin/env bash
set -euo pipefail

# Full MHW-NeurRL pipeline runner.
# Usage:
#   bash run_full_pipeline.sh
# Optional examples:
#   FORCE=1 bash run_full_pipeline.sh
#   EPOCHS_UNET=10 EPOCHS_VERIFIER=5 bash run_full_pipeline.sh
#   MHWNEURRL_ROOT=/path/to/MHWNeurRL bash run_full_pipeline.sh

PROJECT_ROOT="${MHWNEURRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export MHWNEURRL_ROOT="$PROJECT_ROOT"
CODE_DIR="$PROJECT_ROOT/code"
DATA_NC="$PROJECT_ROOT/data/oisst_scs_1982_2023.nc"
OUTPUT_DIR="$PROJECT_ROOT/outputs"

# Main parameters.
HISTORY="${HISTORY:-10}"
LEAD="${LEAD:-5}"
STRIDE="${STRIDE:-1}"
UNET_EPOCHS="${EPOCHS_UNET:-30}"
UNET_BATCH="${BATCH_UNET:-32}"
VERIFIER_EPOCHS="${EPOCHS_VERIFIER:-20}"
VERIFIER_BATCH="${BATCH_VERIFIER:-256}"
VIS_DATE="${VIS_DATE:-2023-07-01}"
VIS_YEAR="${VIS_YEAR:-2023}"
THRESHOLDS="${THRESHOLDS:-0.30,0.40,0.50,0.60,0.70,0.75,0.80,0.85,0.90}"
CORRECTION_THR="${CORRECTION_THR:-0.75}"
FORCE="${FORCE:-0}"

LABEL_NC="$OUTPUT_DIR/01_mhw_labels/mhw_labels_strict_hobday_1982_2023.nc"
FORECAST_DONE="$OUTPUT_DIR/03_forecast_dataset_h10_l5/X_train.npy"
UNET_BEST="$OUTPUT_DIR/04_unet_baseline_h10_l5/best_model.pt"
UNET_EVAL="$OUTPUT_DIR/04_unet_baseline_h10_l5/eval_metrics.csv"
FIG_EVENT="$OUTPUT_DIR/06_neurrl_event_dataset_h10_l5/figure_event_train.npz"
MULTI_EVENT="$OUTPUT_DIR/06_neurrl_event_dataset_h10_l5/multi_event_train.npz"
VERIFIER_BEST="$OUTPUT_DIR/06_neurrl_event_dataset_h10_l5/08_figure_event_verifier_cnn/best_model.pt"
CORR_CSV="$OUTPUT_DIR/04_unet_baseline_h10_l5/09_unet_plus_figure_verifier/correction_metrics.csv"
FINAL_CSV="$OUTPUT_DIR/04_unet_baseline_h10_l5/10_compare_baseline_vs_verifier/final_comparison.csv"
VIS_DIR="$OUTPUT_DIR/04_unet_baseline_h10_l5/11_correction_visualization"

run_step() {
  local name="$1"
  local target="$2"
  shift 2
  echo
  echo "============================================================"
  echo "[STEP] $name"
  echo "============================================================"
  if [[ "$FORCE" != "1" && -e "$target" ]]; then
    echo "[SKIP] $target exists. Use FORCE=1 to rerun."
  else
    "$@"
  fi
}

if [[ ! -f "$DATA_NC" ]]; then
  echo "[ERROR] Raw data not found: $DATA_NC"
  echo "Put oisst_scs_1982_2023.nc under data/ first."
  exit 1
fi

cd "$CODE_DIR"
echo "[ROOT] $PROJECT_ROOT"
echo "[CODE] $CODE_DIR"
echo "[DATA] $DATA_NC"
echo "[OUTPUT] $OUTPUT_DIR"

run_step "01 Strict Hobday MHW labels" "$LABEL_NC" \
  python 01_make_mhw_labels.py \
    --raw_nc "$DATA_NC" \
    --out_nc "$LABEL_NC" \
    --clim_start 1982 --clim_end 2011 \
    --window_half_width 5 \
    --smooth_width 31 \
    --percentile 90 \
    --min_duration 5 \
    --max_gap 2

run_step "02 Label visualization" "$OUTPUT_DIR/02_label_visualization/mhw_area_${VIS_YEAR}.png" \
  python 02_visualize_mhw_labels.py --date "$VIS_DATE" --year "$VIS_YEAR"

run_step "02b Label validation" "$OUTPUT_DIR/02_label_validation/label_summary.json" \
  python 02b_validate_mhw_labels.py

run_step "02c Literature comparison" "$OUTPUT_DIR/02_label_validation/literature_comparison/literature_comparison_summary.json" \
  python 02c_literature_comparison.py

run_step "02d Corrected literature plot" "$OUTPUT_DIR/02_label_validation/literature_comparison/jja_mhw_days_trend_bar_comparison.png" \
  python 02d_fix_literature_comparison_plot.py

run_step "03 Forecast dataset" "$FORECAST_DONE" \
  python 03_make_forecast_dataset.py --history "$HISTORY" --lead "$LEAD" --stride "$STRIDE"

run_step "04 U-Net baseline training" "$UNET_BEST" \
  python 04_train_unet_baseline.py --epochs "$UNET_EPOCHS" --batch_size "$UNET_BATCH" --history "$HISTORY"

run_step "05 U-Net baseline evaluation" "$UNET_EVAL" \
  python 05_eval_unet_baseline.py --splits train,val,test --batch_size 64

run_step "06 Figure-NeurRL event dataset" "$FIG_EVENT" \
  python 06_make_figure_neurrl_event_dataset.py \
    --splits train,val,test \
    --crop_size 64 \
    --min_area 8 \
    --iou_thr 0.10 \
    --overlap_thr 0.30 \
    --balance_train

run_step "07 Multi-NeurRL event sequence dataset" "$MULTI_EVENT" \
  python 07_make_multineurrl_event_sequence_dataset.py --splits train,val,test

run_step "08 Figure event verifier training" "$VERIFIER_BEST" \
  python 08_train_figure_event_verifier.py --epochs "$VERIFIER_EPOCHS" --batch_size "$VERIFIER_BATCH"

run_step "09 Event-level correction" "$CORR_CSV" \
  python 09_apply_event_verifier_correction.py --splits train,val,test --thresholds "$THRESHOLDS"

run_step "10 Baseline vs verifier comparison" "$FINAL_CSV" \
  python 10_compare_baseline_vs_verifier.py --select_split val --select_metric pixel_f1

# Visualization uses the selected/default correction threshold. Rerun manually if selected threshold changes.
run_step "11 Correction visualization examples" "$VIS_DIR" \
  python 11_visualize_correction_examples.py --split test --threshold "$CORRECTION_THR" --top_k 8

echo
echo "============================================================"
echo "[DONE] Full pipeline finished."
echo "Final comparison: $FINAL_CSV"
echo "============================================================"
