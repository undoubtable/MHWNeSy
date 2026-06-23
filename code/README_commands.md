# MHW-NeurRL Commands

The GitHub-facing pipeline is intentionally compact. Use the root-level scripts in `code/`; detailed intermediate scripts and old experiment scripts are archived under `code/legacy/`.

Run from the repository root:

```bash
cd /path/to/MHWNeurRL
export MHWNEURRL_ROOT=$PWD
pip install -r requirements.txt
```

## 1. Environment

Script:

```text
code/00_config.py
```

Purpose:

```text
Defines ROOT, DATA_DIR, OUTPUT_DIR, CODE_DIR, and LEGACY_CODE_DIR.
Supports the MHWNEURRL_ROOT environment variable.
```

Inputs:

```text
data/oisst_scs_1982_2023.nc
```

Outputs:

```text
outputs/
```

## 2. Data Preparation

The raw NOAA OISST file is not included in the GitHub repository. Place it at:

```text
data/oisst_scs_1982_2023.nc
```

The strict Hobday-style label definition uses:

```text
climatology: 1982-2011
percentile: 90th
minimum duration: 5 days
maximum gap: 2 days
```

## 3. Main 01-07 Pipeline

Recommended full run:

```bash
python code/01_build_labels.py --all
python code/02_build_forecast_dataset.py --mode multichannel
python code/03_train_eval_unet.py --model multichannel --train --eval
python code/04_build_event_dataset.py --source multichannel
python code/05_learn_rules.py --type all
python code/06_apply_rule_verifier.py --all
python code/07_visualize_results.py --type all
```

The main model path is the multichannel U-Net. SSTA-only commands remain available as historical/control baselines.

## 4. Stage-by-Stage Commands

### Stage 1: Build Labels

Script:

```text
code/01_build_labels.py
```

Function:

```text
Build strict Hobday-style MHW labels, create basic label plots, and run label validation.
```

Inputs:

```text
data/oisst_scs_1982_2023.nc
```

Outputs:

```text
outputs/01_mhw_labels/
outputs/02_label_visualization/
outputs/02_label_validation/
```

Recommended command:

```bash
python code/01_build_labels.py --all
```

Options:

```bash
python code/01_build_labels.py --make
python code/01_build_labels.py --visualize
python code/01_build_labels.py --validate
python code/01_build_labels.py --all
```

Corresponding legacy scripts:

```text
legacy/01_make_mhw_labels.py
legacy/02_visualize_mhw_labels.py
legacy/02b_validate_mhw_labels.py
```

### Stage 2: Build Forecast Dataset

Script:

```text
code/02_build_forecast_dataset.py
```

Function:

```text
Build SSTA-only or multichannel lead-5 forecast datasets.
```

Inputs:

```text
outputs/01_mhw_labels/mhw_labels_strict_hobday_1982_2023.nc
```

Outputs:

```text
outputs/03_forecast_dataset_h10_l5/
outputs/03b_forecast_dataset_multichannel_h10_l5/
```

Recommended command:

```bash
python code/02_build_forecast_dataset.py --mode multichannel
```

Options:

```bash
python code/02_build_forecast_dataset.py --mode ssta
python code/02_build_forecast_dataset.py --mode multichannel
python code/02_build_forecast_dataset.py --mode all
```

Corresponding legacy scripts:

```text
legacy/03_make_forecast_dataset.py
legacy/03b_make_forecast_dataset_multichannel.py
```

### Stage 3: Train and Evaluate U-Net

Script:

```text
code/03_train_eval_unet.py
```

Function:

```text
Train and evaluate either the SSTA-only control U-Net or the main multichannel U-Net.
```

Inputs:

```text
outputs/03_forecast_dataset_h10_l5/
outputs/03b_forecast_dataset_multichannel_h10_l5/
```

Outputs:

```text
outputs/04_unet_baseline_h10_l5/
outputs/04c_unet_multichannel_h10_l5/
```

Recommended command:

```bash
python code/03_train_eval_unet.py --model multichannel --train --eval
```

Options:

```bash
python code/03_train_eval_unet.py --model ssta --train
python code/03_train_eval_unet.py --model ssta --eval
python code/03_train_eval_unet.py --model multichannel --train
python code/03_train_eval_unet.py --model multichannel --eval
python code/03_train_eval_unet.py --model multichannel --train --eval
```

Corresponding legacy scripts:

```text
legacy/04_train_unet_baseline.py
legacy/05_eval_unet_baseline.py
legacy/04c_train_unet_multichannel.py
legacy/05c_eval_unet_multichannel.py
```

### Stage 4: Build Event Dataset

Script:

```text
code/04_build_event_dataset.py
```

Function:

```text
Convert U-Net predicted masks into connected-component candidate event datasets.
The main path uses multichannel U-Net predictions.
```

Inputs:

```text
outputs/04c_unet_multichannel_h10_l5/pred_prob_{split}.npy
outputs/04c_unet_multichannel_h10_l5/selected_threshold.json
outputs/03b_forecast_dataset_multichannel_h10_l5/y_{split}.npy
```

Outputs:

```text
outputs/06c_neurrl_event_dataset_from_multichannel_h10_l5/figure_event_{split}.npz
outputs/06c_neurrl_event_dataset_from_multichannel_h10_l5/multi_event_{split}.npz
```

Recommended command:

```bash
python code/04_build_event_dataset.py --source multichannel
```

Options:

```bash
python code/04_build_event_dataset.py --source ssta
python code/04_build_event_dataset.py --source multichannel
python code/04_build_event_dataset.py --source all
```

Corresponding legacy scripts:

```text
legacy/06_make_figure_neurrl_event_dataset.py
legacy/07_make_multineurrl_event_sequence_dataset.py
legacy/06c_make_figure_event_dataset_multichannel.py
legacy/07c_make_multineurrl_event_sequence_multichannel.py
```

### Stage 5: Learn Rules

Script:

```text
code/05_learn_rules.py
```

Function:

```text
Learn event-level symbolic rules for correction and region-level / NeurRL-style rules for spatial interpretability.
```

Inputs:

```text
outputs/06c_neurrl_event_dataset_from_multichannel_h10_l5/multi_event_{split}.npz
outputs/06c_neurrl_event_dataset_from_multichannel_h10_l5/figure_event_{split}.npz
```

Outputs:

```text
outputs/20_event_rule_learning/
outputs/26_region_rule_learning/
```

Recommended command:

```bash
python code/05_learn_rules.py --type all
```

Options:

```bash
python code/05_learn_rules.py --type event
python code/05_learn_rules.py --type region
python code/05_learn_rules.py --type all
```

Corresponding legacy scripts:

```text
legacy/20_build_event_rule_table.py
legacy/21_learn_event_rules.py
legacy/26_build_region_rule_atoms.py
legacy/27_learn_region_rules.py
```

### Stage 6: Apply Rule Verifier

Script:

```text
code/06_apply_rule_verifier.py
```

Function:

```text
Apply learned invalid-event rules to delete selected predicted components, then compare against the multichannel U-Net baseline.
```

Inputs:

```text
outputs/20_event_rule_learning/event_rule_predictions_{split}.csv
outputs/04c_unet_multichannel_h10_l5/pred_mask_{split}.npy
```

Outputs:

```text
outputs/20_event_rule_learning/rule_verifier_correction/
outputs/23_rule_verifier_comparison/final_comparison.csv
```

Recommended command:

```bash
python code/06_apply_rule_verifier.py --all
```

Options:

```bash
python code/06_apply_rule_verifier.py --apply
python code/06_apply_rule_verifier.py --compare
python code/06_apply_rule_verifier.py --all
```

Corresponding legacy scripts:

```text
legacy/22_apply_event_rule_verifier.py
legacy/23_compare_multichannel_unet_vs_rule_verifier.py
```

### Stage 7: Visualize Results

Script:

```text
code/07_visualize_results.py
```

Function:

```text
Create compact removed-event figures, event-level symbolic rule figures, and region-level / NeurRL-style rule figures.
```

Inputs:

```text
outputs/20_event_rule_learning/
outputs/20_event_rule_learning/rule_verifier_correction/
outputs/26_region_rule_learning/
outputs/06c_neurrl_event_dataset_from_multichannel_h10_l5/
```

Outputs:

```text
outputs/24b_removed_rule_event_compact_visualization/
outputs/25b_event_symbolic_rule_visualization/
outputs/28_region_rule_visualization/
```

Recommended command:

```bash
python code/07_visualize_results.py --type all
```

Options:

```bash
python code/07_visualize_results.py --type removed
python code/07_visualize_results.py --type event_rules
python code/07_visualize_results.py --type region_rules
python code/07_visualize_results.py --type all
```

Corresponding legacy scripts:

```text
legacy/24b_visualize_removed_rule_events_compact.py
legacy/25b_visualize_event_symbolic_rules.py
legacy/28_visualize_region_rules.py
```

## 5. Main Outputs

```text
outputs/24b_removed_rule_event_compact_visualization/
```

Shows the rule deletion effect, including target MHW, original prediction, removed component, corrected prediction, and TP/FP/FN overlay.

```text
outputs/25b_event_symbolic_rule_visualization/
```

Shows event-level symbolic rule metrics, triggering cases, and key atom distributions.

```text
outputs/28_region_rule_visualization/
```

Shows region-level / NeurRL-style rule figures, with region boxes marking rule-triggered locations on the patch.

## 6. Latest Reproducible Results

### Forecasting baseline comparison

Persistence baseline and SSTA-only U-Net are retained as historical/reference baselines. The multichannel U-Net row is the latest result from the full 01-07 rerun.

| Model | Precision | Recall | F1 | IoU/CSI | Accuracy |
|---|---:|---:|---:|---:|---:|
| Persistence baseline | 0.7483 | 0.7472 | 0.7478 | 0.5972 | 0.8971 |
| SSTA-only U-Net | 0.5817 | 0.8251 | 0.6824 | 0.5179 | 0.8431 |
| Multichannel U-Net | 0.733330 | 0.815710 | 0.772329 | 0.629101 | 0.901778 |

The main performance gain comes from the multichannel forecasting input. The symbolic rule verifier provides a small but interpretable correction on top of the strong multichannel U-Net baseline.

### Symbolic rule verifier

| Model | Precision | Recall | F1 | IoU/CSI | Accuracy |
|---|---:|---:|---:|---:|---:|
| Multichannel U-Net | 0.733330 | 0.815710 | 0.772329 | 0.629101 | 0.901778 |
| Multichannel U-Net + symbolic rule verifier | 0.734164 | 0.815551 | 0.772720 | 0.629621 | 0.902016 |

Delta rule verifier - baseline:

```text
Precision = +0.000835
Recall    = -0.000159
F1        = +0.000391
IoU/CSI   = +0.000519
Accuracy  = +0.000238
```

The symbolic rule verifier provides a small pixel-level improvement on top of the multichannel U-Net. Its main contribution is not large segmentation improvement, but interpretable event-level false-positive filtering.

Event-level deletion statistics:

```text
test candidate events: 12484
removed by symbolic rule: 91
correctly removed invalid events: 78
wrongly removed valid events: 13
event-level removal precision: 85.7%
removed-event ratio: 0.729%
```

The rule verifier removed only 0.729% of candidate events, but 85.7% of the removed events were invalid candidates. This indicates that the rule verifier behaves as a conservative high-precision, low-recall false-positive filter.

## 7. Rule Interpretation

### Event-level symbolic rules

Event-level rules are used for actual correction / verifier behavior. The main rule is:

```text
IF recent threshold_gap inside candidate <= 0
THEN remove candidate as invalid
```

This rule removes candidate MHW events whose recent threshold support is weak.

### Region-level / NeurRL-style rules

Region-level / NeurRL-style rules are used for spatial interpretability, not as the main correction metric. Examples:

```text
IF EXCEED90_R3C3_ACTIVE THEN valid candidate
IF MHW_R3C3_ACTIVE THEN valid candidate
```

`R3C3` denotes a spatial region in the 4 x 4 patch grid. Region-level rules are mainly used to visualize which physical channels and spatial regions support the candidate-event decision.

Event-level rules are used for conservative correction, while region-level rules are used mainly for NeurRL-style spatial interpretability.

## 8. Legacy Scripts

Legacy scripts are archived under:

```text
code/legacy/
```

They are intermediate scripts and old experiment scripts from development. They are kept for reproducibility and traceability, but GitHub users should run the compact `code/01_build_labels.py` through `code/07_visualize_results.py` entrypoints.

See:

```text
code/legacy/README_legacy.md
```
