# MHW-NeurRL Compact Pipeline Commands

The GitHub-facing pipeline is intentionally compact. Use the root-level scripts in `code/`; detailed intermediate scripts are archived under `code/legacy/`.

Run from the repository root:

```bash
cd /path/to/MHWNeurRL
export MHWNEURRL_ROOT=$PWD
```

## 0. Configuration

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

## 1. Build Labels

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

## 2. Build Forecast Dataset

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

## 3. Train and Evaluate U-Net

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

## 4. Build Event Dataset

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

## 5. Learn Rules

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

## 6. Apply Rule Verifier

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

## 7. Visualize Results

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

## Main Run Order

```bash
python code/01_build_labels.py --all
python code/02_build_forecast_dataset.py --mode multichannel
python code/03_train_eval_unet.py --model multichannel --train --eval
python code/04_build_event_dataset.py --source multichannel
python code/05_learn_rules.py --type all
python code/06_apply_rule_verifier.py --all
python code/07_visualize_results.py --type all
```

## Legacy Scripts

Legacy scripts are archived under:

```text
code/legacy/
```

See:

```text
code/legacy/README_legacy.md
```
