# Pipeline Overview

## Project Task

MHW-NeurRL studies South China Sea marine heatwave forecasting and interpretable event-level verification. The forecasting target is a 5-day lead MHW mask. The verification target is whether each predicted connected component is a valid candidate MHW event or a false positive.

The main performance gain comes from the multichannel forecasting input. The symbolic rule verifier provides a small but interpretable correction on top of the strong multichannel U-Net baseline.

## GitHub-facing Data Flow

```text
code/01_build_labels.py
  NOAA OISST -> strict Hobday-style MHW labels

code/02_build_forecast_dataset.py
  labels -> multichannel lead-5 forecast dataset

code/03_train_eval_unet.py
  dataset -> multichannel U-Net predictions

code/04_build_event_dataset.py
  predicted masks -> connected-component event candidates

code/05_learn_rules.py
  event candidates -> event-level symbolic rules and region-level rules

code/06_apply_rule_verifier.py
  invalid-event rules -> corrected forecast masks and comparison tables

code/07_visualize_results.py
  candidates/rules/corrections -> compact explanation figures
```

The detailed intermediate scripts are archived under `code/legacy/`. The root-level scripts are wrappers that call those legacy scripts without changing the experiment logic.

## Stage Inputs and Outputs

| Stage | Main script | Input | Output |
|---|---|---|---|
| Labels | `01_build_labels.py` | `data/oisst_scs_1982_2023.nc` | `outputs/01_mhw_labels/`, `outputs/02_label_visualization/`, `outputs/02_label_validation/` |
| Forecast dataset | `02_build_forecast_dataset.py` | strict MHW labels | `outputs/03b_forecast_dataset_multichannel_h10_l5/` |
| U-Net | `03_train_eval_unet.py` | multichannel forecast dataset | `outputs/04c_unet_multichannel_h10_l5/` |
| Events | `04_build_event_dataset.py` | multichannel predictions and targets | `outputs/06c_neurrl_event_dataset_from_multichannel_h10_l5/` |
| Rules | `05_learn_rules.py` | event candidates | `outputs/20_event_rule_learning/`, `outputs/26_region_rule_learning/` |
| Correction | `06_apply_rule_verifier.py` | event rules and pred masks | `outputs/20_event_rule_learning/rule_verifier_correction/`, `outputs/23_rule_verifier_comparison/` |
| Visualization | `07_visualize_results.py` | rules and corrections | `outputs/24b_removed_rule_event_compact_visualization/`, `outputs/25b_event_symbolic_rule_visualization/`, `outputs/28_region_rule_visualization/` |

## Main Results

### Forecasting Baselines

| Model | Precision | Recall | F1 | IoU/CSI | Accuracy |
|---|---:|---:|---:|---:|---:|
| Persistence baseline | 0.7483 | 0.7472 | 0.7478 | 0.5972 | 0.8971 |
| SSTA-only U-Net | 0.5817 | 0.8251 | 0.6824 | 0.5179 | 0.8431 |
| Multichannel U-Net | 0.7433 | 0.8180 | 0.7789 | 0.6378 | 0.9051 |

The multichannel U-Net improves substantially over SSTA-only U-Net and beats persistence in F1, IoU/CSI, and accuracy.

### Symbolic Rule Verifier

| Model | Precision | Recall | F1 | IoU/CSI | Accuracy |
|---|---:|---:|---:|---:|---:|
| Multichannel U-Net | 0.743348 | 0.817988 | 0.778884 | 0.637846 | 0.905145 |
| Multichannel U-Net + symbolic rule verifier | 0.743730 | 0.817932 | 0.779068 | 0.638093 | 0.905252 |

The pixel-level improvement is small. The main value is interpretability: the rules remove a small set of false-positive event candidates with high precision.

Event-level deletion statistics:

```text
test candidate events: 12206
removed by symbolic rule: 96
correctly removed invalid events: 82
wrongly removed valid events: 14
event-level removal precision: 85.4%
removed-event ratio: 0.786%
```

## Rule Module Positioning

Event-level symbolic rules are learned from candidate-event summary features, such as recent threshold support, historical MHW days, exceedance days, and component area behavior. They are used for conservative correction.

Region-level / NeurRL-style rules are learned from a fixed 4 x 4 grid over each event patch. They provide spatial explanations such as `TGAP_R2C1_LOW` or `EXCEED90_R3C3_ACTIVE`. They are mainly for spatial interpretability, not the main correction mechanism.

## Two Rule Figure Types

Event-level rule figures:

```text
outputs/25b_event_symbolic_rule_visualization/
```

These show a triggering candidate event, its threshold_gap, target MHW, original prediction, candidate component, TP/FP/FN overlay, and actual atom values.

Region-level rule figures:

```text
outputs/28_region_rule_visualization/
```

These show top spatial rules and representative patches. Yellow boxes mark the 4 x 4 grid cells used by each rule.
