# Pipeline Overview

## Project Task

MHW-NeurRL studies South China Sea marine heatwave forecasting and interpretable event-level verification. The forecasting target is a 5-day lead MHW mask. The verification target is whether each predicted connected component is a valid candidate MHW event or a false positive.

The full GitHub-facing pipeline has been verified from Stage 1 to Stage 7. The main performance gain comes from the multichannel U-Net input design. The event-level symbolic rule verifier gives a small but positive pixel-level improvement and removes false-positive candidate events with 85.7% event-level removal precision. Region-level rules provide NeurRL-style spatial interpretability.

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
| Multichannel U-Net | 0.733330 | 0.815710 | 0.772329 | 0.629101 | 0.901778 |

Persistence baseline and SSTA-only U-Net are retained as historical/reference baselines. The multichannel U-Net row is the latest result from the full 01-07 rerun. The main performance gain comes from the multichannel forecasting input.

### Symbolic Rule Verifier

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

## Rule Module Positioning

Event-level symbolic rules are learned from candidate-event summary features and are used for conservative correction. The main rule is:

```text
IF recent threshold_gap inside candidate <= 0
THEN remove candidate as invalid
```

This rule removes candidate MHW events whose recent threshold support is weak.

Region-level / NeurRL-style rules are learned from a fixed 4 x 4 grid over each event patch. They provide spatial explanations such as `EXCEED90_R3C3_ACTIVE` or `MHW_R3C3_ACTIVE`. `R3C3` denotes a spatial region in the 4 x 4 patch grid. Region-level rules are mainly used to visualize which physical channels and spatial regions support the candidate-event decision.

Event-level rules are used for conservative correction, while region-level rules are used mainly for NeurRL-style spatial interpretability.

## Two Rule Figure Types

Event-level rule figures:

```text
outputs/25b_event_symbolic_rule_visualization/
```

These show event-level symbolic rule metrics, triggering cases, and key atom distributions.

Region-level rule figures:

```text
outputs/28_region_rule_visualization/
```

These show top spatial rules and representative patches. Yellow boxes mark the 4 x 4 grid cells used by each rule.

Removed-event compact figures:

```text
outputs/24b_removed_rule_event_compact_visualization/
```

These show the rule deletion effect, including target MHW, original prediction, removed component, corrected prediction, and TP/FP/FN overlay.
