# MHW-NeurRL: Event-level Neurosymbolic Verification for Marine Heatwave Forecasting

This repository contains a compact GitHub-facing pipeline for South China Sea marine heatwave (MHW) mask forecasting and event-level neurosymbolic verification.

The main scripts in `code/` are intentionally small wrappers. They call archived, already-tested experiment scripts under `code/legacy/` without changing the experiment logic.

## What This Project Does

MHW-NeurRL:

1. Builds strict Hobday-style MHW labels from NOAA OISST daily SST.
2. Builds 5-day lead MHW forecasting datasets.
3. Trains a persistence-aware multichannel U-Net forecast model.
4. Converts predicted masks into candidate MHW events.
5. Learns event-level symbolic rules for conservative false-positive filtering.
6. Builds region-level / NeurRL-style rules for spatial interpretability.
7. Produces compact visualizations for removed events and learned rules.

The main performance gain comes from the multichannel forecasting input. The symbolic rule verifier provides a small but interpretable event-level correction.

## Repository Layout

```text
MHWNeurRL/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── pipeline_overview.md
└── code/
    ├── 00_config.py
    ├── 01_build_labels.py
    ├── 02_build_forecast_dataset.py
    ├── 03_train_eval_unet.py
    ├── 04_build_event_dataset.py
    ├── 05_learn_rules.py
    ├── 06_apply_rule_verifier.py
    ├── 07_visualize_results.py
    ├── README_commands.md
    └── legacy/
        ├── README_legacy.md
        └── ... intermediate experiment scripts
```

Detailed intermediate scripts are archived under `code/legacy/`. The recommended GitHub-facing pipeline is `code/00_config.py` and `code/01_build_labels.py` through `code/07_visualize_results.py`.

## Environment

```bash
pip install -r requirements.txt
```

## Path Configuration

All scripts use `code/00_config.py`. By default, the project root is inferred from the repository layout. You can override it with:

```bash
export MHWNEURRL_ROOT=/path/to/MHWNeurRL
```

Expected local data path:

```text
data/oisst_scs_1982_2023.nc
```

`data/`, `outputs/`, model weights, NumPy arrays, NetCDF files, and generated figures are ignored by Git.

## Data

Raw data are not uploaded to GitHub.

```text
Data source: NOAA OISST daily SST
Region: South China Sea
lon: 100-125E
lat: 0-25N
period: 1982-2023
grid: 101 x 101
```

MHW label definition:

```text
climatology: 1982-2011
percentile: 90th
minimum duration: 5 days
maximum gap: 2 days
```

## Main GitHub-facing Pipeline

Run from the repository root:

```bash
python code/01_build_labels.py --all
python code/02_build_forecast_dataset.py --mode multichannel
python code/03_train_eval_unet.py --model multichannel --train --eval
python code/04_build_event_dataset.py --source multichannel
python code/05_learn_rules.py --type all
python code/06_apply_rule_verifier.py --all
python code/07_visualize_results.py --type all
```

### Stage Summary

| Stage | Main script | Purpose | Main outputs |
|---|---|---|---|
| 1 | `01_build_labels.py` | Build, visualize, and validate MHW labels | `outputs/01_mhw_labels/`, `outputs/02_label_visualization/`, `outputs/02_label_validation/` |
| 2 | `02_build_forecast_dataset.py` | Build SSTA-only and/or multichannel forecast datasets | `outputs/03_forecast_dataset_h10_l5/`, `outputs/03b_forecast_dataset_multichannel_h10_l5/` |
| 3 | `03_train_eval_unet.py` | Train/evaluate SSTA-only or multichannel U-Net | `outputs/04_unet_baseline_h10_l5/`, `outputs/04c_unet_multichannel_h10_l5/` |
| 4 | `04_build_event_dataset.py` | Build candidate event datasets from predicted masks | `outputs/06c_neurrl_event_dataset_from_multichannel_h10_l5/` |
| 5 | `05_learn_rules.py` | Learn event-level and region-level rules | `outputs/20_event_rule_learning/`, `outputs/26_region_rule_learning/` |
| 6 | `06_apply_rule_verifier.py` | Apply symbolic rule verifier and compare metrics | `outputs/20_event_rule_learning/rule_verifier_correction/`, `outputs/23_rule_verifier_comparison/` |
| 7 | `07_visualize_results.py` | Visualize removed events and learned rules | `outputs/24b_removed_rule_event_compact_visualization/`, `outputs/25b_event_symbolic_rule_visualization/`, `outputs/28_region_rule_visualization/` |

See [code/README_commands.md](code/README_commands.md) for command options, inputs, outputs, and legacy-script mapping.

## Optional Baselines

The SSTA-only U-Net remains available as a control:

```bash
python code/02_build_forecast_dataset.py --mode ssta
python code/03_train_eval_unet.py --model ssta --train --eval
```

The multichannel U-Net is the main forecasting model.

## Main Results

### Forecasting Baseline Comparison

| Model | Precision | Recall | F1 | IoU/CSI | Accuracy |
|---|---:|---:|---:|---:|---:|
| Persistence baseline | 0.7483 | 0.7472 | 0.7478 | 0.5972 | 0.8971 |
| SSTA-only U-Net | 0.5817 | 0.8251 | 0.6824 | 0.5179 | 0.8431 |
| Multichannel U-Net | 0.7433 | 0.8180 | 0.7789 | 0.6378 | 0.9051 |

The multichannel U-Net substantially improves over the SSTA-only U-Net and also outperforms the persistence baseline in F1, IoU/CSI, and accuracy.

### Symbolic Rule Verifier Comparison

| Model | Precision | Recall | F1 | IoU/CSI | Accuracy |
|---|---:|---:|---:|---:|---:|
| Multichannel U-Net | 0.743348 | 0.817988 | 0.778884 | 0.637846 | 0.905145 |
| Multichannel U-Net + symbolic rule verifier | 0.743730 | 0.817932 | 0.779068 | 0.638093 | 0.905252 |

Delta:

```text
Precision: +0.000382
Recall:    -0.000056
F1:        +0.000184
IoU/CSI:   +0.000247
Accuracy:  +0.000108
```

The symbolic rule verifier brings only a small pixel-level improvement, but it provides an interpretable event-level false-positive filtering mechanism.

### Event-level Rule Deletion

```text
test candidate events: 12206
removed by symbolic rule: 96
correctly removed invalid events: 82
wrongly removed valid events: 14
event-level removal precision: 85.4%
removed-event ratio: 0.786%
```

The rule verifier behaves as a conservative high-precision, low-recall false-positive filter.

## Rule Interpretation

Event-level symbolic rules are used for conservative correction.

Example:

```text
IF recent threshold_gap inside candidate <= 0
THEN remove candidate as invalid
```

Region-level / NeurRL-style rules are used mainly for spatial interpretability.

Example:

```text
IF EXCEED90_R3C3_ACTIVE
THEN valid candidate
```

`R3C3` denotes a spatial region in the 4 x 4 patch grid.

## Legacy Scripts

The legacy scripts are not deleted. They are archived under:

```text
code/legacy/
```

See `code/legacy/README_legacy.md` for the mapping from old intermediate scripts to the new compact entrypoints.

## GitHub Notes

Do not add generated data or outputs to Git:

```text
data/
outputs/
*.nc
*.npy
*.npz
*.pt
*.zip
```

Recommended commit commands:

```bash
git status
git add README.md .gitignore requirements.txt code docs
git commit -m "Simplify MHW-NeurRL code entrypoints"
git push
```
