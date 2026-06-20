# MHW-NeurRL pipeline commands

Project root:

```bash
cd /ybz/ybz/2026/MHWNeurRL
mkdir -p code
# put all .py files into /ybz/ybz/2026/MHWNeurRL/code
cd code
```

Raw file expected at:

```bash
/ybz/ybz/2026/MHWNeurRL/data/oisst_scs_1982_2023.nc
```

## 1. Make MHW labels

```bash
python 01_make_mhw_labels.py \
  --raw_nc /ybz/ybz/2026/MHWNeurRL/data/oisst_scs_1982_2023.nc \
  --clim_start 1982 --clim_end 2011 \
  --min_duration 5 --max_gap 2
```

Output:

```bash
/ybz/ybz/2026/MHWNeurRL/data/processed/mhw_labels_1982_2023.nc
```

## 2. Visualize labels

```bash
python 02_visualize_mhw_labels.py --date 2023-07-01 --year 2023
```

Output:

```bash
/ybz/ybz/2026/MHWNeurRL/figures/02_labels/
```

## 3. Make forecast dataset

```bash
python 03_make_forecast_dataset.py --history 10 --lead 5 --stride 1
```

Output:

```bash
/ybz/ybz/2026/MHWNeurRL/data/forecast_h10_l5/
```

## 4. Train U-Net baseline

```bash
python 04_train_unet_baseline.py \
  --epochs 30 \
  --batch_size 32 \
  --history 10
```

Output:

```bash
/ybz/ybz/2026/MHWNeurRL/runs/04_unet_baseline_h10_l5/best_model.pt
/ybz/ybz/2026/MHWNeurRL/runs/04_unet_baseline_h10_l5/train_log.csv
```

## 5. Evaluate baseline and save predictions

```bash
python 05_eval_unet_baseline.py \
  --splits train,val,test \
  --batch_size 64
```

Output:

```bash
/ybz/ybz/2026/MHWNeurRL/runs/04_unet_baseline_h10_l5/eval_metrics.csv
/ybz/ybz/2026/MHWNeurRL/runs/04_unet_baseline_h10_l5/pred_prob_test.npy
/ybz/ybz/2026/MHWNeurRL/runs/04_unet_baseline_h10_l5/pred_mask_test.npy
```

## 6. Make Figure-NeurRL event dataset

```bash
python 06_make_figure_neurrl_event_dataset.py \
  --splits train,val,test \
  --crop_size 64 \
  --min_area 8 \
  --iou_thr 0.10 \
  --overlap_thr 0.30 \
  --balance_train
```

Output:

```bash
/ybz/ybz/2026/MHWNeurRL/data/event_h10_l5/figure_event_train.npz
/ybz/ybz/2026/MHWNeurRL/data/event_h10_l5/figure_event_val.npz
/ybz/ybz/2026/MHWNeurRL/data/event_h10_l5/figure_event_test.npz
```

## 7. Make Multi-NeurRL event sequence dataset

```bash
python 07_make_multineurrl_event_sequence_dataset.py --splits train,val,test
```

Output:

```bash
/ybz/ybz/2026/MHWNeurRL/data/event_h10_l5/multi_event_train.npz
/ybz/ybz/2026/MHWNeurRL/data/event_h10_l5/multi_event_val.npz
/ybz/ybz/2026/MHWNeurRL/data/event_h10_l5/multi_event_test.npz
```
