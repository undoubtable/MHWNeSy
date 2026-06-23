# Test Experiments: Hierarchical Point-to-Region MHW-NeurRL

This folder contains experimental code for the next-stage MHW-NeurRL framework.

The goal is to test a hierarchical neuro-symbolic marine heatwave forecasting pipeline:

1. Point-wise MHW forecasting
2. Point-level temporal rule learning
3. Region-level event construction
4. Region-level spatial rule learning
5. Intensity-level rule learning
6. Neuro-symbolic correction and explanation

These scripts are experimental and are not part of the stable 01-07 pipeline yet.

## Planned scripts

- 00_test_config.py: shared paths and experiment settings
- 01_build_point_dataset.py: build point-wise sequence dataset
- 02_train_point_lstm.py: train point-wise LSTM/GRU/TCN baseline
- 03_eval_point_lstm.py: evaluate point-wise forecasting
- 04_learn_point_rules.py: learn point-level temporal rules
- 05_build_region_from_point.py: aggregate point predictions into candidate MHW regions
- 06_learn_region_rules.py: learn region-level spatial rules
- 07_learn_intensity_rules.py: learn MHW intensity rules

## Outputs

Large files should be saved under:

test/outputs/
test/logs/

These folders should not be uploaded to GitHub.
