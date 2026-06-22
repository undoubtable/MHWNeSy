# Legacy and Intermediate Scripts

This folder contains legacy and intermediate experimental scripts from the
development process.

The recommended GitHub-facing pipeline is defined by:

```text
code/00_config.py
code/01_build_labels.py
code/02_build_forecast_dataset.py
code/03_train_eval_unet.py
code/04_build_event_dataset.py
code/05_learn_rules.py
code/06_apply_rule_verifier.py
code/07_visualize_results.py
```

The root-level scripts are lightweight wrappers that call these legacy scripts
without changing the experiment logic.

## Mapping to the GitHub-facing Pipeline

```text
01_make_mhw_labels.py
02_visualize_mhw_labels.py
02b_validate_mhw_labels.py
    -> 01_build_labels.py

03_make_forecast_dataset.py
03b_make_forecast_dataset_multichannel.py
    -> 02_build_forecast_dataset.py

04_train_unet_baseline.py
05_eval_unet_baseline.py
04c_train_unet_multichannel.py
05c_eval_unet_multichannel.py
    -> 03_train_eval_unet.py

06_make_figure_neurrl_event_dataset.py
07_make_multineurrl_event_sequence_dataset.py
06c_make_figure_event_dataset_multichannel.py
07c_make_multineurrl_event_sequence_multichannel.py
    -> 04_build_event_dataset.py

20_build_event_rule_table.py
21_learn_event_rules.py
26_build_region_rule_atoms.py
27_learn_region_rules.py
    -> 05_learn_rules.py

22_apply_event_rule_verifier.py
23_compare_multichannel_unet_vs_rule_verifier.py
    -> 06_apply_rule_verifier.py

24b_visualize_removed_rule_events_compact.py
25b_visualize_event_symbolic_rules.py
28_visualize_region_rules.py
    -> 07_visualize_results.py
```

Additional diagnostic and development scripts are kept here for reproducibility:

```text
01_make_mhw_labels_fast_approx.py
02c_literature_comparison.py
02d_fix_literature_comparison_plot.py
08_train_figure_event_verifier.py
08c_train_figure_event_verifier_multichannel.py
09_apply_event_verifier_correction.py
09c_apply_event_verifier_correction_multichannel.py
10_compare_baseline_vs_verifier.py
10c_compare_multichannel_baseline_vs_verifier.py
11_visualize_correction_examples.py
12_eval_persistence_baseline.py
13_unet_threshold_sweep.py
14_compare_forecast_baselines.py
15_visualize_forecast_predictions.py
16_plot_forecast_area_timeseries.py
17_compute_extended_forecast_metrics.py
18_visualize_constructed_datasets.py
19_visualize_event_positive_negative_examples.py
24_visualize_removed_rule_events.py
25_visualize_symbolic_rules.py
```
