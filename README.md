# MHW-NeurRL：面向海洋热浪预报的事件级神经符号验证

本项目是一个面向 GitHub 展示的精简版 MHW-NeurRL pipeline，用于南海海洋热浪（Marine Heatwave, MHW）掩膜预报、候选事件提取、事件级 symbolic rule verifier，以及 NeurRL-style 空间规则可解释性展示。

`code/` 根目录下只保留 01–07 这些主流程入口脚本。开发过程中产生的中间实验脚本没有删除，已经归档到 `code/legacy/`，主流程 wrapper 会调用这些已跑通的 legacy 脚本，不改变实验逻辑。

## 项目做什么

MHW-NeurRL 的主要流程包括：

1. 基于 NOAA OISST 日尺度 SST 数据构建严格 Hobday-style MHW 标签；
2. 构建 lead=5 天的 MHW mask forecasting 数据集；
3. 训练 persistence-aware multichannel U-Net 预报模型；
4. 将 U-Net 预测 mask 转换成候选 MHW connected components；
5. 学习 event-level symbolic rules，用于保守地删除 false-positive 候选事件；
6. 构建 region-level / NeurRL-style rules，用于空间可解释性展示；
7. 输出 compact removed-event visualization、event-level rule visualization 和 region-level rule visualization。

核心结论是：The main performance gain comes from the multichannel forecasting input. The symbolic rule verifier provides a small but interpretable correction on top of the strong multichannel U-Net baseline.

## 目录结构

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

推荐阅读和运行 `code/00_config.py` 以及 `code/01_build_labels.py` 到 `code/07_visualize_results.py`。旧的分散实验脚本保存在 `code/legacy/`，用于复现中间步骤和保留开发记录。GitHub 用户推荐使用 `code/01_build_labels.py` 到 `code/07_visualize_results.py` 这些主入口。

## 环境安装

```bash
pip install -r requirements.txt
```

主要依赖包括：`numpy`、`pandas`、`xarray`、`netCDF4`、`scipy`、`scikit-image`、`scikit-learn`、`matplotlib`、`tqdm`、`torch`、`torchvision` 和 `Pillow`。

## 路径配置

所有主脚本统一使用：

```text
code/00_config.py
```

默认项目根目录由仓库结构自动推断，也可以通过环境变量指定：

```bash
export MHWNEURRL_ROOT=/path/to/MHWNeurRL
```

本地数据默认放在：

```text
data/oisst_scs_1982_2023.nc
```

`data/`、`outputs/`、模型权重、`.nc`、`.npy`、`.npz`、`.pt` 等大文件不会上传 GitHub，已由 `.gitignore` 排除。

## 数据说明

原始数据不包含在 GitHub 仓库中。

```text
数据来源：NOAA OISST daily SST
研究区域：South China Sea
经度范围：100–125E
纬度范围：0–25N
时间范围：1982–2023
网格大小：101 x 101
```

MHW 标签采用严格 Hobday-style 定义：

```text
climatology: 1982–2011
percentile: 90th
minimum duration: 5 days
maximum gap: 2 days
```

## GitHub 主流程

在项目根目录运行：

```bash
python code/01_build_labels.py --all
python code/02_build_forecast_dataset.py --mode multichannel
python code/03_train_eval_unet.py --model multichannel --train --eval
python code/04_build_event_dataset.py --source multichannel
python code/05_learn_rules.py --type all
python code/06_apply_rule_verifier.py --all
python code/07_visualize_results.py --type all
```

### 主流程阶段

| 阶段 | 主脚本 | 功能 | 主要输出 |
|---|---|---|---|
| 1 | `01_build_labels.py` | 构建、可视化并验证 MHW 标签 | `outputs/01_mhw_labels/`, `outputs/02_label_visualization/`, `outputs/02_label_validation/` |
| 2 | `02_build_forecast_dataset.py` | 构建 SSTA-only 或 multichannel forecast dataset | `outputs/03_forecast_dataset_h10_l5/`, `outputs/03b_forecast_dataset_multichannel_h10_l5/` |
| 3 | `03_train_eval_unet.py` | 训练和评估 SSTA-only / multichannel U-Net | `outputs/04_unet_baseline_h10_l5/`, `outputs/04c_unet_multichannel_h10_l5/` |
| 4 | `04_build_event_dataset.py` | 从预测 mask 提取候选 MHW 事件 | `outputs/06c_neurrl_event_dataset_from_multichannel_h10_l5/` |
| 5 | `05_learn_rules.py` | 学习 event-level symbolic rules 和 region-level rules | `outputs/20_event_rule_learning/`, `outputs/26_region_rule_learning/` |
| 6 | `06_apply_rule_verifier.py` | 应用 symbolic rule verifier 并比较结果 | `outputs/20_event_rule_learning/rule_verifier_correction/`, `outputs/23_rule_verifier_comparison/` |
| 7 | `07_visualize_results.py` | 生成 removed-event、event-rule 和 region-rule 可视化 | `outputs/24b_removed_rule_event_compact_visualization/`, `outputs/25b_event_symbolic_rule_visualization/`, `outputs/28_region_rule_visualization/` |

更详细的命令、输入输出和 legacy 脚本对应关系见：

```text
code/README_commands.md
code/legacy/README_legacy.md
```

## 可选 baseline

SSTA-only U-Net 作为对照 baseline 保留：

```bash
python code/02_build_forecast_dataset.py --mode ssta
python code/03_train_eval_unet.py --model ssta --train --eval
```

主模型是 multichannel U-Net。

## 主要结果

### Forecasting baseline comparison

| Model | Precision | Recall | F1 | IoU/CSI | Accuracy |
|---|---:|---:|---:|---:|---:|
| Persistence baseline | 0.7483 | 0.7472 | 0.7478 | 0.5972 | 0.8971 |
| SSTA-only U-Net | 0.5817 | 0.8251 | 0.6824 | 0.5179 | 0.8431 |
| Multichannel U-Net | 0.733330 | 0.815710 | 0.772329 | 0.629101 | 0.901778 |

Persistence baseline 和 SSTA-only U-Net 是历史对比/参考结果；Multichannel U-Net 使用本次从头跑通 01–07 主流程后的最新版 test 结果。The main performance gain comes from the multichannel forecasting input. The symbolic rule verifier provides a small but interpretable correction on top of the strong multichannel U-Net baseline.

### Symbolic rule verifier comparison

| Model | Precision | Recall | F1 | IoU/CSI | Accuracy |
|---|---:|---:|---:|---:|---:|
| Multichannel U-Net | 0.733330 | 0.815710 | 0.772329 | 0.629101 | 0.901778 |
| Multichannel U-Net + symbolic rule verifier | 0.734164 | 0.815551 | 0.772720 | 0.629621 | 0.902016 |

Delta：

```text
Precision: +0.000835
Recall:    -0.000159
F1:        +0.000391
IoU/CSI:   +0.000519
Accuracy:  +0.000238
```

The symbolic rule verifier provides a small pixel-level improvement on top of the multichannel U-Net. Its main contribution is not large segmentation improvement, but interpretable event-level false-positive filtering.

也就是说，symbolic rule verifier 在强 multichannel U-Net baseline 基础上只带来较小的像素级提升，其主要贡献是提供可解释的事件级误报过滤机制。

### Event-level rule deletion

```text
test candidate events: 12484
removed by symbolic rule: 91
correctly removed invalid events: 78
wrongly removed valid events: 13
event-level removal precision: 85.7%
removed-event ratio: 0.729%
```

The rule verifier removed only 0.729% of candidate events, but 85.7% of the removed events were invalid candidates. This indicates that the rule verifier behaves as a conservative high-precision, low-recall false-positive filter.

## 规则解释

### Event-level symbolic rules

Event-level rules 用于实际 correction / verifier。主要规则来自候选事件内部的 recent threshold support：

示例：

```text
IF recent threshold_gap inside candidate <= 0
THEN remove candidate as invalid
```

This rule removes candidate MHW events whose recent threshold support is weak.

### Region-level / NeurRL-style rules

Region-level / NeurRL-style rules 用于空间可解释性展示，不作为主要 correction 指标。它们把 event patch 划分成 4 x 4 区域，并学习空间区域上的规则。

示例：

```text
IF EXCEED90_R3C3_ACTIVE
THEN valid candidate

IF MHW_R3C3_ACTIVE
THEN valid candidate
```

`R3C3` denotes a spatial region in the 4 x 4 patch grid. Region-level rules are mainly used to visualize which physical channels and spatial regions support the candidate-event decision.

Event-level rules are used for conservative correction, while region-level rules are used mainly for NeurRL-style spatial interpretability.

## 重要可视化输出

```text
outputs/24b_removed_rule_event_compact_visualization/
outputs/25b_event_symbolic_rule_visualization/
outputs/28_region_rule_visualization/
```

其中：

- `24b_removed_rule_event_compact_visualization/`：展示规则删除前后效果，包括 target MHW、original prediction、removed component、corrected prediction、TP/FP/FN overlay；
- `25b_event_symbolic_rule_visualization/`：展示 event-level symbolic rules 的规则指标、触发案例和关键 atom 分布；
- `28_region_rule_visualization/`：展示 region-level / NeurRL-style 规则图，在 patch 上用 region 框标出规则触发位置。

## Legacy scripts

旧脚本没有删除，统一放在：

```text
code/legacy/
```

这些脚本是开发过程中已经跑通的中间实验脚本。新的 01–07 主流程脚本只是 wrapper，用于让 GitHub 用户更容易理解和运行项目。

对应关系见：

```text
code/legacy/README_legacy.md
```

## GitHub 上传注意事项

不要把以下文件加入 Git：

```text
data/
outputs/
*.nc
*.npy
*.npz
*.pt
*.zip
```

上传前建议检查：

```bash
git status
git status --ignored | head -100
```

推荐提交命令：

```bash
git add -A README.md .gitignore requirements.txt run_full_pipeline.sh code docs
git commit -m "Simplify MHW-NeurRL code entrypoints"
git push
```
