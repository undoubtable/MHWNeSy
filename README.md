# MHW-NeurRL：面向南海海洋热浪预报的神经符号增强 Pipeline

本项目面向南海海洋热浪（Marine Heatwave, MHW）预报任务，基于 OISST 日尺度 SST 数据构建严格 Hobday-style MHW 标签，训练 U-Net 像素级预报 baseline，并进一步将像素级预测结果转化为候选 MHW 事件，通过事件级 verifier / NeurRL 思路进行验证、修正与解释。

---

## 1. 技术路线

本研究首先基于 OISST 日尺度 SST 数据，按照严格 Hobday-style 定义构建南海 MHW 标签，并通过可视化、统计趋势与文献对比验证标签合理性。

随后构造 MHW 预报数据集，以过去 10 天 SSTA 作为输入，预测未来第 5 天的 MHW mask。

在此基础上训练 U-Net baseline，获得像素级 MHW 预报概率图与初始分割结果。

进一步对 U-Net 预测结果进行连通域分析，提取候选 MHW 事件，并构建 Figure-NeurRL 图像 patch 数据和 Multi-NeurRL 事件序列数据。

最后，利用事件级 verifier / NeurRL 思路学习候选事件质量与结构一致性规则，对 baseline 预测结果进行验证、修正与解释，实现 Baseline 与 Baseline + Verifier 的对比。

简化流程如下：

```text
OISST → Strict MHW Label → Forecast Dataset → U-Net Baseline
      → Event Candidate Extraction → Figure/Multi Event Dataset
      → Event Verifier → Corrected Prediction → Final Comparison
```

---

## 2. 项目结构

```text
MHWNeurRL/
├── code/
│   ├── 00_config.py
│   ├── 01_make_mhw_labels.py
│   ├── 01_make_mhw_labels_fast_approx.py
│   ├── 02_visualize_mhw_labels.py
│   ├── 02b_validate_mhw_labels.py
│   ├── 02c_literature_comparison.py
│   ├── 02d_fix_literature_comparison_plot.py
│   ├── 03_make_forecast_dataset.py
│   ├── 04_train_unet_baseline.py
│   ├── 05_eval_unet_baseline.py
│   ├── 06_make_figure_neurrl_event_dataset.py
│   ├── 07_make_multineurrl_event_sequence_dataset.py
│   ├── 08_train_figure_event_verifier.py
│   ├── 09_apply_event_verifier_correction.py
│   ├── 10_compare_baseline_vs_verifier.py
│   ├── 11_visualize_correction_examples.py
│   └── README_commands.md
├── data/
│   └── oisst_scs_1982_2023.nc
├── outputs/
├── README.md
├── .gitignore
└── run_full_pipeline.sh
```

其中：

```text
data/      存放原始 OISST 数据，不上传 GitHub
outputs/   存放所有中间结果、模型权重和图片，不上传 GitHub
code/      存放完整 pipeline 代码
```

---

## 3. 数据说明

本项目使用 South China Sea 区域的 OISST 日尺度 SST 数据：

```text
data/oisst_scs_1982_2023.nc
```

当前默认区域和时间范围为：

```text
时间：1982-01-01 至 2023-12-31
区域：约 100°E–125°E, 0°N–25°N
变量：sst
```

该数据文件较大，默认不包含在 GitHub 仓库中。运行前需要自行放置到 `data/` 目录下。

---

## 4. 环境依赖

建议使用 Python 3.10 或以上版本。主要依赖包括：

```bash
pip install numpy pandas xarray netCDF4 h5netcdf h5py scipy matplotlib tqdm scikit-image torch
```

如果使用 Conda，可根据服务器环境自行安装 PyTorch GPU 版本。

---

## 5. 配置文件

核心路径由 `code/00_config.py` 控制。默认情况下，项目根目录自动识别为 `code/` 的上一级目录。

也可以手动指定项目根目录：

```bash
export MHWNEURRL_ROOT=/path/to/MHWNeurRL
```

默认输入数据路径：

```text
data/oisst_scs_1982_2023.nc
```

默认输出目录：

```text
outputs/
```

---

## 6. 一键运行完整 Pipeline

在项目根目录下运行：

```bash
bash run_full_pipeline.sh
```

如果需要强制重新运行所有阶段：

```bash
FORCE=1 bash run_full_pipeline.sh
```

如果只想快速测试流程，可以减少训练轮数：

```bash
EPOCHS_UNET=3 EPOCHS_VERIFIER=3 bash run_full_pipeline.sh
```

---

## 7. Pipeline 各阶段说明

### Step 01：构建 Strict Hobday-style MHW 标签

脚本：

```text
code/01_make_mhw_labels.py
```

功能：

```text
根据 Hobday-style 定义构建 MHW 标签：
1. 1982–2011 作为气候态基准期；
2. 使用日历日 ±5 天窗口计算 90% 阈值；
3. 对 climatology 和 threshold 进行 31 天平滑；
4. 超过阈值且持续至少 5 天定义为 MHW；
5. 间隔小于等于 2 天的事件合并。
```

输出：

```text
outputs/01_mhw_labels/mhw_labels_strict_hobday_1982_2023.nc
```

主要变量：

```text
ssta
exceed90
mhw
clim_mean
thresh90
clim_mean_raw
thresh90_raw
```

---

### Step 02：MHW 标签可视化

脚本：

```text
code/02_visualize_mhw_labels.py
```

功能：

```text
可视化指定日期的 SSTA、exceed90 和 MHW mask；
同时绘制指定年份的 MHW area proxy 时间序列。
```

示例：

```bash
python code/02_visualize_mhw_labels.py --date 2023-07-01 --year 2023
```

输出：

```text
outputs/02_label_visualization/
```

---

### Step 02b：标签统计验证

脚本：

```text
code/02b_validate_mhw_labels.py
```

功能：

```text
统计 annual MHW days、MHW intensity、seasonal MHW days；
绘制长期趋势图、空间分布图和季节分布图。
```

输出：

```text
outputs/02_label_validation/
```

主要文件：

```text
annual_mhw_days.csv
annual_mhw_intensity.csv
seasonal_mhw_days.csv
label_summary.json
annual_mhw_days_trend.png
annual_mhw_intensity_trend.png
seasonal_mhw_days_map.png
mhw_days_mean_map_1982_2023.png
```

---

### Step 02c / 02d：文献趋势对比

脚本：

```text
code/02c_literature_comparison.py
code/02d_fix_literature_comparison_plot.py
```

功能：

```text
将本项目生成的南海夏季 JJA MHW days 趋势与已有文献中的趋势结果进行 benchmark-level comparison。
```

输出：

```text
outputs/02_label_validation/literature_comparison/
```

主要结果示例：

```text
This study JJA MHW days trend: 5.16 days/decade
Literature benchmark: approximately 3.0 days/decade
```

说明：该对比用于验证趋势方向和量级，不是对前人工作的完全复现。

---

### Step 03：构造 MHW 预报数据集

脚本：

```text
code/03_make_forecast_dataset.py
```

功能：

```text
构造监督学习样本：
输入 X：过去 10 天 SSTA
输出 y：未来第 5 天 MHW mask
```

默认设置：

```text
history = 10
lead = 5
stride = 1
```

输出：

```text
outputs/03_forecast_dataset_h10_l5/
```

主要文件：

```text
X_train.npy
y_train.npy
target_dates_train.npy
X_val.npy
y_val.npy
target_dates_val.npy
X_test.npy
y_test.npy
target_dates_test.npy
```

默认时间切分：

```text
train: 1982–2015
val:   2016–2018
test:  2019–2023
```

---

### Step 04：训练 U-Net Baseline

脚本：

```text
code/04_train_unet_baseline.py
```

功能：

```text
训练 U-Net 模型，输入过去 10 天 SSTA，预测未来第 5 天 MHW mask。
```

示例：

```bash
python code/04_train_unet_baseline.py --epochs 30 --batch_size 32 --history 10
```

输出：

```text
outputs/04_unet_baseline_h10_l5/
```

主要文件：

```text
best_model.pt
train_log.csv
config.json
```

---

### Step 05：评估 U-Net Baseline

脚本：

```text
code/05_eval_unet_baseline.py
```

功能：

```text
使用 best_model.pt 对 train / val / test 进行预测和评估。
```

示例：

```bash
python code/05_eval_unet_baseline.py --splits train,val,test --batch_size 64
```

输出：

```text
outputs/04_unet_baseline_h10_l5/
```

主要文件：

```text
pred_prob_train.npy
pred_mask_train.npy
pred_prob_val.npy
pred_mask_val.npy
pred_prob_test.npy
pred_mask_test.npy
eval_metrics.csv
```

当前第一版 baseline 测试集结果示例：

```text
Test Precision = 0.5017
Test Recall    = 0.9129
Test F1        = 0.6476
Test IoU       = 0.4788
```

---

### Step 06：构造 Figure-NeurRL 候选事件 Patch 数据

脚本：

```text
code/06_make_figure_neurrl_event_dataset.py
```

功能：

```text
对 U-Net 预测 mask 进行连通域分析；
将像素级预测结果转换为候选 MHW 事件；
为每个候选事件裁剪局部 patch；
根据与真实 MHW mask 的重叠程度标注 valid / invalid。
```

示例：

```bash
python code/06_make_figure_neurrl_event_dataset.py \
  --splits train,val,test \
  --crop_size 64 \
  --min_area 8 \
  --iou_thr 0.10 \
  --overlap_thr 0.30 \
  --balance_train
```

输出：

```text
outputs/06_neurrl_event_dataset_h10_l5/
```

主要文件：

```text
figure_event_train.npz
figure_event_val.npz
figure_event_test.npz
```

数据格式：

```text
X:       [N, 11, 64, 64]
y_valid: [N]
meta:    [N, 5]
```

其中：

```text
前 10 个通道：过去 10 天 SSTA patch
最后 1 个通道：预测候选事件 mask
y_valid = 1：合理候选事件
y_valid = 0：误报或低质量候选事件
```

---

### Step 07：构造 Multi-NeurRL 事件序列数据

脚本：

```text
code/07_make_multineurrl_event_sequence_dataset.py
```

功能：

```text
将 Figure-NeurRL patch 数据转换为事件级多变量时间序列数据。
```

示例：

```bash
python code/07_make_multineurrl_event_sequence_dataset.py --splits train,val,test
```

输出：

```text
outputs/06_neurrl_event_dataset_h10_l5/
```

主要文件：

```text
multi_event_train.npz
multi_event_val.npz
multi_event_test.npz
```

数据格式：

```text
X_seq: [N, 5, 10]
y:     [N]
```

含义：

```text
N 个候选事件；
5 个事件特征变量；
10 天时间序列。
```

---

### Step 08：训练事件级 Verifier

脚本：

```text
code/08_train_figure_event_verifier.py
```

功能：

```text
训练事件级候选质量分类器，判断 U-Net 预测出来的候选 MHW 事件是否合理。
```

说明：当前版本使用 CNN event verifier 完成 pipeline 闭环，后续可替换或扩展为 NeurRL rule verifier。

示例：

```bash
python code/08_train_figure_event_verifier.py --epochs 20 --batch_size 256
```

输出：

```text
outputs/06_neurrl_event_dataset_h10_l5/08_figure_event_verifier_cnn/
```

主要文件：

```text
best_model.pt
train_history.csv
metrics.csv
event_score_train.npy
event_pred_train.npy
event_score_val.npy
event_pred_val.npy
event_score_test.npy
event_pred_test.npy
```

当前第一版 verifier 测试集事件级结果示例：

```text
Test event Precision = 0.6281
Test event Recall    = 0.8707
Test event F1        = 0.7298
```

---

### Step 09：应用 Verifier 修正 U-Net 预测结果

脚本：

```text
code/09_apply_event_verifier_correction.py
```

功能：

```text
对 U-Net 预测 mask 进行连通域分析；
读取每个候选事件的 verifier score；
删除 score 低于阈值的预测连通域；
生成修正后的 MHW mask。
```

示例：

```bash
python code/09_apply_event_verifier_correction.py \
  --splits train,val,test \
  --thresholds 0.30,0.40,0.50,0.60,0.70,0.75,0.80,0.85,0.90
```

输出：

```text
outputs/04_unet_baseline_h10_l5/09_unet_plus_figure_verifier/
```

主要文件：

```text
corrected_mask_train_thr*.npy
corrected_mask_val_thr*.npy
corrected_mask_test_thr*.npy
correction_metrics.csv
correction_summary.json
```

---

### Step 10：比较 Baseline 与 Baseline + Verifier

脚本：

```text
code/10_compare_baseline_vs_verifier.py
```

功能：

```text
根据验证集指标选择最佳 verifier threshold；
比较 U-Net baseline 与 U-Net + event verifier correction 的 train / val / test 指标。
```

示例：

```bash
python code/10_compare_baseline_vs_verifier.py \
  --select_split val \
  --select_metric pixel_f1
```

输出：

```text
outputs/04_unet_baseline_h10_l5/10_compare_baseline_vs_verifier/
```

主要文件：

```text
final_comparison.csv
selected_threshold.json
```

当前第一版测试集结果示例：

```text
U-Net baseline:
F1  = 0.6476
IoU = 0.4788

U-Net + event verifier:
F1  = 0.6515
IoU = 0.4832
```

说明：

```text
事件级 verifier 删除部分低质量预测连通域，使 precision、F1 和 IoU 小幅提升，但 recall 略有下降。
```

---

### Step 11：可视化修正案例

脚本：

```text
code/11_visualize_correction_examples.py
```

功能：

```text
可视化 Ground Truth、U-Net baseline、Verifier correction 和 removed components。
```

示例：

```bash
python code/11_visualize_correction_examples.py \
  --split test \
  --threshold 0.75 \
  --top_k 8
```

输出：

```text
outputs/04_unet_baseline_h10_l5/11_correction_visualization/
```

---

## 8. 当前主要结果

当前第一版 pipeline 的核心结果如下：

```text
Pixel-level U-Net baseline, test:
Precision = 0.5017
Recall    = 0.9129
F1        = 0.6476
IoU       = 0.4788

U-Net + event verifier, test:
Precision = 0.5122
Recall    = 0.8951
F1        = 0.6515
IoU       = 0.4832
```

相比 baseline：

```text
Precision +0.0104
F1        +0.0040
IoU       +0.0044
Accuracy  +0.0074
Recall    -0.0178
```

这说明事件级 verifier 能够删除一部分低质量预测连通域，使预测结果更加保守和精确。

---

## 9. 当前版本说明

当前版本已经完成完整 pipeline 闭环：

```text
MHW label construction
→ Forecast baseline
→ Event candidate extraction
→ Event verifier
→ Prediction correction
→ Final comparison
```

需要注意的是：

```text
当前 08 阶段使用的是 CNN event verifier；
后续工作将进一步引入 NeurRL / rule learning 模块，
学习显式的事件质量规则、空间结构规则和时间演化规则。
```

因此，当前版本可以作为 MHW-NeurRL 的 baseline pipeline 和实验框架。

---

## 10. 不上传到 GitHub 的文件

以下文件或目录默认由 `.gitignore` 排除：

```text
data/
outputs/
logs/
*.nc
*.npy
*.npz
*.pt
*.pth
*.h5
*.hdf5
*.zip
__pycache__/
.vscode/
```

因此，GitHub 仓库只保留代码、README 和运行脚本，不包含原始数据、模型权重和输出结果。

---

## 11. 后续计划

后续可以在当前 pipeline 基础上继续扩展：

```text
1. 将 CNN event verifier 替换为 Figure-NeurRL rule verifier；
2. 使用 Multi-NeurRL 学习候选事件的时间演化规则；
3. 加入 point-wise temporal NeurRL，用于解释局地格点级 MHW 发生机制；
4. 引入更强 baseline，如 ConvLSTM、3D U-Net 或多源变量输入；
5. 完善事件级评估指标，包括事件检测率、误报率、持续时间和面积误差。
```
