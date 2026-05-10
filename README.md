# CausalGait 可复现实验包

本文件夹用于让其他人复现论文修回稿中的主要实验结果。

## 目录结构

```text
可复现/
  code/                 实验源码
  data/SEMG_DB1/        UCI Lower Limb EMG 数据集副本
  expected_results/     本次修回中已经生成的预期结果
  outputs/              重新运行脚本后生成，初始不存在或为空
```

## 环境安装

推荐使用 Python 3.10 或更新版本。Windows PowerShell 中执行：

```powershell
cd "C:\Users\ali\Desktop\CausalGaint\可复现"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\code\requirements.txt
```

如果机器有 NVIDIA GPU，请安装与你 CUDA 版本匹配的 PyTorch。没有 GPU 也可以运行，但完整实验会慢很多。

## 快速验证

快速验证只跑 1 个 subject、1 个 epoch，用来确认环境和数据路径正确：

```powershell
.\run_smoke_test.ps1
```

输出目录：

```text
outputs/smoke/
```

## 复现主实验

主实验对应论文中的 leakage-aware 四通道 Transformer 结果：

```powershell
.\run_main_experiment.ps1
```

主要输出：

```text
outputs/artifacts_enhanced_publishable_reproduced/metrics_summary.csv
outputs/artifacts_enhanced_publishable_reproduced/metrics_by_subject.csv
outputs/artifacts_enhanced_publishable_reproduced/leakage_audit.json
outputs/artifacts_enhanced_publishable_reproduced/publication_readiness_report.json
```

预期主结果接近：

```text
model: transformer_4ch_extended
n_subjects: 22
MAE:  6.2966 deg
RMSE: 9.0339 deg
leakage audit: 110/110 safe
```

由于 GPU、PyTorch 版本和底层算子可能存在微小非确定性，重新运行得到的最后几位小数可能不同。论文中建议报告两位小数：`MAE=6.30 deg`，`RMSE=9.03 deg`。

## 生成审稿图表和统计表

主实验完成后，运行：

```powershell
.\run_reviewer_artifacts.ps1
```

输出目录：

```text
outputs/revision_package_reproduced/
```

该脚本会生成：

- dataset summary
- baseline comparison
- healthy/pathological bootstrap CI
- permutation tests
- feature-correlation matrix
- LOSO ridge baseline
- RF/SVR/XGBoost baselines
- prediction figures
- protocol schematic

## 复现多尺度消融和鲁棒性实验

主实验完成后，运行：

```powershell
.\run_numeric_reviewer_experiments.ps1
```

该脚本会：

- 训练 no-multiscale Transformer (`scales=1`)
- 与 multiscale Transformer (`scales=1,2,4,8`) 比较
- 使用主实验 checkpoint 进行 test-time robustness 分析

预期多尺度消融结果接近：

```text
No multiscale: MAE=6.6415 deg, RMSE=9.7923 deg
Multiscale:    MAE=6.3191 deg, RMSE=9.0362 deg
Delta:         MAE=0.3224 deg, RMSE=0.7561 deg
```

## 已保存的预期结果

本包已经保存了一份修回时使用的结果：

```text
expected_results/revision_package/
```

其中最重要的文件：

- `metrics_summary.csv`
- `metrics_by_subject.csv`
- `leakage_audit.json`
- `publication_readiness_report.json`
- `revision_tables/*.csv`
- `revision_figures/*.png`
- `response_to_reviewers.md`
- `revised_manuscript_draft.md`

## 论文写作时的正确主张

可以写：

> Under a leakage-aware purged temporal protocol, the validation-selected four-channel Transformer achieved MAE = 6.30 deg and RMSE = 9.03 deg across 22 subjects.

可以写：

> Compared with the two-channel RF/VM Transformer, the four-channel RF/BF/VM/ST Transformer reduced MAE from 9.06 deg to 6.30 deg and RMSE from 13.98 deg to 9.03 deg.

不要写：

> The revised leakage-safe model outperformed the original Table I result of MAE = 3.707 deg and RMSE = 4.691 deg.

原 Table I 结果来自更乐观的重叠窗口协议，不应作为修回稿主结论。
