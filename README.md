# Leakage-Aware Subject-Specific Knee-Angle Estimation

This repository provides a reproducible experiment package for the revised manuscript experiments on leakage-aware subject-specific knee-angle estimation from lower-limb EMG signals.

## Repository Structure

```text
.
├── code/                 Experiment source code
├── data/SEMG_DB1/        Copy of the UCI Lower Limb EMG dataset
├── expected_results/     Reference outputs generated for the revision
├── outputs/              Outputs generated after rerunning the scripts
├── run_smoke_test.ps1
├── run_main_experiment.ps1
├── run_reviewer_artifacts.ps1
└── run_numeric_reviewer_experiments.ps1
```

## Environment Setup

Python 3.10 or later is recommended. On Windows PowerShell, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\code\requirements.txt
```

If an NVIDIA GPU is available, install a PyTorch build that matches the local CUDA version. The code can also run on CPU, but the full experiments will take substantially longer.

## Quick Smoke Test

The smoke test runs one subject for one epoch. It is intended to verify that the environment and data paths are configured correctly.

```powershell
.\run_smoke_test.ps1
```

Expected output directory:

```text
outputs/smoke/
```

## Main Experiment

The main experiment reproduces the leakage-aware four-channel Transformer result reported in the revised manuscript.

```powershell
.\run_main_experiment.ps1
```

Main output files:

```text
outputs/artifacts_enhanced_publishable_reproduced/metrics_summary.csv
outputs/artifacts_enhanced_publishable_reproduced/metrics_by_subject.csv
outputs/artifacts_enhanced_publishable_reproduced/leakage_audit.json
outputs/artifacts_enhanced_publishable_reproduced/publication_readiness_report.json
```

The expected main result is approximately:

```text
model: transformer_4ch_extended
n_subjects: 22
MAE:  6.2966 deg
RMSE: 9.0339 deg
leakage audit: 110/110 safe
```

Small numerical differences may occur because of GPU hardware, PyTorch versions, and low-level operator nondeterminism. In the manuscript, report the rounded values as `MAE = 6.30 deg` and `RMSE = 9.03 deg`.

## Reviewer Figures and Statistical Tables

After completing the main experiment, generate the reviewer-facing figures and statistical tables with:

```powershell
.\run_reviewer_artifacts.ps1
```

Expected output directory:

```text
outputs/revision_package_reproduced/
```

This script generates:

- dataset summary
- baseline comparison
- healthy/pathological bootstrap confidence intervals
- permutation tests
- feature-correlation matrix
- LOSO ridge baseline
- RF, SVR, and XGBoost baselines
- prediction figures
- protocol schematic

## Multiscale Ablation and Robustness Experiments

After completing the main experiment, run:

```powershell
.\run_numeric_reviewer_experiments.ps1
```

This script:

- trains the no-multiscale Transformer (`scales=1`)
- compares it with the multiscale Transformer (`scales=1,2,4,8`)
- performs test-time robustness analysis using the main experiment checkpoint

The expected multiscale ablation result is approximately:

```text
No multiscale: MAE=6.6415 deg, RMSE=9.7923 deg
Multiscale:    MAE=6.3191 deg, RMSE=9.0362 deg
Delta:         MAE=0.3224 deg, RMSE=0.7561 deg
```

## Saved Reference Results

Reference results used during the manuscript revision are included in:

```text
expected_results/revision_package/
```

Important files include:

- `metrics_summary.csv`
- `metrics_by_subject.csv`
- `leakage_audit.json`
- `publication_readiness_report.json`
- `revision_tables/*.csv`
- `revision_figures/*.png`
- `response_to_reviewers.md`
- `revised_manuscript_draft.md`

## Recommended Manuscript Claims

Appropriate claim:

> Under a leakage-aware purged temporal protocol, the validation-selected four-channel Transformer achieved MAE = 6.30 deg and RMSE = 9.03 deg across 22 subjects.

Appropriate claim:

> Compared with the two-channel RF/VM Transformer, the four-channel RF/BF/VM/ST Transformer reduced MAE from 9.06 deg to 6.30 deg and RMSE from 13.98 deg to 9.03 deg.

Avoid claiming:

> The revised leakage-safe model outperformed the original Table I result of MAE = 3.707 deg and RMSE = 4.691 deg.

The original Table I result was obtained under a more optimistic overlapping-window protocol and should not be used as the main conclusion of the revised manuscript.
