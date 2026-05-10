# Leakage-Aware Subject-Specific Knee-Angle Estimation

This repository contains the experiment code and reproducibility materials for the leakage-aware subject-specific knee-angle estimation experiments reported in the revised manuscript. The goal is to let readers rerun the main experiment, audit the train/validation/test protocol, and compare their outputs with the saved reference results.

## What Is Included

```text
.
|-- code/                 Python source code
|-- data/SEMG_DB1/        Copy of the UCI Lower Limb EMG dataset
|-- expected_results/     Reference outputs from the manuscript revision
|-- REPRODUCIBILITY_MANIFEST.md
|-- run_smoke_test.ps1
|-- run_main_experiment.ps1
|-- run_verify_results.ps1
|-- run_reviewer_artifacts.ps1
`-- run_numeric_reviewer_experiments.ps1
```

Generated files are written to `outputs/`. This directory is intentionally ignored by Git so that reruns do not overwrite the reference package.

## Reproducibility Checklist

1. Clone the repository.
2. Create a fresh Python environment.
3. Install the dependencies from `code/requirements.txt`.
4. Run the smoke test.
5. Run the main experiment.
6. Verify the generated metrics against the reference values.

The main result is produced by a validation-selected four-channel Transformer under a purged temporal protocol with train-only scaling. The expected rounded test metrics are:

```text
MAE  = 6.30 deg
RMSE = 9.03 deg
leakage audit = 110/110 safe
```

## Environment Setup

Python 3.10 or 3.11 is recommended. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\code\requirements.txt
```

The scripts use `--device auto`, which selects CUDA when a compatible NVIDIA GPU and PyTorch build are available and otherwise falls back to CPU. CPU execution works, but the full experiment can be slow.

For a CUDA-specific PyTorch installation, follow the official PyTorch selector for your local CUDA version, then rerun the dependency installation if needed.

## Smoke Test

Run the smoke test before launching the full experiment:

```powershell
.\run_smoke_test.ps1
```

The smoke test runs unit checks and then trains one subject for one epoch. It confirms that imports, data discovery, feature extraction, model construction, and output writing are functioning. Its results are not intended for manuscript reporting.

## Main Experiment

Run:

```powershell
.\run_main_experiment.ps1
```

This writes the main outputs to:

```text
outputs/artifacts_enhanced_publishable_reproduced/
```

Key files:

```text
metrics_summary.csv
metrics_by_subject.csv
leakage_audit.json
publication_readiness_report.json
validation_selection_report.json
```

The main command uses:

```text
subjects: all 22 walking files
channels: RF, BF, VM, ST
feature set: extended
split: purged temporal
scalers: fit on training data only
epochs: 160
seeds: 42, 3407, 2026, 2027, 7
selection seed: 42
```

Small numerical differences may occur across GPU models, CUDA versions, PyTorch versions, and low-level kernels. The manuscript-level comparison should use the rounded values `MAE = 6.30 deg` and `RMSE = 9.03 deg`.

## Verify a Rerun

After the main experiment finishes, run:

```powershell
.\run_verify_results.ps1
```

The verifier checks that:

- `transformer_4ch_extended` is the selected main model.
- The overall test MAE and RMSE are close to the reference values.
- The leakage audit reports all subject/model checks as publication safe.

By default, the tolerance is `0.25 deg` for both MAE and RMSE. You can use a wider tolerance for substantially different hardware or library versions:

```powershell
.\run_verify_results.ps1 -Tolerance 0.5
```

## Reference Results

Saved reference outputs are provided in:

```text
expected_results/revision_package/
```

These files are useful for checking a rerun without retraining:

- `metrics_summary.csv`
- `metrics_by_subject.csv`
- `leakage_audit.json`
- `publication_readiness_report.json`
- `validation_selection_report.json`
- `revision_tables/*.csv`
- `revision_figures/*.png`

## Reviewer Tables and Figures

After the main experiment has produced checkpoints and predictions, generate the reviewer-facing tables and figures with:

```powershell
.\run_reviewer_artifacts.ps1
```

Output directory:

```text
outputs/revision_package_reproduced/
```

This produces dataset summaries, baseline comparisons, confidence intervals, permutation tests, feature-correlation matrices, LOSO ridge baselines, RF/SVR/XGBoost baselines, prediction figures, and protocol schematics.

## Multiscale Ablation and Robustness Experiments

After the main experiment, run:

```powershell
.\run_numeric_reviewer_experiments.ps1
```

This script trains the no-multiscale Transformer, compares it with the multiscale Transformer, and performs test-time robustness analyses using the main experiment checkpoints.

Expected multiscale ablation result:

```text
No multiscale: MAE=6.6415 deg, RMSE=9.7923 deg
Multiscale:    MAE=6.3191 deg, RMSE=9.0362 deg
Delta:         MAE=0.3224 deg, RMSE=0.7561 deg
```

## Dataset Notes

The included dataset copy corresponds to the UCI EMG Dataset in Lower Limb:

- DOI: `10.24432/C5ZW3P`
- Walking files used by the main experiment: `*mar.csv`
- Subjects: 22 male subjects, 11 healthy and 11 pathological
- Channels: RF, BF, VM, ST, and knee flexion angle FX

If you replace the data directory with a fresh UCI download, keep the `A_CSV/` and `N_CSV/` folder structure. The loader searches for these folders automatically under `data/SEMG_DB1/`.

## Appropriate Manuscript Claims

Appropriate:

> Under a leakage-aware purged temporal protocol, the validation-selected four-channel Transformer achieved MAE = 6.30 deg and RMSE = 9.03 deg across 22 subjects.

Appropriate:

> Compared with the two-channel RF/VM Transformer, the four-channel RF/BF/VM/ST Transformer reduced MAE from 9.06 deg to 6.30 deg and RMSE from 13.98 deg to 9.03 deg.

Avoid:

> The revised leakage-safe model outperformed the original Table I result of MAE = 3.707 deg and RMSE = 4.691 deg.

The original Table I result used a more optimistic overlapping-window protocol and should not be treated as the main leakage-aware conclusion.

## Troubleshooting

- If `python` resolves to the wrong interpreter, activate `.venv` again or use `py -3.10`.
- If CUDA is requested but unavailable, the scripts fall back to CPU.
- If `xgboost` installation fails, the main experiment and smoke test are still the priority; `xgboost` is only needed for reviewer baseline generation.
- If rerun metrics differ slightly, first check `outputs/artifacts_enhanced_publishable_reproduced/run_config.json`, `metrics_summary.csv`, and `leakage_audit.json`.
