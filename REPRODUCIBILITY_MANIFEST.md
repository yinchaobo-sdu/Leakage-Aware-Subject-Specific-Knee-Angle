# Reproducibility Manifest

This manifest records the data, entry points, commands, and expected outputs needed to reproduce the main leakage-aware experiment.

## Dataset

- Included path: `data/SEMG_DB1/SEMG_DB1`
- Source: UCI EMG Dataset in Lower Limb
- DOI: `10.24432/C5ZW3P`
- Files used by the main experiment: 22 walking files matching `*mar.csv`
- Subject groups: 11 healthy subjects from `N_CSV/` and 11 pathological subjects from `A_CSV/`
- Input channels: RF, BF, VM, ST
- Target channel: FX knee flexion angle

The loader searches for folders named `A_CSV/` and `N_CSV/` below the supplied `--data-root`, so the repository can be moved without editing hard-coded paths.

## Recommended Environment

- Python: 3.10 or 3.11
- Operating system used for the supplied PowerShell wrappers: Windows
- Main Python dependencies: NumPy, PyTorch, Matplotlib, tqdm, scikit-learn, and XGBoost
- Install command: `pip install -r .\code\requirements.txt`

The code can run on CPU or CUDA. CUDA is recommended for the full 22-subject experiment. Small floating-point differences across GPU, CUDA, and PyTorch versions are expected.

## Main Entry Points

- `run_smoke_test.ps1`: runs unit checks and a one-subject, one-epoch sanity run
- `run_main_experiment.ps1`: reproduces the main leakage-aware experiment
- `run_verify_results.ps1`: verifies generated main metrics and leakage audit
- `run_reviewer_artifacts.ps1`: regenerates reviewer tables, figures, and statistics
- `run_numeric_reviewer_experiments.ps1`: regenerates multiscale ablation and robustness analyses

Python entry points:

- `code/run_tests.py`
- `code/run_enhanced_publishable_experiment.py`
- `code/verify_reproduction.py`
- `code/run_reviewer_artifacts.py`
- `code/run_reviewer_numeric_experiments.py`

## Main Command

PowerShell:

```powershell
.\run_main_experiment.ps1
```

Equivalent Python command:

```powershell
cd .\code
python run_enhanced_publishable_experiment.py --data-root ..\data\SEMG_DB1 --output-dir ..\outputs\artifacts_enhanced_publishable_reproduced --subjects all --epochs 160 --seeds 42,3407,2026,2027,7 --selection-seeds 42 --device auto
```

## Protocol

- Evaluation mode: leakage-aware subject-specific prediction
- Split mode: purged temporal train/validation/test split
- Sequence length: 128
- Feature window: 250 samples
- Feature step: 1 sample
- Feature scaler: standard scaler fitted on training rows only
- Target scaler: standard scaler fitted on training targets only
- Model selection: mean validation MAE across subjects
- Main model: `transformer_4ch_extended`
- Training seeds: `42,3407,2026,2027,7`
- Selection seed: `42`

## Expected Main Result

The selected main model should be:

- Model: `transformer_4ch_extended`
- Channels: RF/BF/VM/ST
- Feature set: extended
- Subjects: 22
- Mean test MAE: approximately `6.30 deg`
- Mean test RMSE: approximately `9.03 deg`
- Leakage audit: `110/110` subject/model checks safe

Reference row from `expected_results/revision_package/metrics_summary.csv`:

```text
model=transformer_4ch_extended
group=overall
main_result=True
mae=6.296599355610934
rmse=9.033915257953181
```

Verification command after a rerun:

```powershell
.\run_verify_results.ps1
```

## Output Policy

- `expected_results/` contains saved reference artifacts and is tracked by Git.
- `outputs/` contains local rerun artifacts and is ignored by Git.
- Model checkpoints, feature caches, and local run configurations should be regenerated locally rather than committed.

## Notes on Determinism

The code fixes Python, NumPy, PyTorch, and CUDA seeds for each training run. CuDNN benchmarking is disabled to reduce run-to-run variability. Exact bitwise reproducibility is still not guaranteed across different hardware and library builds, so manuscript comparisons should use rounded MAE/RMSE values and the leakage audit status.
