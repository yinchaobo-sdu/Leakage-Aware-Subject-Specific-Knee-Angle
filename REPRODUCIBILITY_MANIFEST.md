# Reproducibility Manifest

## Dataset

- Included path: `data/SEMG_DB1/SEMG_DB1`
- Source: UCI EMG Dataset in Lower Limb
- DOI: `10.24432/C5ZW3P`
- Subjects used: 22 walking files (`*mar.csv`), 11 healthy and 11 pathological

## Code Entry Points

- `code/run_tests.py`: smoke/unit tests
- `code/run_enhanced_publishable_experiment.py`: main leakage-aware neural experiments
- `code/run_reviewer_artifacts.py`: reviewer tables/figures/statistics
- `code/run_reviewer_numeric_experiments.py`: multiscale ablation and robustness experiments
- `code/run_optimized_experiment.py`: legacy helper imported by the unit tests; not used for the main revised claim

## Main Command

```powershell
.\run_main_experiment.ps1
```

Equivalent Python command:

```powershell
cd .\code
python run_enhanced_publishable_experiment.py --data-root ..\data\SEMG_DB1 --output-dir ..\outputs\artifacts_enhanced_publishable_reproduced --subjects all --epochs 160 --seeds 42,3407,2026,2027,7 --selection-seeds 42 --device auto
```

## Expected Main Result

The expected main model is selected by validation MAE:

- Model: `transformer_4ch_extended`
- Channels: RF/BF/VM/ST
- Feature set: extended
- Split: purged temporal
- Scalers: train-only
- Mean test MAE: approximately `6.30 deg`
- Mean test RMSE: approximately `9.03 deg`

## Files Not Included

Large prior artifact folders and old checkpoints from exploratory runs are not copied into this reproducibility package. They are not required to rerun the main experiment. New checkpoints will be generated under `outputs/` when `run_main_experiment.ps1` is executed.
