# Revision Package Summary

This package contains the reviewer-facing materials for the revised IEEE Sensors Journal manuscript:

`Sensors-91956-2025, Transformer-Based Knee Angle Prediction Using Surface Electromyography`.

## Main Revision Strategy

The original optimistic result (`MAE=3.707 deg`, `RMSE=4.691 deg`) should not be used as the main claim because it was produced under an overlapping-window protocol that reviewers correctly questioned.

The revised publication-safe claim is narrower:

> Four-channel RF/BF/VM/ST sEMG with a multi-scale Transformer improves subject-specific knee-angle prediction under a leakage-aware purged temporal validation protocol, and the study demonstrates why protocol auditing is necessary for highly overlapping sEMG windows.

## Main Leakage-Aware Result

Validation-selected model:

- `transformer_4ch_extended`
- Channels: RF, BF, VM, ST
- Feature set: extended time-domain features
- Protocol: purged temporal train/validation/test split
- Scalers: fit on training data only
- Model selection: validation MAE only
- Main test result across 22 subjects: `MAE=6.2966 deg`, `RMSE=9.0339 deg`
- Leakage audit: `110/110` checks passed

The enhanced run used five training seeds (`42, 3407, 2026, 2027, 7`) for the neural models. Main configuration selection was based on validation performance, not test performance.

## Reviewer-Requested Evidence

Completed:

- Clarified population: 22 male subjects, 11 healthy and 11 pathological.
- Added four-channel RF/BF/VM/ST experiment.
- Added purged temporal validation and leakage audit.
- Added MAE/RMSE-focused reporting.
- Added RF/SVR/XGBoost traditional baselines.
- Added LOSO ridge cross-subject baseline.
- Added multiscale ablation.
- Added confidence intervals and subject-level permutation tests.
- Added per-subject window counts.
- Added feature-correlation matrix.
- Added robustness tests for noise, channel dropout, amplitude scaling, and smoothing.
- Added clinical interpretation and limitations.

## Core Files

- `cover_letter_to_editor.md`: cover letter for the resubmission.
- `response_to_reviewers.md`: point-by-point reviewer response.
- `revised_manuscript_insertions.md`: copy-ready replacement/insertion text for the manuscript.
- `revision_checklist.md`: reviewer-comment tracking table.
- `revision_tables/`: CSV tables for dataset summary, baselines, ablations, LOSO, statistics, literature comparison, and robustness.
- `revision_figures/`: generated figures for setup, protocol audit, feature correlation, and example predictions.
- `leakage_audit.json`: machine-readable leakage audit.
- `publication_readiness_report.json`: summary of main result and protocol safety.

## Remaining Submission Step

Merge the revised text, tables, and figures into the Word/LaTeX manuscript source and enable Track Changes, underlining, or highlighting, as requested in the IEEE decision letter.
