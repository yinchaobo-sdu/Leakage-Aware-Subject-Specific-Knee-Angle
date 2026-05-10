# Revision Checklist

| Reviewer | Issue | Action | Artifact | Status |
|---|---|---|---|---|
| R1 | Inconsistent aim and unreadable graphical abstract | Reframed aim as healthy + pathological subject-specific leakage-aware prediction; generated setup schematic | `revision_figures/experimental_setup_schematic.png` | Analysis completed; integrate final graphic into manuscript |
| R1 | Weak quantitative literature review | Added prior-work comparison table with dataset/protocol/result fields and verified key DOI entries | `revision_tables/literature_comparison.csv` | Completed |
| R1 | Missing demographics/setup/duration | Added dataset summary with 22 male subjects, 11/11 groups, durations, valid rows, skipped rows, and window counts | `revision_tables/dataset_summary.csv` | Completed |
| R1 | BF/ST omitted | Main enhanced experiment uses RF/BF/VM/ST and reports 2ch vs 4ch improvement | `revision_tables/ablation_model_components.csv` | Completed |
| R1 | Negative knee-angle values unclear | Added manuscript explanation based on goniometer reference orientation and figure-caption instruction | `revised_manuscript_insertions.md` | Completed |
| R1 | 250 ms / 1 ms overlap leakage | Replaced main result with purged temporal split and train-only scaling; added audit | `leakage_audit.json`, `revision_figures/protocol_leakage_diagram.png` | Completed |
| R1 | Feature redundancy | Added feature correlation matrix and CSV | `revision_figures/feature_correlation_matrix.png`, `revision_tables/feature_correlation.csv` | Completed |
| R1 | Single 70/30 split weak | Added purged temporal validation and LOSO ridge cross-subject baseline | `revision_tables/loso_metrics.csv`, `revision_tables/loso_summary.csv` | Completed |
| R1 | Baselines insufficient | Added MLP, two-channel Transformer, four-channel Transformer, CausalGaitNet, LOSO ridge, RF, SVR, and XGBoost | `revision_tables/baseline_comparison.csv`, `revision_tables/traditional_baseline_summary.csv` | Completed |
| R1 | Discussion and clinical interpretation missing | Added clinical interpretation and limitations text | `revised_manuscript_insertions.md` | Completed |
| R2 | Need LOSO/cross-subject analysis | Added LOSO ridge baseline and limitation text | `revision_tables/loso_summary.csv` | Completed |
| R2 | Need multiscale ablation | Added matched no-multiscale vs multiscale Transformer ablation | `revision_tables/ablation_multiscale.csv`, `revision_tables/ablation_multiscale_by_subject.csv` | Completed |
| R2 | Need confidence intervals/statistical tests/window counts | Added bootstrap CIs, subject-level permutation tests, and subject window counts | `revision_tables/healthy_unhealthy_statistics.csv`, `revision_tables/statistical_tests.csv`, `revision_tables/dataset_summary.csv` | Completed |
| R2 | Signal reliability | Added perturbation robustness tests for noise, channel dropout, amplitude scaling, and smoothing | `revision_tables/robustness_summary.csv`, `revision_tables/robustness_by_subject.csv` | Completed |
| R2 | Minor typo/figure/language edits | Prepared corrected wording for RF/VM typo, Figure 2 legend labels, and noun-verb agreement | `revised_manuscript_insertions.md` | Text prepared; apply in final manuscript source |

## Publication Decision

The leakage-safe enhanced result is suitable as the main result for a revised manuscript if the claims are narrowed. It supports the claim that four-channel sEMG and strict validation improve over the two-channel strict baseline. It does not support the claim that the revised model outperforms the original optimistic Table I result.

## Submission Step

Merge the revised text into the manuscript source with highlighted or tracked changes before uploading to IEEE Sensors Journal.
