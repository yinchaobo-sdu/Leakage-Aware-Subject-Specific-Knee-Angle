# Response to Reviewers

Manuscript: Sensors-91956-2025, "Transformer-Based Knee Angle Prediction Using Surface Electromyography"

Dear Dr. Eid and Reviewers,

We sincerely thank the editor and reviewers for the detailed comments. We have thoroughly reworked the manuscript in response to the concerns, especially the risk of over-optimistic performance caused by highly overlapping windows. The revised manuscript no longer presents the original overlapping-window 70/30 split as the main evidence. Instead, it reframes the study as a leakage-aware, subject-specific evaluation of sEMG-based knee-angle prediction.

The major changes are as follows:

1. The study aim has been clarified as subject-specific knee-angle prediction in both healthy and pathological male subjects.
2. The main evaluation protocol has been replaced by purged temporal train/validation/test splitting with train-only scaling and validation-only model selection.
3. The model input has been extended from two channels (RF/VM) to four physiologically relevant thigh channels (RF/BF/VM/ST).
4. The revised manuscript now reports MAE and RMSE as primary metrics, adds RF/SVR/XGBoost baselines, adds a leave-one-subject-out analysis, and includes statistical tests, confidence intervals, window counts, robustness tests, and a feature-correlation matrix.
5. A leakage audit was added and all checked subject/model specifications passed (110/110).

Under the revised leakage-aware protocol, the validation-selected four-channel Transformer achieved ME = -1.04 deg, MAE = 6.30 deg, and RMSE = 9.03 deg across 22 subjects. We explicitly state that these leakage-safe results should not be compared as if they were the same protocol as the original optimistic Table I values (MAE = 3.71 deg, RMSE = 4.69 deg). The claims have therefore been narrowed to the more defensible conclusion that four-channel sEMG and leakage-aware validation provide a more reliable assessment of subject-specific knee-angle prediction.

## Reviewer 1

**Comment 1: The pictorial abstract's font is too small to read, and the stated aim is inconsistent: first targeting patients with knee abnormalities, then both healthy subjects and patients. It is unclear which populations were studied.**

Response: We agree and have revised the aim throughout the manuscript. The revised study population is now stated consistently as 22 male subjects from the public UCI Lower Limb EMG dataset, including 11 healthy subjects and 11 subjects with lower-limb pathology. The revised aim is subject-specific, leakage-aware knee-angle prediction from lower-limb sEMG in both groups. The graphical abstract will be redrawn with larger typography and the same population statement. We also generated a new experimental setup schematic (`revision_figures/experimental_setup_schematic.png`) for manuscript integration.

**Comment 2: The literature review lacks quantitative context; prior EMG-based knee-angle prediction results, especially those using this public dataset, should be summarized and compared to the current findings.**

Response: We expanded the literature review and added a quantitative comparison table. The table separates the public UCI dataset source from related sEMG knee-angle estimation studies that used different subjects, sensors, and validation protocols. It includes, among others, the UCI Lower Limb EMG dataset (DOI: 10.24432/C5ZW3P), Li et al.'s MKRVR knee-angle estimation study (MAE 3.27 +/- 1.20 deg, RMSE 4.81 +/- 1.37 deg, DOI: 10.3390/s23104934), Wang et al.'s ROBIO 2024 CNN-LSTM-Attention study (RMSE 3.09 +/- 0.90 deg, DOI: 10.1109/ROBIO64047.2024.10907528), and Wang et al.'s 2025 Dual Transformer Network study (knee-angle RMSE 1.4312 deg under a different multi-output protocol, DOI: 10.1109/JBHI.2025.3555255). The generated table is provided in `revision_tables/literature_comparison.csv`.

**Comment 3: Subject demographics are not reported, yet gait varies by age, sex, and pathology. A schematic of the experimental setup and clear trial durations are needed.**

Response: We expanded the dataset description. The public dataset reports 22 male subjects, including 11 healthy and 11 pathological subjects, recorded at 1000 Hz using four sEMG channels and a knee goniometer. We added a per-subject dataset summary table reporting the walking file, valid samples, skipped malformed rows, approximate duration, and train/validation/test window counts (`revision_tables/dataset_summary.csv`). Age, height, mass, and detailed pathology type are not available in the public dataset documentation; the revised manuscript now explicitly lists these missing variables as limitations. A setup schematic has also been added as `revision_figures/experimental_setup_schematic.png`.

**Comment 4: Biceps femoris and semitendinosus are key to knee flexion but were excluded; the muscle-selection rationale must be explained. Negative joint-angle values and the "typical subject" status are unclear.**

Response: We agree. The revised main experiment now uses all four available sEMG channels: rectus femoris (RF), biceps femoris (BF), vastus medialis (VM), and semitendinosus (ST). Under the same leakage-safe protocol, the four-channel Transformer improved over the two-channel RF/VM Transformer from MAE = 9.06 deg and RMSE = 13.98 deg to MAE = 6.30 deg and RMSE = 9.03 deg. The example prediction figures now identify whether the subject is healthy or pathological. Negative knee-angle values are explained as a consequence of the goniometer reference orientation and the flexion-extension coordinate convention, not as anatomically impossible knee motion.

**Comment 5: A 250 ms window with a 1 ms step introduces highly overlapping, non-IID samples and risks data leakage if trials are merged before feature extraction; the choice of window and step size should be justified.**

Response: We agree that this was the most important methodological risk. The revised manuscript keeps the 250 ms window because it provides sufficient short-term sEMG context for time-domain feature extraction, but it no longer evaluates by random or unpurged window-level splitting. We implemented purged temporal train/validation/test splitting for each subject. Windows crossing split boundaries are excluded from evaluation regions, and feature and target scalers are fit only on the training segment. The revised manuscript includes a leakage protocol diagram (`revision_figures/protocol_leakage_diagram.png`) and a machine-readable leakage audit (`leakage_audit.json`), which passed all checked specifications (110/110).

**Comment 6: The eight time-domain features may be redundant and sensitive to outliers; their selection requires justification, and a correlation matrix would better illustrate feature relationships than partial plots in Figure 2.**

Response: We added a feature-correlation analysis and revised the feature discussion. The revised manuscript acknowledges that some time-domain features are correlated, which is expected for short-window sEMG descriptors. The correlation matrix is now used to show feature relationships more clearly than the original partial feature traces. The matrix figure is provided in `revision_figures/feature_correlation_matrix.png`, and the numerical values are provided in `revision_tables/feature_correlation.csv`. We also added an extended feature-set experiment including waveform length, integrated EMG, slope sign changes, skewness, kurtosis, and spectral centroid.

**Comment 7: A single 70/30 hold-out split on overlapping observations is prone to leakage; trials should be divided first, then features extracted, and model performance validated with k-fold or repeated hold-out.**

Response: We replaced the original 70/30 overlapping hold-out with purged temporal train/validation/test evaluation. The revised analysis divides each subject's chronological walking record into train, validation, and test regions before assigning windows to downstream evaluation regions. We also added a leave-one-subject-out (LOSO) ridge baseline to characterize cross-subject generalization. The LOSO model achieved MAE = 18.09 deg and RMSE = 23.57 deg, substantially worse than subject-specific training. This result is now used to explicitly limit the claims to subject-specific calibration. We agree that future work should include repeated purged hold-out or cycle-level blocked k-fold validation once reliable gait-cycle annotations are available.

**Comment 8: Reporting mean error is misleading since positive and negative deviations cancel; MAE and RMSE are more appropriate. Comparing only against an MLP baseline is insufficient; standard models such as RF, SVM, and XGBoost should be included, with architectures and tuning ranges detailed.**

Response: We now use MAE and RMSE as the primary metrics and report ME only as a bias indicator. We added random forest, RBF-SVR, and XGBoost baselines under the same purged temporal boundaries and train-only standardization. The traditional baselines achieved RF MAE = 7.91 deg and RMSE = 11.36 deg, SVR MAE = 8.90 deg and RMSE = 12.56 deg, and XGBoost MAE = 7.98 deg and RMSE = 11.17 deg. Their tuning grids are reported in `revision_tables/traditional_baseline_summary.csv`. The manuscript also continues to report the MLP baseline under the revised protocol for continuity with the original submission.

**Comment 9: The manuscript lacks a discussion of results, clinical implications, and statistical comparisons with previous studies; significance testing and interpretation of RMSE in a rehabilitation context are essential.**

Response: We added a discussion section addressing clinical interpretation, practical reliability, limitations, and comparison with prior studies. The revised leakage-safe RMSE of approximately 9 deg is interpreted as sufficient for coarse temporal tracking of knee flexion-extension during walking, but not sufficient as a replacement for high-precision clinical goniometry or motion capture in fine-grained rehabilitation decisions. We added subject-level bootstrap confidence intervals and non-parametric permutation tests for healthy versus pathological groups. Healthy subjects had MAE = 7.39 deg with a 95% bootstrap CI of [5.84, 9.13], while pathological subjects had MAE = 5.20 deg with a 95% CI of [4.56, 5.90]. A two-sided permutation test found a significant MAE difference (p = 0.0320), whereas the RMSE difference was not significant at the 0.05 level (p = 0.1004). We caution that these group results should not be over-interpreted because age and detailed pathology information are unavailable.

## Reviewer 2

**Comment 1: The results seem specific to each individual subject. Please provide a leave-one-subject-out or other cross-subject analysis, or clearly explain this limitation.**

Response: We added a LOSO cross-subject baseline and explicitly state that the main model is subject-specific. The LOSO ridge baseline achieved MAE = 18.09 deg and RMSE = 23.57 deg, confirming that cross-subject generalization is substantially weaker than subject-specific calibration on this dataset. This limitation is now stated in the abstract, discussion, and limitations.

**Comment 2: Please quantify the benefit of using multi-scale down-sampling by including an ablation study that compares results with and without this technique.**

Response: We added a matched multiscale ablation. With the same seed and purged temporal protocol, the no-multiscale Transformer (`scales=1`) achieved MAE = 6.64 deg and RMSE = 9.79 deg, whereas the multiscale Transformer (`scales=1,2,4,8`) achieved MAE = 6.32 deg and RMSE = 9.04 deg. Multiscale aggregation therefore reduced MAE by 0.32 deg and RMSE by 0.76 deg. Full subject-level results are provided in `revision_tables/ablation_multiscale_by_subject.csv`.

**Comment 3: The study includes 11 healthy people and 11 people with conditions. When discussing differences between these groups, the authors should provide confidence intervals and proper statistical tests. They should also state how many data windows come from each person to make sure the data are not accidentally repeated or mixed.**

Response: We added per-subject window counts in `revision_tables/dataset_summary.csv`, bootstrap confidence intervals in `revision_tables/healthy_unhealthy_statistics.csv`, and two-sided non-parametric permutation tests in `revision_tables/statistical_tests.csv`. We also clarified that statistical tests are performed at the subject level, not at the overlapping-window level, to avoid treating highly correlated windows as independent samples.

**Comment 4: For real-world use, could the authors discuss how reliable the model is when there are changes or problems with signal recording and processing?**

Response: We added test-time robustness analyses without retraining. Gaussian feature noise had negligible effect (std = 0.05: delta MAE +0.002 deg; std = 0.10: delta MAE -0.000 deg). Channel dropout caused the largest degradation: RF dropout increased MAE by +1.88 deg, BF by +1.34 deg, VM by +0.95 deg, and ST by +1.96 deg. Amplitude scaling caused moderate degradation (0.8x: delta MAE +0.65 deg; 1.2x: delta MAE +0.33 deg). The revised discussion therefore states that the model is stable to small feature noise but sensitive to complete electrode/channel failure, motivating electrode-quality monitoring and recalibration in practical deployment. Full results are provided in `revision_tables/robustness_summary.csv` and `revision_tables/robustness_by_subject.csv`.

**Comment 5: Minor edits: (a) Correct "RM and VF muscles" to "RF and VM muscles". (b) Figure 2: label the six displayed features directly in the legend. (c) Ensure consistent noun-verb agreement.**

Response: We corrected "RM and VF" to "RF and VM" and revised the figure-caption plan so that displayed features are labeled directly in the legend. We also edited the manuscript for noun-verb agreement, including consistent use of "data were" where "data" is treated as plural.

## Final Revision Status

All reviewer-requested analyses have been completed in the revision package:

- Leakage-aware purged temporal protocol and audit.
- Four-channel RF/BF/VM/ST main experiment.
- MAE/RMSE-focused reporting.
- RF/SVR/XGBoost baselines and MLP continuity baseline.
- LOSO cross-subject baseline.
- Multiscale ablation.
- Healthy/pathological confidence intervals and subject-level permutation tests.
- Per-subject window counts.
- Feature-correlation matrix.
- Test-time robustness analysis.
- Expanded discussion, clinical interpretation, and limitations.

The remaining submission step is to merge the revised text into the Word or LaTeX manuscript using Track Changes, underline, or highlighting, as requested by IEEE Sensors Journal.
