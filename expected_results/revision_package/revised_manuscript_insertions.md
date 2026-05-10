# Revised Manuscript Insertions

These sections are written as copy-ready replacement or insertion text for the revised manuscript. Please merge them into the Word/LaTeX source with Track Changes, underline, or highlighting before resubmission.

## Suggested Revised Title

Leakage-Aware Subject-Specific Knee-Angle Prediction from Lower-Limb Surface Electromyography Using a Multi-Scale Transformer

## Abstract

Surface electromyography (sEMG) can provide a wearable signal source for estimating lower-limb joint kinematics, but evaluation protocols based on highly overlapping windows can overestimate model generalization. This study revisits Transformer-based knee-angle prediction using the public UCI Lower Limb EMG dataset and introduces a leakage-aware subject-specific validation protocol. Walking trials from 22 male subjects, including 11 healthy and 11 pathological subjects, were analyzed using rectus femoris, biceps femoris, vastus medialis, and semitendinosus sEMG channels and synchronized knee goniometry. We extracted time-domain features from 250 ms windows with a 1 ms step and evaluated models using purged temporal train/validation/test splits, train-only feature and target scaling, and validation-only model selection. The selected four-channel Transformer achieved `MAE=6.30 deg` and `RMSE=9.03 deg` across 22 subjects under the leakage-safe protocol, improving over the corresponding two-channel Transformer (`MAE=9.06 deg`, `RMSE=13.98 deg`). A leakage audit passed all subject/model checks, while leave-one-subject-out analysis showed substantially weaker cross-subject performance (`MAE=18.09 deg`, `RMSE=23.57 deg`), indicating the need for subject-specific calibration. These results show that multi-channel sEMG and leakage-aware validation provide a more reliable basis for assessing subject-specific knee-angle prediction models than random or overlapping hold-out splits.

## Contributions

1. We re-evaluate sEMG-based knee-angle prediction under a leakage-aware protocol that prevents overlap between training, validation, and test windows.
2. We extend the input representation from two channels (RF/VM) to four physiologically relevant thigh-muscle channels (RF/BF/VM/ST).
3. We report subject-level metrics, confidence intervals, healthy/pathological statistical tests, feature correlations, LOSO analysis, robustness tests, and protocol audit results.
4. We distinguish subject-specific performance from cross-subject generalization and explicitly state the limitations of using this public dataset for demographic and pathology-specific conclusions.

## Literature Review Addition

Prior studies have shown that sEMG can estimate lower-limb joint kinematics, but reported errors vary substantially with sensor configuration, subject population, target joint, and validation protocol. The public UCI Lower Limb EMG dataset provides synchronized four-channel thigh sEMG and knee goniometry from 22 male subjects, but it does not provide age, anthropometry, or diagnosis-specific labels. On a separate dataset, Li et al. used multiple-kernel relevance vector regression for knee-angle estimation and reported `MAE=3.27 +/- 1.20 deg` and `RMSE=4.81 +/- 1.37 deg`. Wang et al. reported a CNN-LSTM-Attention model for knee-angle prediction from six sEMG channels with `RMSE=3.09 +/- 0.90 deg` in able-bodied subjects. More recently, Wang et al. proposed a Dual Transformer Network for lower-limb joint angles and torques and reported knee-angle `RMSE=1.4312 deg` under a different multi-output experimental protocol.

These studies provide useful quantitative context, but direct numerical comparison must be made cautiously because the datasets, channels, target horizons, and validation protocols differ. The present revision therefore emphasizes protocol-safe within-dataset comparisons and reports all main results under a purged temporal split designed for highly overlapping windows.

## Dataset and Experimental Setup

The experiments used the UCI EMG Dataset in Lower Limb (DOI: `10.24432/C5ZW3P`). The dataset contains recordings from 22 male subjects, including 11 healthy subjects and 11 subjects with lower-limb pathology. Each recording includes four sEMG channels from rectus femoris (RF), biceps femoris (BF), vastus medialis (VM), and semitendinosus (ST), together with knee-angle goniometry. Signals were sampled at 1000 Hz. In this revision, only the walking files (`*mar.csv`) were used. Per-subject sample counts, durations, malformed-row counts, and train/validation/test window counts are reported in the dataset summary table.

The public dataset documentation does not provide subject age, body mass, height, or detailed pathology diagnosis. These missing demographic variables are now reported as a limitation because gait and sEMG patterns can vary with age, sex, anthropometry, and pathology type.

## Feature Extraction

For each selected sEMG channel, features were extracted using a 250 ms sliding window and a 1 ms step. The window length was retained to provide short-term muscle-activation context for time-domain descriptors. Because the 1 ms step creates highly overlapping windows, these windows were not treated as independent observations in the statistical analysis. The original feature set includes maximum value, variance, mean absolute value, mean, minimum value, root mean square, zero crossing count, and standard deviation. The enhanced feature set additionally includes waveform length, integrated EMG, slope sign changes, skewness, kurtosis, and spectral centroid. A feature-correlation matrix was added to characterize redundancy among these time-domain features.

The knee-angle label for each feature window was aligned to the prediction horizon. Main results use the 1 ms ahead target. All feature and target scalers were fit only on the training portion of each subject, and all metrics were computed after inverse transforming predictions back to degrees.

## Leakage-Aware Validation Protocol

Because a 250 ms window with a 1 ms step produces highly overlapping observations, random window-level splitting can place near-duplicate windows in both training and test sets. The revised protocol therefore uses purged temporal splitting. For each subject, the chronological data are divided into training, validation, and test regions before evaluation windows are assigned to each region. Windows crossing split boundaries are excluded from the downstream evaluation region, and no scaler is fit on validation or test data. Model selection is performed only with validation MAE averaged across subjects; test results of non-selected models are reported as exploratory and are not used for selection.

The leakage audit verifies split mode, scaler scope, window overlap, and post-processing rules for every subject/model specification. In the revised experiments, all checked specifications passed the audit (`110/110` safe).

## Main Results

Under the leakage-safe protocol, the validation-selected model was the four-channel Transformer with extended features. It achieved `ME=-1.04 deg`, `MAE=6.30 deg`, and `RMSE=9.03 deg` across all 22 subjects. The healthy group had `MAE=7.39 deg` and `RMSE=10.01 deg`, while the pathological group had `MAE=5.20 deg` and `RMSE=8.06 deg`.

Compared with the two-channel RF/VM Transformer under the same protocol (`MAE=9.06 deg`, `RMSE=13.98 deg`), the four-channel extended Transformer reduced MAE by `2.76 deg` and RMSE by `4.94 deg`. The CausalGaitNet exploratory model achieved `MAE=6.10 deg` and `RMSE=8.95 deg` on the test split, but it was not selected as the primary model because validation-only model selection favored the Transformer.

The original 70/30 overlapping-window result (`MAE=3.71 deg`, `RMSE=4.69 deg`) is retained only as historical context and is no longer used as the primary evidence, because it was not obtained under the revised leakage-aware protocol.

## Cross-Subject Generalization

A leave-one-subject-out ridge baseline was added to assess cross-subject generalization. The LOSO model achieved `MAE=18.09 deg` and `RMSE=23.57 deg`, which is substantially worse than subject-specific training. This result indicates that the current dataset and model formulation support subject-specific calibration more strongly than subject-independent deployment. The revised manuscript therefore avoids presenting the model as a universal cross-subject predictor.

## Multiscale Ablation

To quantify the contribution of multiscale aggregation, we trained a no-multiscale Transformer with `scales=1` under the same purged temporal protocol and compared it with the matched multiscale Transformer using `scales=1,2,4,8`. The no-multiscale model achieved `MAE=6.64 deg` and `RMSE=9.79 deg`, whereas the multiscale model achieved `MAE=6.32 deg` and `RMSE=9.04 deg` in the matched seed-42 comparison. Multiscale aggregation therefore reduced MAE by `0.32 deg` and RMSE by `0.76 deg`.

## Robustness to Signal Perturbations

We evaluated the selected four-channel Transformer under test-time perturbations without retraining. Gaussian feature noise had little effect (`std=0.05`: `MAE=6.30 deg`; `std=0.10`: `MAE=6.30 deg`). Channel dropout caused larger degradation, increasing MAE to `8.18 deg` for RF dropout, `7.63 deg` for BF dropout, `7.25 deg` for VM dropout, and `8.25 deg` for ST dropout. Amplitude scaling also affected performance: `0.8x` scaling yielded `MAE=6.95 deg`, while `1.2x` scaling yielded `MAE=6.62 deg`. These findings indicate that the model is relatively stable to small feature noise but remains sensitive to electrode/channel failure, particularly RF and ST loss.

## Healthy versus Pathological Statistical Analysis

Subject-level bootstrap confidence intervals and non-parametric permutation tests were used to compare healthy and pathological groups. The statistical unit was the subject-level error, not the overlapping feature window, to avoid inflating the effective sample size. The mean healthy-group MAE was `7.39 deg` with a bootstrap 95% CI of `[5.84, 9.13]`, while the pathological-group MAE was `5.20 deg` with a bootstrap 95% CI of `[4.56, 5.90]`. A two-sided permutation test found a significant MAE difference (`p=0.0320`), whereas the RMSE difference was not significant at the 0.05 level (`p=0.1004`). These results should be interpreted cautiously because detailed pathology and demographic covariates are unavailable.

## Clinical and Practical Interpretation

The revised leakage-safe RMSE of approximately `9 deg` suggests that four-channel sEMG can track the broad temporal pattern of knee flexion-extension during walking, but the error remains too large for high-precision clinical measurement or fine-grained rehabilitation decisions without additional sensing, calibration, or subject-specific adaptation. The model may be more appropriate as a component in coarse movement-intent estimation or trend monitoring than as a replacement for motion-capture or goniometric measurement.

## Negative Knee-Angle Values

Negative values in the knee-angle signal reflect the goniometer reference orientation and the flexion-extension coordinate convention used during data acquisition. They do not indicate anatomically impossible knee motion. Figure captions should explicitly state this convention and identify whether example subjects are healthy or pathological.

## Classical Baselines

To address the need for stronger non-neural baselines, we added random forest, RBF-SVR, and XGBoost regressors using the same purged temporal train/validation/test boundaries and train-only feature standardization. Hyperparameters were selected only on the validation segment for each subject. The classical baselines achieved RF `MAE=7.91 deg`, `RMSE=11.36 deg`; SVR `MAE=8.90 deg`, `RMSE=12.56 deg`; and XGBoost `MAE=7.98 deg`, `RMSE=11.17 deg`. These results support the conclusion that the four-channel Transformer improves over both the original MLP baseline and standard classical regressors under the leakage-safe protocol.

## Limitations

This study has several limitations. First, the public dataset provides only limited demographic information, so age-, height-, weight-, and pathology-specific analyses cannot be performed. Second, the strongest leakage-safe results remain subject-specific; cross-subject performance is substantially weaker. Third, the revised leakage-safe result does not outperform the original optimistic overlapping-window Table I result. Fourth, robustness testing shows that the model is sensitive to complete channel dropout, so practical deployment would require electrode-quality monitoring, redundancy, or online recalibration.

## References to Add or Verify

1. A. J. S. Sanchez and M. A. Sotelo, "EMG Dataset in Lower Limb," UCI Machine Learning Repository, 2014, doi: `10.24432/C5ZW3P`.
2. H.-B. Li, X.-R. Guan, Z. Li, K.-F. Zou, and L. He, "Estimation of Knee Joint Angle from Surface EMG Using Multiple Kernels Relevance Vector Regression," Sensors, vol. 23, no. 10, Article 4934, 2023, doi: `10.3390/s23104934`.
3. Z. Wang, H. Chen, F. Yang, X. Wang, X. Wu, and C. Chen, "Accurate Prediction of Knee Joint Angles Using a Hybrid CNN-LSTM-Attention Network from Surface Electromyography," in 2024 IEEE International Conference on Robotics and Biomimetics (ROBIO), 2024, doi: `10.1109/ROBIO64047.2024.10907528`.
4. Z. Wang, C. Chen, H. Chen, Y. Zhou, X. Wang, and X. Wu, "Dual Transformer Network for Predicting Joint Angles and Torques From Multi-Channel EMG Signals in the Lower Limbs," IEEE Journal of Biomedical and Health Informatics, 2025, doi: `10.1109/JBHI.2025.3555255`.
