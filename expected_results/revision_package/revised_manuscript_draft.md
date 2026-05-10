# Leakage-Aware Subject-Specific Knee-Angle Prediction from Lower-Limb Surface Electromyography Using a Multi-Scale Transformer

Guiqian Gao, Zijing Yang, and Huanghe Zhang, Member, IEEE

## Abstract

Surface electromyography (sEMG) can provide a wearable signal source for estimating lower-limb joint kinematics, but evaluation protocols based on highly overlapping windows can overestimate model generalization. This study revisits Transformer-based knee-angle prediction using the public UCI Lower Limb EMG dataset and introduces a leakage-aware subject-specific validation protocol. Walking trials from 22 male subjects, including 11 healthy and 11 pathological subjects, were analyzed using rectus femoris, biceps femoris, vastus medialis, and semitendinosus sEMG channels and synchronized knee goniometry. We extracted time-domain features from 250 ms windows with a 1 ms step and evaluated models using purged temporal train/validation/test splits, train-only feature and target scaling, and validation-only model selection. The selected four-channel Transformer achieved MAE = 6.30 deg and RMSE = 9.03 deg across 22 subjects under the leakage-safe protocol, improving over the corresponding two-channel Transformer (MAE = 9.06 deg, RMSE = 13.98 deg). A leakage audit passed all subject/model checks, while leave-one-subject-out analysis showed substantially weaker cross-subject performance (MAE = 18.09 deg, RMSE = 23.57 deg), indicating the need for subject-specific calibration. These results show that multi-channel sEMG and leakage-aware validation provide a more reliable basis for assessing subject-specific knee-angle prediction models than random or overlapping hold-out splits.

Index Terms: Knee joint angle prediction, surface electromyography, Transformer model, gait analysis, wearable sensing, leakage-aware validation.

## I. Introduction

Gait patterns are important indicators of lower-limb function and are widely used in clinical gait analysis, rehabilitation assessment, and wearable human-machine interfaces. Conventional gait evaluation often relies on visual inspection or laboratory-grade sensing, which can limit accessibility, repeatability, and deployment outside specialized environments. Wearable sensors, including inertial measurement units, goniometers, force sensors, and bioelectrical sensors, offer an alternative path toward objective and portable gait monitoring.

Surface electromyography is especially relevant for movement-intent estimation because muscle activation precedes observable joint motion and can be measured non-invasively. Previous studies have mapped sEMG to joint angles using classical regressors, kernel models, recurrent neural networks, convolutional networks, and Transformer-based architectures. However, reported performance varies strongly with sensor placement, subject population, prediction horizon, and validation protocol. For time-series studies using dense sliding windows, random or unpurged window-level splitting can place near-duplicate observations in both training and testing sets, leading to overly optimistic results.

The original version of this study proposed a Transformer model using two sEMG channels from rectus femoris (RF) and vastus medialis (VM). In response to reviewer concerns, the present revision substantially changes the evaluation protocol and interpretation. We now treat the task as subject-specific knee-angle prediction in both healthy and pathological male subjects, extend the input representation to include biceps femoris (BF) and semitendinosus (ST), and evaluate all main results using a purged temporal validation protocol.

The contributions of this revision are:

1. We re-evaluate sEMG-based knee-angle prediction under a leakage-aware protocol that prevents overlap between training, validation, and test windows.
2. We extend the input representation from two channels (RF/VM) to four physiologically relevant thigh-muscle channels (RF/BF/VM/ST).
3. We report subject-level metrics, confidence intervals, healthy/pathological statistical tests, feature correlations, leave-one-subject-out analysis, robustness tests, and protocol audit results.
4. We distinguish subject-specific performance from cross-subject generalization and explicitly state the limitations of using this public dataset for demographic and pathology-specific conclusions.

## II. Related Work

Prior studies have shown that sEMG can estimate lower-limb joint kinematics, but direct numerical comparison is difficult because datasets, channels, target horizons, and validation protocols differ. The UCI Lower Limb EMG dataset provides synchronized four-channel thigh sEMG and knee goniometry from 22 male subjects. Li et al. used multiple-kernel relevance vector regression for knee-angle estimation and reported MAE = 3.27 +/- 1.20 deg and RMSE = 4.81 +/- 1.37 deg on a separate dataset. Wang et al. reported a CNN-LSTM-Attention model for knee-angle prediction from six sEMG channels with RMSE = 3.09 +/- 0.90 deg in able-bodied subjects. A recent Dual Transformer Network reported knee-angle RMSE = 1.4312 deg under a different multi-output lower-limb protocol.

These studies provide useful quantitative context, but they are not protocol-identical comparisons. The present work therefore emphasizes within-dataset comparisons under the same leakage-aware protocol and reports the original overlapping-window result only as historical context.

Insert Table: `revision_tables/literature_comparison.csv`.

## III. Materials and Methods

### A. Dataset and Experimental Setup

The experiments used the UCI EMG Dataset in Lower Limb (DOI: 10.24432/C5ZW3P). The dataset contains recordings from 22 male subjects, including 11 healthy subjects and 11 subjects with lower-limb pathology. Each recording includes four sEMG channels from RF, BF, VM, and ST, together with knee-angle goniometry. Signals were sampled at 1000 Hz. In this revision, only the walking files (`*mar.csv`) were used.

Per-subject sample counts, durations, malformed-row counts, and train/validation/test window counts are reported in the dataset summary table. The public dataset documentation does not provide subject age, body mass, height, or detailed pathology diagnosis. These missing demographic variables are treated as limitations because gait and sEMG patterns can vary with age, sex, anthropometry, and pathology type.

Insert Figure: `revision_figures/experimental_setup_schematic.png`.

Insert Table: `revision_tables/dataset_summary.csv`.

### B. Feature Extraction

For each selected sEMG channel, features were extracted using a 250 ms sliding window and a 1 ms step. The window length was retained to provide short-term muscle-activation context for time-domain descriptors. Because the 1 ms step creates highly overlapping windows, these windows were not treated as independent observations in statistical testing.

The original feature set includes maximum value, variance, mean absolute value, mean, minimum value, root mean square, zero crossing count, and standard deviation. The enhanced feature set additionally includes waveform length, integrated EMG, slope sign changes, skewness, kurtosis, and spectral centroid. A feature-correlation matrix was added to characterize redundancy among the descriptors.

The knee-angle label for each feature window was aligned to the prediction horizon. Main results use the 1 ms ahead target. All feature and target scalers were fit only on the training portion of each subject, and all metrics were computed after inverse transforming predictions back to degrees.

Insert Figure: `revision_figures/feature_correlation_matrix.png`.

Insert Table: `revision_tables/feature_correlation.csv`.

### C. Leakage-Aware Validation Protocol

A 250 ms window with a 1 ms step produces highly overlapping observations. Random window-level splitting can therefore place near-duplicate windows in both training and test sets. The revised protocol uses purged temporal splitting. For each subject, the chronological data are divided into training, validation, and test regions before evaluation windows are assigned to each region. Windows crossing split boundaries are excluded from downstream evaluation regions, and no scaler is fit on validation or test data.

Model selection is performed only with validation MAE averaged across subjects; test results of non-selected models are reported as exploratory and are not used for selection. The leakage audit verifies split mode, scaler scope, window overlap, and post-processing rules for every subject/model specification. In the revised experiments, all checked specifications passed the audit (110/110 safe).

Insert Figure: `revision_figures/protocol_leakage_diagram.png`.

### D. Models and Baselines

The main model is a multi-scale Transformer encoder using the four sEMG channels RF, BF, VM, and ST with extended time-domain features. The input is a sequence of feature vectors, and down-sampled branches at multiple temporal scales are aggregated before regression to the knee-angle target.

We compared the selected four-channel Transformer with a two-channel RF/VM Transformer, a two-channel MLP continuity baseline, CausalGaitNet exploratory variants, and traditional regressors. The traditional regressors include random forest, RBF-SVR, and XGBoost, all trained under the same purged temporal boundaries with train-only standardization. Hyperparameters were selected only on validation data.

### E. Evaluation Metrics and Statistical Analysis

The primary metrics are mean absolute error (MAE) and root mean square error (RMSE). Mean error (ME) is reported only as a bias indicator because positive and negative errors can cancel. Healthy/pathological comparisons were performed using subject-level errors rather than overlapping-window errors. Bootstrap confidence intervals and two-sided non-parametric permutation tests were computed on subject-level metrics.

## IV. Results

### A. Main Leakage-Aware Performance

Under the leakage-safe protocol, the validation-selected model was the four-channel Transformer with extended features. It achieved ME = -1.04 deg, MAE = 6.30 deg, and RMSE = 9.03 deg across all 22 subjects. The healthy group had MAE = 7.39 deg and RMSE = 10.01 deg, while the pathological group had MAE = 5.20 deg and RMSE = 8.06 deg.

Compared with the two-channel RF/VM Transformer under the same protocol (MAE = 9.06 deg, RMSE = 13.98 deg), the four-channel extended Transformer reduced MAE by 2.76 deg and RMSE by 4.94 deg. The original 70/30 overlapping-window result (MAE = 3.71 deg, RMSE = 4.69 deg) is retained only as historical context and is no longer used as the primary evidence because it was not obtained under the revised leakage-aware protocol.

Insert Table: `revision_tables/baseline_comparison.csv`.

Insert Figure: `revision_figures/prediction_healthy_8Nmar.png`.

Insert Figure: `revision_figures/prediction_unhealthy_3Amar.png`.

### B. Classical Baselines

Random forest, RBF-SVR, and XGBoost were evaluated under the same purged temporal train/validation/test boundaries. The traditional baselines achieved RF MAE = 7.91 deg and RMSE = 11.36 deg, SVR MAE = 8.90 deg and RMSE = 12.56 deg, and XGBoost MAE = 7.98 deg and RMSE = 11.17 deg. These results support the conclusion that the four-channel Transformer improves over both the original MLP continuity baseline and standard classical regressors under the leakage-safe protocol.

Insert Table: `revision_tables/traditional_baseline_summary.csv`.

### C. Cross-Subject Generalization

A leave-one-subject-out ridge baseline was added to assess cross-subject generalization. The LOSO model achieved MAE = 18.09 deg and RMSE = 23.57 deg, which is substantially worse than subject-specific training. This result indicates that the current dataset and model formulation support subject-specific calibration more strongly than subject-independent deployment.

Insert Table: `revision_tables/loso_summary.csv`.

### D. Multiscale Ablation

To quantify the contribution of multiscale aggregation, we trained a no-multiscale Transformer with `scales=1` under the same purged temporal protocol and compared it with the matched multiscale Transformer using `scales=1,2,4,8`. The no-multiscale model achieved MAE = 6.64 deg and RMSE = 9.79 deg, whereas the multiscale model achieved MAE = 6.32 deg and RMSE = 9.04 deg in the matched seed-42 comparison. Multiscale aggregation therefore reduced MAE by 0.32 deg and RMSE by 0.76 deg.

Insert Table: `revision_tables/ablation_multiscale.csv`.

### E. Healthy versus Pathological Statistical Analysis

Subject-level bootstrap confidence intervals and non-parametric permutation tests were used to compare healthy and pathological groups. The mean healthy-group MAE was 7.39 deg with a bootstrap 95% CI of [5.84, 9.13], while the pathological-group MAE was 5.20 deg with a bootstrap 95% CI of [4.56, 5.90]. A two-sided permutation test found a significant MAE difference (p = 0.0320), whereas the RMSE difference was not significant at the 0.05 level (p = 0.1004). These results should be interpreted cautiously because detailed pathology and demographic covariates are unavailable.

Insert Table: `revision_tables/healthy_unhealthy_statistics.csv`.

Insert Table: `revision_tables/statistical_tests.csv`.

### F. Robustness to Signal Perturbations

We evaluated the selected four-channel Transformer under test-time perturbations without retraining. Gaussian feature noise had little effect (std = 0.05: MAE = 6.30 deg; std = 0.10: MAE = 6.30 deg). Channel dropout caused larger degradation, increasing MAE to 8.18 deg for RF dropout, 7.63 deg for BF dropout, 7.25 deg for VM dropout, and 8.25 deg for ST dropout. Amplitude scaling also affected performance: 0.8x scaling yielded MAE = 6.95 deg, while 1.2x scaling yielded MAE = 6.62 deg. These findings indicate that the model is relatively stable to small feature noise but remains sensitive to electrode/channel failure, particularly RF and ST loss.

Insert Table: `revision_tables/robustness_summary.csv`.

## V. Discussion

The revised results show that strict validation changes the interpretation of the study. The original overlapping-window result was lower in numerical error, but the revised leakage-aware protocol provides a more credible estimate of subject-specific performance. Under this protocol, using all four available thigh muscles substantially improves performance relative to the two-channel RF/VM setting. This result is physiologically plausible because BF and ST contribute to knee flexion and provide information not captured by knee-extensor channels alone.

The revised RMSE of approximately 9 deg suggests that four-channel sEMG can track the broad temporal pattern of knee flexion-extension during walking, but the error remains too large for high-precision clinical measurement or fine-grained rehabilitation decisions without additional sensing, calibration, or subject-specific adaptation. The model may be more appropriate as a component in coarse movement-intent estimation or trend monitoring than as a replacement for motion-capture or goniometric measurement.

The LOSO result shows that cross-subject generalization remains weak. Therefore, the revised manuscript frames the model as subject-specific and avoids claiming universal deployment across populations. Practical deployment would also require attention to electrode placement, signal quality, channel loss, and online recalibration. The robustness analysis suggests that small feature noise is tolerable, but complete channel dropout can meaningfully degrade performance.

Negative values in the knee-angle signal reflect the goniometer reference orientation and the flexion-extension coordinate convention used during data acquisition. They do not indicate anatomically impossible knee motion. Example prediction figures should explicitly state this convention and identify whether the subject is healthy or pathological.

## VI. Limitations

This study has several limitations. First, the public dataset provides only limited demographic information, so age-, height-, weight-, and pathology-specific analyses cannot be performed. Second, the strongest leakage-safe results remain subject-specific; cross-subject performance is substantially weaker. Third, the revised leakage-safe result does not outperform the original optimistic overlapping-window Table I result. Fourth, robustness testing shows that the model is sensitive to complete channel dropout, so practical deployment would require electrode-quality monitoring, redundancy, or online recalibration. Finally, cycle-level blocked k-fold validation would be valuable in future work once reliable gait-cycle annotations or a validated cycle detector are available.

## VII. Conclusion

This revised study presents a leakage-aware subject-specific evaluation of Transformer-based knee-angle prediction from lower-limb sEMG. By replacing the original overlapping-window hold-out evaluation with purged temporal validation, train-only scaling, validation-only model selection, and explicit leakage auditing, the revised analysis provides a more reliable estimate of model performance. The four-channel RF/BF/VM/ST Transformer achieved MAE = 6.30 deg and RMSE = 9.03 deg across 22 subjects and improved over the corresponding two-channel Transformer under the same strict protocol. However, LOSO results indicate weak cross-subject generalization, and the model should be interpreted as requiring subject-specific calibration. These findings highlight both the promise of multi-channel sEMG for wearable knee-angle estimation and the importance of leakage-aware evaluation in highly overlapping biomedical time-series prediction.

## References to Add or Verify

1. A. J. S. Sanchez and M. A. Sotelo, "EMG Dataset in Lower Limb," UCI Machine Learning Repository, 2014, doi: 10.24432/C5ZW3P.
2. H.-B. Li, X.-R. Guan, Z. Li, K.-F. Zou, and L. He, "Estimation of Knee Joint Angle from Surface EMG Using Multiple Kernels Relevance Vector Regression," Sensors, vol. 23, no. 10, Article 4934, 2023, doi: 10.3390/s23104934.
3. Z. Wang, H. Chen, F. Yang, X. Wang, X. Wu, and C. Chen, "Accurate Prediction of Knee Joint Angles Using a Hybrid CNN-LSTM-Attention Network from Surface Electromyography," in 2024 IEEE International Conference on Robotics and Biomimetics (ROBIO), 2024, doi: 10.1109/ROBIO64047.2024.10907528.
4. Z. Wang, C. Chen, H. Chen, Y. Zhou, X. Wang, and X. Wu, "Dual Transformer Network for Predicting Joint Angles and Torques From Multi-Channel EMG Signals in the Lower Limbs," IEEE Journal of Biomedical and Health Informatics, 2025, doi: 10.1109/JBHI.2025.3555255.
