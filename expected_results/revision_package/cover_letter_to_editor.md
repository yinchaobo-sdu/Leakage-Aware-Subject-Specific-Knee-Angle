# Cover Letter for Revised Submission

Dear Dr. Eid,

Thank you for considering our manuscript, Sensors-91956-2025, "Transformer-Based Knee Angle Prediction Using Surface Electromyography," for IEEE Sensors Journal. We are grateful to you and the reviewers for the detailed comments. We have thoroughly reworked the manuscript and submit a substantially revised version for further consideration.

The revised manuscript directly addresses the main methodological concern raised by Reviewer 1: the risk of over-optimistic performance from highly overlapping sliding windows. We no longer use the original overlapping-window 70/30 split as the main evidence. Instead, the revised manuscript introduces a leakage-aware subject-specific evaluation protocol with purged temporal train/validation/test splits, train-only feature and target scaling, validation-only model selection, and an explicit leakage audit.

Major revisions include:

1. Clarifying the study aim and population as subject-specific knee-angle prediction in 22 male subjects from the public UCI Lower Limb EMG dataset, including 11 healthy and 11 pathological subjects.
2. Extending the model input from two sEMG channels (RF/VM) to four physiologically relevant thigh channels (RF/BF/VM/ST).
3. Adding a leakage-aware validation protocol and reporting MAE/RMSE as the primary metrics.
4. Adding stronger baselines, including random forest, RBF-SVR, XGBoost, MLP, and a leave-one-subject-out ridge baseline.
5. Adding a multiscale ablation, feature-correlation analysis, robustness tests, subject-level confidence intervals, non-parametric statistical tests, and per-subject window counts.
6. Expanding the discussion, clinical interpretation, limitations, and quantitative literature comparison.

Under the revised leakage-aware protocol, the validation-selected four-channel Transformer achieved MAE = 6.30 deg and RMSE = 9.03 deg across 22 subjects. We explicitly narrowed the claims and now present the study as a leakage-aware subject-specific evaluation rather than as a universal cross-subject predictor. The revised manuscript also states that leave-one-subject-out performance remains substantially weaker, indicating the need for subject-specific calibration.

We have prepared a point-by-point response to every reviewer comment and highlighted/underlined the changes in the revised manuscript as requested. We hope that the revision resolves the reviewers' concerns and provides a technically stronger and more transparent contribution.

Sincerely,

Guiqian Gao, Zijing Yang, and Huanghe Zhang
