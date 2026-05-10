from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from causalgait.data import SequenceDataset, read_emg_csv
from causalgait.data import SubjectRecord
from causalgait.features import (
    TargetScaler,
    compute_multichannel_features,
    compute_two_channel_features,
    feature_columns_for_channels,
    split_and_scale,
    train_test_scale,
)
from causalgait.models import CausalGaitNet, MLPRegressor, MultiScaleTransformerRegressor
from causalgait.train import PipelineConfig, PreparedSubject, compute_metrics, leakage_audit_for_subject
from run_enhanced_publishable_experiment import target_matrix
from run_pipeline import apply_preset, validate_args
from run_optimized_experiment import nearest_index_prediction


def test_feature_shape() -> None:
    rf = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    vm = np.linspace(1.0, -1.0, 20, dtype=np.float32)
    fx = np.arange(20, dtype=np.float32)
    bundle = compute_two_channel_features(rf, vm, fx, window=5, step=2)
    assert bundle.features.shape == (8, 16), bundle.features.shape
    assert bundle.angles.tolist() == [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]


def test_multichannel_extended_features() -> None:
    signals = {
        "RF": np.linspace(-1.0, 1.0, 20, dtype=np.float32),
        "BF": np.linspace(1.0, 2.0, 20, dtype=np.float32),
        "VM": np.linspace(1.0, -1.0, 20, dtype=np.float32),
        "ST": np.sin(np.linspace(0.0, 2.0, 20)).astype(np.float32),
    }
    fx = np.arange(20, dtype=np.float32)
    bundle = compute_multichannel_features(signals, fx, channels="RF,BF,VM,ST", feature_set="extended", window=5, step=2)
    assert bundle.features.shape == (8, 56), bundle.features.shape
    assert len(bundle.feature_columns) == 56
    assert feature_columns_for_channels("RF,BF", "original")[:2] == ["RF_max", "RF_variance"]
    try:
        compute_multichannel_features(signals, fx, channels="RF,BAD", feature_set="extended", window=5, step=2)
    except ValueError as exc:
        assert "Unsupported channel" in str(exc)
    else:
        raise AssertionError("Invalid channel was accepted.")


def test_four_channel_csv_read() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "toy.csv"
        path.write_text(
            "RF(mV),BF(mV),VM(mV),ST(mV),FX(deg)\n"
            "1,2,3,4,5\n"
            "6,7,8,9,10\n"
            "malformed,row\n",
            encoding="utf-8",
        )
        raw = read_emg_csv(path)
        assert raw.skipped_rows == 1
        assert raw.channels["RF"].tolist() == [1.0, 6.0]
        assert raw.channels["BF"].tolist() == [2.0, 7.0]
        assert raw.channels["VM"].tolist() == [3.0, 8.0]
        assert raw.channels["ST"].tolist() == [4.0, 9.0]


def test_scaling_and_sequence_dataset() -> None:
    features = np.arange(12 * 16, dtype=np.float32).reshape(12, 16)
    angles = np.arange(12, dtype=np.float32)
    scaled, scaled_angles, scaler, train_idx, test_idx = train_test_scale(features, angles, seq_len=4, train_ratio=0.5)
    assert scaled.shape == features.shape
    assert len(train_idx) == 4
    assert len(test_idx) == 4
    assert np.allclose(scaled_angles, angles)
    assert len(scaler.to_dict()["min"]) == 16
    ds = SequenceDataset(scaled, scaled_angles, test_idx, seq_len=4)
    x, y = ds[0]
    assert x.shape == (4, 16)
    assert float(y) == 8.0


def test_random_split_and_full_scaler() -> None:
    features = np.arange(40 * 16, dtype=np.float32).reshape(40, 16)
    angles = np.arange(40, dtype=np.float32)
    scaled, _, scaler, train_idx, val_idx, test_idx = split_and_scale(
        features,
        angles,
        seq_len=4,
        split_mode="random_overlap",
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        scaler_scope="full",
        seed=7,
    )
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    assert len(all_idx) == len(set(all_idx.tolist()))
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(features) - 4
    assert np.allclose(scaler.min_, features.min(axis=0))
    assert np.allclose(scaler.max_, features.max(axis=0))
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0


def test_purged_temporal_split_and_train_scaler() -> None:
    features = np.arange(100 * 16, dtype=np.float32).reshape(100, 16)
    features[80:] += 100000.0
    angles = np.arange(100, dtype=np.float32)
    _, _, scaler, train_idx, val_idx, test_idx = split_and_scale(
        features,
        angles,
        seq_len=8,
        split_mode="purged_temporal",
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        scaler_scope="train",
        seed=7,
    )
    assert val_idx[0] - train_idx[-1] >= 8
    assert test_idx[0] - val_idx[-1] >= 8
    train_mask = np.zeros(len(features), dtype=bool)
    for start in train_idx:
        train_mask[start : start + 8] = True
    assert np.allclose(scaler.min_, features[train_mask].min(axis=0))
    assert np.allclose(scaler.max_, features[train_mask].max(axis=0))
    assert not np.allclose(scaler.max_, features.max(axis=0))


def test_standard_scaler_and_target_scaler() -> None:
    features = np.arange(120 * 4, dtype=np.float32).reshape(120, 4)
    features[90:] += 10000.0
    angles = np.arange(120, dtype=np.float32)
    scaled, _, scaler, train_idx, _, test_idx = split_and_scale(
        features,
        angles,
        seq_len=8,
        split_mode="purged_temporal",
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        scaler_scope="train",
        feature_scaler="standard",
        forecast_horizon=20,
        seed=7,
    )
    assert hasattr(scaler, "mean_")
    assert abs(float(scaled[train_idx[0] : train_idx[0] + 8].mean())) < 2.0
    assert int(test_idx[-1] + 8 + 20 - 1) < len(angles)
    labels = target_matrix(angles, train_idx, seq_len=8, horizons=[1, 5, 20])
    target_scaler = TargetScaler.fit(labels.reshape(-1), "standard")
    restored = target_scaler.inverse_transform(target_scaler.transform(labels))
    assert np.allclose(restored, labels)


def test_leakage_audit_marks_purged_temporal_safe() -> None:
    features = np.arange(100 * 16, dtype=np.float32).reshape(100, 16)
    angles = np.arange(100, dtype=np.float32)
    scaled, _, scaler, train_idx, val_idx, test_idx = split_and_scale(
        features,
        angles,
        seq_len=8,
        split_mode="purged_temporal",
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        scaler_scope="train",
        seed=7,
    )
    cfg = PipelineConfig(
        preset="publishable",
        evaluation_mode="publishable",
        split_mode="purged_temporal",
        scaler_scope="train",
        seq_len=8,
    )
    prepared = PreparedSubject(
        record=SubjectRecord("toy", "healthy", __file__),
        train_dataset=SequenceDataset(scaled, angles, train_idx, 8),
        val_dataset=SequenceDataset(scaled, angles, val_idx, 8),
        test_dataset=SequenceDataset(scaled, angles, test_idx, 8),
        scaler=scaler,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        n_feature_rows=len(features),
        feature_columns=[f"f{i}" for i in range(16)],
        source_rows=len(features),
        total_rows=len(features),
        skipped_rows=0,
    )
    audit = leakage_audit_for_subject(prepared, cfg)
    assert audit["leakage_safe_for_publication"] is True
    assert audit["pairwise"]["train_val"]["input_windows_overlap"] is False
    assert audit["pairwise"]["train_test"]["input_windows_overlap"] is False
    assert audit["scaler"]["fit_row_max"] < int(test_idx[0])


def test_publishable_rejects_leaky_arguments() -> None:
    class Args:
        preset = "publishable"
        split_mode = "random_overlap"
        scaler_scope = None
        scheduler_type = None
        loss = None
        val_ratio = None
        test_ratio = None
        train_ratio = 0.7
        epochs = 30
        lr = 1e-3
        d_model = 128
        heads = 4
        ff_dim = 256
        encoder_layers = 1
        dropout = 0.1
        scales = "1,2,4"
        grad_clip = 0.0
        early_stopping_patience = 0
        best_checkpoint = False

    args = apply_preset(Args())
    try:
        validate_args(args)
    except ValueError as exc:
        assert "purged_temporal" in str(exc)
    else:
        raise AssertionError("publishable preset accepted random_overlap split")


def test_nearest_index_prediction() -> None:
    source_idx = np.asarray([0, 10, 20])
    source_values = np.asarray([1.0, 5.0, 9.0], dtype=np.float32)
    target_idx = np.asarray([1, 9, 16, 30])
    pred = nearest_index_prediction(source_idx, source_values, target_idx)
    assert pred.tolist() == [1.0, 5.0, 9.0, 9.0]


def test_model_forward() -> None:
    x = torch.randn(2, 16, 16)
    transformer = MultiScaleTransformerRegressor(
        seq_len=16,
        feature_dim=16,
        scales=(1, 2),
        patch_size=4,
        d_model=32,
        nhead=4,
        num_layers=1,
        ff_dim=64,
    )
    mlp = MLPRegressor(seq_len=16, feature_dim=16)
    enhanced = CausalGaitNet(
        seq_len=16,
        feature_dim=16,
        scales=(1, 2),
        patch_size=4,
        d_model=32,
        nhead=4,
        num_layers=1,
        ff_dim=64,
        output_dim=3,
    )
    assert transformer(x).shape == (2,)
    assert mlp(x).shape == (2,)
    assert enhanced(x).shape == (2, 3)


def test_metrics() -> None:
    y_true = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
    y_pred = np.asarray([2.0, 1.0, 4.0], dtype=np.float32)
    metrics = compute_metrics(y_true, y_pred)
    assert abs(metrics["me"] - 0.0) < 1e-6
    assert abs(metrics["mae"] - (2.0 / 3.0)) < 1e-6


def main() -> None:
    test_feature_shape()
    test_multichannel_extended_features()
    test_four_channel_csv_read()
    test_scaling_and_sequence_dataset()
    test_random_split_and_full_scaler()
    test_purged_temporal_split_and_train_scaler()
    test_standard_scaler_and_target_scaler()
    test_leakage_audit_marks_purged_temporal_safe()
    test_publishable_rejects_leaky_arguments()
    test_nearest_index_prediction()
    test_model_forward()
    test_metrics()
    print("All tests passed.")


if __name__ == "__main__":
    main()
