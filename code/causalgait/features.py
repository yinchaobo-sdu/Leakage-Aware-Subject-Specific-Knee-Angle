from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import ensure_dir


ORIGINAL_FEATURE_NAMES_PER_CHANNEL = [
    "max",
    "variance",
    "mean_absolute_value",
    "mean",
    "min",
    "rms",
    "zero_crossings",
    "std",
]
EXTENDED_FEATURE_NAMES_PER_CHANNEL = ORIGINAL_FEATURE_NAMES_PER_CHANNEL + [
    "waveform_length",
    "integrated_emg",
    "slope_sign_changes",
    "skewness",
    "kurtosis",
    "spectral_centroid",
]
FEATURE_NAMES_PER_CHANNEL = ORIGINAL_FEATURE_NAMES_PER_CHANNEL
DEFAULT_CHANNELS = ("RF", "VM")
FEATURE_COLUMNS = [f"{channel}_{name}" for channel in DEFAULT_CHANNELS for name in ORIGINAL_FEATURE_NAMES_PER_CHANNEL]


def normalize_channels(channels: Sequence[str] | str) -> List[str]:
    if isinstance(channels, str):
        raw_channels = [part.strip() for part in channels.split(",") if part.strip()]
    else:
        raw_channels = [str(part).strip() for part in channels if str(part).strip()]
    if not raw_channels:
        raise ValueError("At least one sEMG channel must be selected.")
    channels_out = [channel.upper() for channel in raw_channels]
    valid = {"RF", "BF", "VM", "ST"}
    invalid = [channel for channel in channels_out if channel not in valid]
    if invalid:
        raise ValueError(f"Unsupported channel(s): {', '.join(invalid)}. Valid channels: RF, BF, VM, ST.")
    if len(set(channels_out)) != len(channels_out):
        raise ValueError(f"Duplicate channel selection is not allowed: {channels_out}.")
    return channels_out


def feature_names_for_set(feature_set: str) -> List[str]:
    feature_set = feature_set.lower()
    if feature_set == "original":
        return list(ORIGINAL_FEATURE_NAMES_PER_CHANNEL)
    if feature_set == "extended":
        return list(EXTENDED_FEATURE_NAMES_PER_CHANNEL)
    raise ValueError(f"Unsupported feature set: {feature_set}")


def feature_columns_for_channels(channels: Sequence[str] | str, feature_set: str = "original") -> List[str]:
    normalized = normalize_channels(channels)
    names = feature_names_for_set(feature_set)
    return [f"{channel}_{name}" for channel in normalized for name in names]


@dataclass
class FeatureBundle:
    features: np.ndarray
    angles: np.ndarray
    feature_columns: List[str]
    source_rows: int
    total_rows: int
    skipped_rows: int


@dataclass
class MinMaxScaler:
    min_: np.ndarray
    max_: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "MinMaxScaler":
        if values.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got shape {values.shape}.")
        return cls(min_=values.min(axis=0), max_=values.max(axis=0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        denom = self.max_ - self.min_
        denom = np.where(denom == 0, 1.0, denom)
        scaled = (values - self.min_) / denom
        return np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)

    def to_dict(self) -> Dict[str, List[float] | str]:
        return {
            "type": "minmax",
            "min": self.min_.astype(float).tolist(),
            "max": self.max_.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, List[float]]) -> "MinMaxScaler":
        return cls(
            min_=np.asarray(payload["min"], dtype=np.float32),
            max_=np.asarray(payload["max"], dtype=np.float32),
        )


@dataclass
class StandardScaler:
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        if values.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got shape {values.shape}.")
        scale = values.std(axis=0)
        scale = np.where(scale == 0, 1.0, scale)
        return cls(mean_=values.mean(axis=0), scale_=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean_) / self.scale_).astype(np.float32, copy=False)

    def to_dict(self) -> Dict[str, List[float] | str]:
        return {
            "type": "standard",
            "mean": self.mean_.astype(float).tolist(),
            "scale": self.scale_.astype(float).tolist(),
        }


@dataclass
class TargetScaler:
    mode: str
    mean_: float = 0.0
    scale_: float = 1.0

    @classmethod
    def fit(cls, values: np.ndarray, mode: str) -> "TargetScaler":
        mode = mode.lower()
        if mode == "none":
            return cls(mode="none", mean_=0.0, scale_=1.0)
        if mode != "standard":
            raise ValueError(f"Unsupported target scaler: {mode}")
        values = np.asarray(values, dtype=np.float32)
        scale = float(values.std())
        if scale == 0.0:
            scale = 1.0
        return cls(mode="standard", mean_=float(values.mean()), scale_=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if self.mode == "none":
            return values.astype(np.float32, copy=False)
        return ((values - self.mean_) / self.scale_).astype(np.float32, copy=False)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if self.mode == "none":
            return values.astype(np.float32, copy=False)
        return (values * self.scale_ + self.mean_).astype(np.float32, copy=False)

    def to_dict(self) -> Dict[str, float | str]:
        return {"type": self.mode, "mean": float(self.mean_), "scale": float(self.scale_)}


def _sliding_window_view_1d(signal: np.ndarray, window: int) -> np.ndarray:
    n_windows = len(signal) - window + 1
    if n_windows <= 0:
        raise ValueError(f"Signal length {len(signal)} is shorter than feature window {window}.")
    stride = signal.strides[0]
    return np.lib.stride_tricks.as_strided(signal, shape=(n_windows, window), strides=(stride, stride))


def _spectral_centroid(windows: np.ndarray, sample_rate: float = 1000.0) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(windows, axis=1)).astype(np.float32, copy=False)
    freqs = np.fft.rfftfreq(windows.shape[1], d=1.0 / sample_rate).astype(np.float32, copy=False)
    denom = spectrum.sum(axis=1)
    denom = np.where(denom == 0, 1.0, denom)
    return (spectrum * freqs).sum(axis=1) / denom


def _windowed_channel_features(signal: np.ndarray, window: int, step: int, feature_set: str) -> np.ndarray:
    if len(signal) < window:
        raise ValueError(f"Signal length {len(signal)} is shorter than feature window {window}.")

    windows = _sliding_window_view_1d(signal.astype(np.float32, copy=False), window)[::step]
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    features = [
        windows.max(axis=1),
        windows.var(axis=1),
        np.abs(windows).mean(axis=1),
        mean,
        windows.min(axis=1),
        np.sqrt(np.square(windows).mean(axis=1)),
        np.diff(np.signbit(windows), axis=1).sum(axis=1).astype(np.float32),
        std,
    ]

    if feature_set.lower() == "extended":
        diffs = np.diff(windows, axis=1)
        std_safe = np.where(std == 0, 1.0, std)
        centered = windows - mean[:, None]
        features.extend(
            [
                np.abs(diffs).sum(axis=1),
                np.abs(windows).sum(axis=1),
                ((diffs[:, :-1] * diffs[:, 1:]) < 0).sum(axis=1).astype(np.float32),
                (np.power(centered, 3).mean(axis=1) / np.power(std_safe, 3)).astype(np.float32),
                (np.power(centered, 4).mean(axis=1) / np.power(std_safe, 4)).astype(np.float32),
                _spectral_centroid(windows),
            ]
        )
    elif feature_set.lower() != "original":
        raise ValueError(f"Unsupported feature set: {feature_set}")
    return np.stack(features, axis=1).astype(np.float32)


def compute_multichannel_features(
    signals: Dict[str, np.ndarray],
    fx: np.ndarray,
    channels: Sequence[str] | str = DEFAULT_CHANNELS,
    feature_set: str = "original",
    window: int = 250,
    step: int = 1,
    total_rows: Optional[int] = None,
    skipped_rows: int = 0,
) -> FeatureBundle:
    channels_norm = normalize_channels(channels)
    if window <= 0 or step <= 0:
        raise ValueError("Feature window and step must be positive.")
    missing = [channel for channel in channels_norm if channel not in signals]
    if missing:
        raise ValueError(f"Missing channel data for: {', '.join(missing)}")
    lengths = [len(signals[channel]) for channel in channels_norm] + [len(fx)]
    if len(set(lengths)) != 1:
        raise ValueError(f"All selected channels and FX lengths must match: {lengths}.")

    channel_features = [
        _windowed_channel_features(signals[channel], window, step, feature_set) for channel in channels_norm
    ]
    n_features = min([len(values) for values in channel_features] + [len(fx[window - 1 :: step])])
    features = np.concatenate([values[:n_features] for values in channel_features], axis=1)
    angles = fx[window - 1 :: step][:n_features].astype(np.float32, copy=False)
    return FeatureBundle(
        features=features.astype(np.float32, copy=False),
        angles=angles,
        feature_columns=feature_columns_for_channels(channels_norm, feature_set),
        source_rows=len(fx),
        total_rows=len(fx) if total_rows is None else int(total_rows),
        skipped_rows=int(skipped_rows),
    )


def compute_two_channel_features(
    rf: np.ndarray,
    vm: np.ndarray,
    fx: np.ndarray,
    window: int = 250,
    step: int = 1,
    total_rows: Optional[int] = None,
    skipped_rows: int = 0,
) -> FeatureBundle:
    return compute_multichannel_features(
        signals={"RF": rf, "VM": vm},
        fx=fx,
        channels=DEFAULT_CHANNELS,
        feature_set="original",
        window=window,
        step=step,
        total_rows=total_rows,
        skipped_rows=skipped_rows,
    )


def cache_path(
    cache_dir: Path,
    source_path: Path,
    window: int,
    step: int,
    channels: Sequence[str] | str = DEFAULT_CHANNELS,
    feature_set: str = "original",
) -> Path:
    channel_key = "-".join(normalize_channels(channels))
    name = f"{source_path.stem}_ch{channel_key}_set{feature_set.lower()}_win{window}_step{step}.npz"
    return cache_dir / name


def load_or_compute_features(
    source_path: Path,
    signals: Dict[str, np.ndarray],
    fx: np.ndarray,
    cache_dir: Optional[Path],
    window: int,
    step: int,
    channels: Sequence[str] | str = DEFAULT_CHANNELS,
    feature_set: str = "original",
    total_rows: Optional[int] = None,
    skipped_rows: int = 0,
    no_cache: bool = False,
) -> FeatureBundle:
    channels_norm = normalize_channels(channels)
    feature_set = feature_set.lower()
    path = cache_path(cache_dir, source_path, window, step, channels_norm, feature_set) if cache_dir else None
    source_mtime = source_path.stat().st_mtime_ns

    if path is not None and path.exists() and not no_cache:
        try:
            with np.load(path, allow_pickle=False) as cached:
                meta = json.loads(str(cached["meta"].item()))
                if (
                    meta.get("cache_version") == 3
                    and meta.get("source_mtime") == source_mtime
                    and meta.get("channels") == channels_norm
                    and meta.get("feature_set") == feature_set
                ):
                    return FeatureBundle(
                        features=cached["features"].astype(np.float32, copy=False),
                        angles=cached["angles"].astype(np.float32, copy=False),
                        feature_columns=list(meta.get("feature_columns", feature_columns_for_channels(channels_norm, feature_set))),
                        source_rows=int(meta["source_rows"]),
                        total_rows=int(meta["total_rows"]),
                        skipped_rows=int(meta["skipped_rows"]),
                    )
        except (KeyError, ValueError, json.JSONDecodeError):
            pass

    bundle = compute_multichannel_features(
        signals=signals,
        fx=fx,
        channels=channels_norm,
        feature_set=feature_set,
        window=window,
        step=step,
        total_rows=total_rows,
        skipped_rows=skipped_rows,
    )

    if path is not None and not no_cache:
        ensure_dir(path.parent)
        meta = {
            "cache_version": 3,
            "source": str(source_path),
            "source_mtime": source_mtime,
            "source_rows": bundle.source_rows,
            "total_rows": bundle.total_rows,
            "skipped_rows": bundle.skipped_rows,
            "window": window,
            "step": step,
            "channels": channels_norm,
            "feature_set": feature_set,
            "feature_columns": bundle.feature_columns,
        }
        np.savez_compressed(
            path,
            features=bundle.features,
            angles=bundle.angles,
            meta=np.asarray(json.dumps(meta)),
        )
    return bundle


def train_test_scale(
    features: np.ndarray,
    angles: np.ndarray,
    seq_len: int,
    train_ratio: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, np.ndarray, np.ndarray]:
    scaled_features, scaled_angles, scaler, train_indices, _, test_indices = split_and_scale(
        features=features,
        angles=angles,
        seq_len=seq_len,
        split_mode="temporal",
        train_ratio=train_ratio,
        val_ratio=0.0,
        test_ratio=1.0 - train_ratio,
        scaler_scope="train",
        seed=42,
        feature_scaler="minmax",
        forecast_horizon=1,
    )
    return scaled_features, scaled_angles, scaler, train_indices, test_indices


def split_and_scale(
    features: np.ndarray,
    angles: np.ndarray,
    seq_len: int,
    split_mode: str = "temporal",
    train_ratio: float = 0.7,
    val_ratio: float = 0.0,
    test_ratio: float = 0.3,
    scaler_scope: str = "train",
    seed: int = 42,
    feature_scaler: str = "minmax",
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler | StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    if seq_len <= 0:
        raise ValueError("Sequence length must be positive.")
    if forecast_horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")
    n_sequences = len(features) - seq_len - (forecast_horizon - 1)
    if n_sequences < 2:
        raise ValueError(f"Need at least 2 sequence samples, got {n_sequences}.")

    split_mode = split_mode.lower()
    scaler_scope = scaler_scope.lower()
    feature_scaler = feature_scaler.lower()
    if scaler_scope not in {"train", "full"}:
        raise ValueError(f"Unsupported scaler scope: {scaler_scope}")
    if feature_scaler not in {"minmax", "standard"}:
        raise ValueError(f"Unsupported feature scaler: {feature_scaler}")

    if split_mode == "temporal":
        n_train = int(n_sequences * train_ratio)
        n_train = max(1, min(n_train, n_sequences - 1))
        remaining = n_sequences - n_train
        n_val = int(remaining * val_ratio / max(val_ratio + test_ratio, 1e-12)) if val_ratio > 0 else 0
        n_val = max(0, min(n_val, remaining - 1))
        train_indices = np.arange(0, n_train, dtype=np.int64)
        val_indices = np.arange(n_train, n_train + n_val, dtype=np.int64)
        test_indices = np.arange(n_train + n_val, n_sequences, dtype=np.int64)
    elif split_mode == "purged_temporal":
        ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
        if np.any(ratios < 0) or ratios.sum() <= 0:
            raise ValueError("Split ratios must be non-negative and sum to a positive value.")
        ratios = ratios / ratios.sum()
        n_gaps = 2 if ratios[1] > 0 else 1
        purge_gap = seq_len + forecast_horizon - 1
        usable = n_sequences - n_gaps * purge_gap
        min_required = 3 if ratios[1] > 0 else 2
        if usable < min_required:
            raise ValueError(
                f"Not enough sequence samples ({n_sequences}) for purged_temporal split "
                f"with seq_len={seq_len} and forecast_horizon={forecast_horizon}."
            )
        n_train = max(1, int(usable * ratios[0]))
        n_val = int(usable * ratios[1])
        if ratios[1] > 0:
            n_val = max(1, n_val)
        n_test = usable - n_train - n_val
        if n_test < 1:
            deficit = 1 - n_test
            if n_val > deficit and ratios[1] > 0:
                n_val -= deficit
            else:
                n_train = max(1, n_train - deficit)
            n_test = usable - n_train - n_val
        train_indices = np.arange(0, n_train, dtype=np.int64)
        if n_val > 0:
            val_start = n_train + purge_gap
            test_start = val_start + n_val + purge_gap
            val_indices = np.arange(val_start, val_start + n_val, dtype=np.int64)
        else:
            test_start = n_train + purge_gap
            val_indices = np.empty((0,), dtype=np.int64)
        test_indices = np.arange(test_start, test_start + n_test, dtype=np.int64)
    elif split_mode == "random_overlap":
        ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
        if np.any(ratios < 0) or ratios.sum() <= 0:
            raise ValueError("Split ratios must be non-negative and sum to a positive value.")
        ratios = ratios / ratios.sum()
        rng = np.random.default_rng(seed)
        indices = np.arange(n_sequences, dtype=np.int64)
        rng.shuffle(indices)
        n_train = max(1, int(n_sequences * ratios[0]))
        n_val = int(n_sequences * ratios[1])
        if ratios[1] > 0:
            n_val = max(1, n_val)
        n_train = min(n_train, n_sequences - 2 if n_val > 0 else n_sequences - 1)
        n_val = min(n_val, n_sequences - n_train - 1)
        train_indices = np.sort(indices[:n_train])
        val_indices = np.sort(indices[n_train : n_train + n_val])
        test_indices = np.sort(indices[n_train + n_val :])
    else:
        raise ValueError(f"Unsupported split mode: {split_mode}")

    if len(test_indices) == 0:
        raise ValueError("Split produced no test samples.")

    if scaler_scope == "full":
        fit_rows = features
    else:
        fit_rows = _feature_rows_for_sequences(features, train_indices, seq_len)
    scaler = MinMaxScaler.fit(fit_rows) if feature_scaler == "minmax" else StandardScaler.fit(fit_rows)
    scaled_features = scaler.transform(features)
    return (
        scaled_features,
        angles.astype(np.float32, copy=False),
        scaler,
        train_indices,
        val_indices,
        test_indices,
    )


def _feature_rows_for_sequences(features: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
    if len(indices) == 0:
        raise ValueError("Cannot fit scaler with no train indices.")
    mask = np.zeros(len(features), dtype=bool)
    for start in indices:
        mask[int(start) : int(start) + seq_len] = True
    return features[mask]
