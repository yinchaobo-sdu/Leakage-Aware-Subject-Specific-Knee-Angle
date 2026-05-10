from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import SequenceDataset, SubjectRecord, load_subject_features
from .features import (
    DEFAULT_CHANNELS,
    FEATURE_COLUMNS,
    MinMaxScaler,
    StandardScaler,
    feature_names_for_set,
    normalize_channels,
    split_and_scale,
)
from .models import CausalGaitNet, MLPRegressor, MultiScaleTransformerRegressor
from .utils import ensure_dir, save_json, stable_seed, write_dict_csv


@dataclass
class PipelineConfig:
    preset: str = "strict"
    evaluation_mode: str = "strict"
    split_mode: str = "temporal"
    scaler_scope: str = "train"
    feature_window: int = 250
    feature_step: int = 1
    channels: Sequence[str] = DEFAULT_CHANNELS
    feature_set: str = "original"
    feature_scaler: str = "minmax"
    feature_dim: int = len(FEATURE_COLUMNS)
    forecast_horizon: int = 1
    seq_len: int = 128
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 0.0
    early_stopping_patience: int = 0
    scheduler_step: int = 10
    scheduler_gamma: float = 0.5
    scheduler_type: str = "step"
    loss: str = "mse"
    train_ratio: float = 0.7
    val_ratio: float = 0.0
    test_ratio: float = 0.3
    best_checkpoint: bool = False
    seed: int = 42
    seed_label: str = "42"
    num_workers: int = 0
    d_model: int = 128
    patch_size: int = 8
    encoder_layers: int = 1
    heads: int = 4
    ff_dim: int = 256
    dropout: float = 0.1
    scales: Sequence[int] = (1, 2, 4)
    plots: bool = True
    plot_max_points: int = 3000
    no_cache: bool = False


@dataclass
class PreparedSubject:
    record: SubjectRecord
    train_dataset: SequenceDataset
    val_dataset: Optional[SequenceDataset]
    test_dataset: SequenceDataset
    scaler: MinMaxScaler | StandardScaler
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    n_feature_rows: int
    feature_columns: List[str]
    source_rows: int
    total_rows: int
    skipped_rows: int


def prepare_subject(record: SubjectRecord, cfg: PipelineConfig, cache_dir: Path) -> PreparedSubject:
    bundle = load_subject_features(
        record=record,
        cache_dir=cache_dir,
        feature_window=cfg.feature_window,
        feature_step=cfg.feature_step,
        no_cache=cfg.no_cache,
        channels=cfg.channels,
        feature_set=cfg.feature_set,
    )
    cfg.feature_dim = len(bundle.feature_columns)
    scaled_features, angles, scaler, train_indices, val_indices, test_indices = split_and_scale(
        bundle.features,
        bundle.angles,
        seq_len=cfg.seq_len,
        split_mode=cfg.split_mode,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        scaler_scope=cfg.scaler_scope,
        seed=stable_seed(cfg.seed, record.subject, cfg.split_mode),
        feature_scaler=cfg.feature_scaler,
        forecast_horizon=cfg.forecast_horizon,
    )
    val_dataset = SequenceDataset(scaled_features, angles, val_indices, cfg.seq_len) if len(val_indices) else None
    return PreparedSubject(
        record=record,
        train_dataset=SequenceDataset(scaled_features, angles, train_indices, cfg.seq_len),
        val_dataset=val_dataset,
        test_dataset=SequenceDataset(scaled_features, angles, test_indices, cfg.seq_len),
        scaler=scaler,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        n_feature_rows=len(scaled_features),
        feature_columns=list(bundle.feature_columns),
        source_rows=bundle.source_rows,
        total_rows=bundle.total_rows,
        skipped_rows=bundle.skipped_rows,
    )


def build_pipeline_model(model_name: str, cfg: PipelineConfig) -> nn.Module:
    if model_name == "transformer":
        return MultiScaleTransformerRegressor(
            seq_len=cfg.seq_len,
            feature_dim=cfg.feature_dim,
            scales=cfg.scales,
            patch_size=cfg.patch_size,
            d_model=cfg.d_model,
            nhead=cfg.heads,
            num_layers=cfg.encoder_layers,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
        )
    if model_name == "mlp":
        return MLPRegressor(seq_len=cfg.seq_len, feature_dim=cfg.feature_dim)
    if model_name == "causalgaitnet":
        return CausalGaitNet(
            seq_len=cfg.seq_len,
            feature_dim=cfg.feature_dim,
            scales=cfg.scales,
            patch_size=cfg.patch_size,
            d_model=cfg.d_model,
            nhead=cfg.heads,
            num_layers=cfg.encoder_layers,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
            output_dim=cfg.forecast_horizon,
        )
    raise ValueError(f"Unsupported model {model_name}.")


def make_loader(dataset: SequenceDataset, cfg: PipelineConfig, shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator if shuffle else None,
        drop_last=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float = 0.0,
) -> float:
    model.train()
    losses: List[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        pred = model(x).detach().cpu().numpy()
        preds.append(pred)
        targets.append(y.numpy())
    return {
        "pred": np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32),
        "true": np.concatenate(targets, axis=0) if targets else np.empty((0,), dtype=np.float32),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    error = y_pred - y_true
    return {
        "me": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(np.mean(np.square(error)))),
    }


def _input_row_bounds(indices: np.ndarray, seq_len: int, forecast_horizon: int = 1) -> Dict[str, Optional[int]]:
    if len(indices) == 0:
        return {"start_min": None, "start_max": None, "input_min": None, "input_max": None, "label_min": None, "label_max": None}
    starts = indices.astype(np.int64, copy=False)
    return {
        "start_min": int(starts.min()),
        "start_max": int(starts.max()),
        "input_min": int(starts.min()),
        "input_max": int(starts.max() + seq_len - 1),
        "label_min": int(starts.min() + seq_len),
        "label_max": int(starts.max() + seq_len + forecast_horizon - 1),
    }


def _min_start_distance(left: np.ndarray, right: np.ndarray) -> Optional[int]:
    if len(left) == 0 or len(right) == 0:
        return None
    left_sorted = np.sort(left.astype(np.int64, copy=False))
    right_sorted = np.sort(right.astype(np.int64, copy=False))
    positions = np.searchsorted(right_sorted, left_sorted)
    distances: List[int] = []
    valid_right = positions < len(right_sorted)
    if np.any(valid_right):
        distances.extend(np.abs(left_sorted[valid_right] - right_sorted[positions[valid_right]]).astype(int).tolist())
    valid_left = positions > 0
    if np.any(valid_left):
        distances.extend(np.abs(left_sorted[valid_left] - right_sorted[positions[valid_left] - 1]).astype(int).tolist())
    return int(min(distances)) if distances else None


def leakage_audit_for_subject(prepared: PreparedSubject, cfg: PipelineConfig) -> Dict[str, Any]:
    subsets = {
        "train": prepared.train_indices,
        "val": prepared.val_indices,
        "test": prepared.test_indices,
    }
    subset_report: Dict[str, Any] = {}
    for name, indices in subsets.items():
        subset_report[name] = {"n_sequences": int(len(indices)), **_input_row_bounds(indices, cfg.seq_len, cfg.forecast_horizon)}

    pair_report: Dict[str, Any] = {}
    for left_name, right_name in [("train", "val"), ("train", "test"), ("val", "test")]:
        distance = _min_start_distance(subsets[left_name], subsets[right_name])
        pair_report[f"{left_name}_{right_name}"] = {
            "min_start_distance": distance,
            "input_windows_overlap": bool(distance is not None and distance < cfg.seq_len),
        }

    if cfg.scaler_scope == "full":
        scaler_rows = np.arange(prepared.n_feature_rows, dtype=np.int64)
    else:
        mask = np.zeros(prepared.n_feature_rows, dtype=bool)
        for start in prepared.train_indices:
            mask[int(start) : int(start) + cfg.seq_len] = True
        scaler_rows = np.flatnonzero(mask)

    scaler_report = {
        "scope": cfg.scaler_scope,
        "fit_row_count": int(len(scaler_rows)),
        "fit_row_min": int(scaler_rows.min()) if len(scaler_rows) else None,
        "fit_row_max": int(scaler_rows.max()) if len(scaler_rows) else None,
    }
    has_overlap = any(item["input_windows_overlap"] for item in pair_report.values())
    leakage_safe = bool(cfg.split_mode == "purged_temporal" and cfg.scaler_scope == "train" and not has_overlap)
    return {
        "subject": prepared.record.subject,
        "status": prepared.record.status,
        "evaluation_mode": cfg.evaluation_mode,
        "split_mode": cfg.split_mode,
        "scaler_scope": cfg.scaler_scope,
        "feature_scaler": cfg.feature_scaler,
        "forecast_horizon": cfg.forecast_horizon,
        "seq_len": cfg.seq_len,
        "subsets": subset_report,
        "pairwise": pair_report,
        "scaler": scaler_report,
        "leakage_safe_for_publication": leakage_safe,
        "notes": (
            "No input-window overlap and train-only scaler."
            if leakage_safe
            else "This split/scaler combination is not sufficient for a publication-grade main result."
        ),
    }


def build_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name in {"huber", "smooth_l1"}:
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError(f"Unsupported loss: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: PipelineConfig):
    scheduler_type = cfg.scheduler_type.lower()
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.scheduler_step,
            gamma=cfg.scheduler_gamma,
        )
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.lr * 0.05)
    if scheduler_type == "none":
        return None
    raise ValueError(f"Unsupported scheduler type: {cfg.scheduler_type}")


def save_prediction_csv(path: Path, test_indices: np.ndarray, seq_len: int, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    rows = []
    for i, (start, actual, pred) in enumerate(zip(test_indices, y_true, y_pred)):
        rows.append(
            {
                "sample_index": int(i),
                "sequence_start": int(start),
                "label_feature_index": int(start + seq_len),
                "y_true": float(actual),
                "y_pred": float(pred),
                "error": float(pred - actual),
            }
        )
    write_dict_csv(
        path,
        rows,
        fieldnames=["sample_index", "sequence_start", "label_feature_index", "y_true", "y_pred", "error"],
    )


def save_prediction_plot(path: Path, subject: str, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, max_points: int) -> None:
    ensure_dir(path.parent)
    n_points = min(len(y_true), max_points)
    plt.figure(figsize=(12, 4))
    plt.plot(y_true[:n_points], label="Actual", linewidth=1.4)
    plt.plot(y_pred[:n_points], label="Predicted", linewidth=1.1)
    plt.title(f"{subject} - {model_name} knee angle prediction")
    plt.xlabel("Test sample")
    plt.ylabel("Knee angle (deg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def train_subject_model(
    prepared: PreparedSubject,
    model_name: str,
    cfg: PipelineConfig,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    record = prepared.record
    model_seed = stable_seed(cfg.seed, record.subject, model_name)
    model = build_pipeline_model(model_name, cfg).to(device)
    loss_fn = build_loss(cfg.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)
    train_loader = make_loader(prepared.train_dataset, cfg, shuffle=True, seed=model_seed, device=device)
    val_loader = (
        make_loader(prepared.val_dataset, cfg, shuffle=False, seed=model_seed, device=device)
        if prepared.val_dataset is not None
        else None
    )
    test_loader = make_loader(prepared.test_dataset, cfg, shuffle=False, seed=model_seed, device=device)

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    start_time = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, cfg.grad_clip)
        lr = float(optimizer.param_groups[0]["lr"])
        val_loss = evaluate_loss(model, val_loader, loss_fn, device) if val_loader is not None else float("nan")
        if cfg.best_checkpoint and val_loader is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
        if epoch == 1 or epoch == cfg.epochs or epoch % max(1, cfg.scheduler_step) == 0:
            print(
                f"  {record.subject} {model_name} epoch {epoch:03d}/{cfg.epochs}: "
                f"train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, lr={lr:.6g}",
                flush=True,
            )
        if (
            cfg.early_stopping_patience > 0
            and cfg.best_checkpoint
            and val_loader is not None
            and epochs_without_improvement >= cfg.early_stopping_patience
        ):
            print(
                f"  {record.subject} {model_name} early stopping at epoch {epoch} "
                f"(best_val_loss={best_val_loss:.5f})",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    predictions = predict(model, test_loader, device)
    val_predictions = predict(model, val_loader, device) if val_loader is not None else None
    metrics = compute_metrics(predictions["true"], predictions["pred"])
    elapsed = time.time() - start_time

    subject_dir = ensure_dir(output_dir / "subjects" / record.subject / model_name)
    save_prediction_csv(
        subject_dir / "predictions.csv",
        prepared.test_indices,
        cfg.seq_len,
        predictions["true"],
        predictions["pred"],
    )
    write_dict_csv(subject_dir / "history.csv", history, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
    if val_predictions is not None:
        save_prediction_csv(
            subject_dir / "val_predictions.csv",
            prepared.val_indices,
            cfg.seq_len,
            val_predictions["true"],
            val_predictions["pred"],
        )
    if cfg.plots:
        save_prediction_plot(
            subject_dir / "prediction_plot.png",
            record.subject,
            model_name,
            predictions["true"],
            predictions["pred"],
            cfg.plot_max_points,
        )

    checkpoint = {
        "subject": record.subject,
        "status": record.status,
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "config": asdict(cfg),
        "scaler": prepared.scaler.to_dict(),
        "feature_columns": prepared.feature_columns,
        "leakage_audit": leakage_audit_for_subject(prepared, cfg),
    }
    torch.save(checkpoint, subject_dir / "checkpoint.pt")
    save_json(
        subject_dir / "metadata.json",
        {
            "subject": record.subject,
            "status": record.status,
            "source_path": record.path,
            "total_rows": prepared.total_rows,
            "valid_rows": prepared.source_rows,
            "skipped_rows": prepared.skipped_rows,
            "feature_rows": prepared.n_feature_rows,
            "feature_columns": prepared.feature_columns,
            "n_train": len(prepared.train_dataset),
            "n_val": len(prepared.val_dataset) if prepared.val_dataset is not None else 0,
            "n_test": len(prepared.test_dataset),
            "split_mode": cfg.split_mode,
            "scaler_scope": cfg.scaler_scope,
            "leakage_audit": leakage_audit_for_subject(prepared, cfg),
            "best_val_loss": best_val_loss if best_state is not None else None,
            "metrics": metrics,
            "elapsed_seconds": elapsed,
        },
    )

    return {
        "subject": record.subject,
        "status": record.status,
        "model": model_name,
        "me": metrics["me"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "n_train": len(prepared.train_dataset),
        "n_val": len(prepared.val_dataset) if prepared.val_dataset is not None else 0,
        "n_test": len(prepared.test_dataset),
        "total_rows": prepared.total_rows,
        "valid_rows": prepared.source_rows,
        "skipped_rows": prepared.skipped_rows,
        "feature_rows": prepared.n_feature_rows,
        "elapsed_seconds": elapsed,
        "evaluation_mode": cfg.evaluation_mode,
        "split_mode": cfg.split_mode,
        "scaler_scope": cfg.scaler_scope,
        "feature_scaler": cfg.feature_scaler,
        "seed_or_ensemble": cfg.seed_label,
    }


def summarize_metrics(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        for group in ["overall", "healthy", "unhealthy"]:
            if group == "overall":
                group_rows = model_rows
            else:
                group_rows = [row for row in model_rows if row["status"] == group]
            if not group_rows:
                continue
            summaries.append(
                {
                    "model": model,
                    "group": group,
                    "n_subjects": len(group_rows),
                    "me": float(np.mean([row["me"] for row in group_rows])),
                    "mae": float(np.mean([row["mae"] for row in group_rows])),
                    "rmse": float(np.mean([row["rmse"] for row in group_rows])),
                }
            )
    return summaries


def save_metrics_tables(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    metric_fields = [
        "subject",
        "status",
        "model",
        "me",
        "mae",
        "rmse",
        "n_train",
        "n_val",
        "n_test",
        "total_rows",
        "valid_rows",
        "skipped_rows",
        "feature_rows",
        "elapsed_seconds",
        "evaluation_mode",
        "split_mode",
        "scaler_scope",
        "feature_scaler",
        "seed_or_ensemble",
    ]
    write_dict_csv(output_dir / "metrics_by_subject.csv", rows, fieldnames=metric_fields)
    write_dict_csv(
        output_dir / "metrics_summary.csv",
        summarize_metrics(rows),
        fieldnames=["model", "group", "n_subjects", "me", "mae", "rmse"],
    )


def save_leakage_audit(output_dir: Path, audits: Sequence[Dict[str, Any]]) -> None:
    safe_count = sum(1 for audit in audits if audit.get("leakage_safe_for_publication"))
    save_json(
        output_dir / "leakage_audit.json",
        {
            "summary": {
                "subjects": len(audits),
                "leakage_safe_subjects": int(safe_count),
                "all_subjects_publication_safe": bool(len(audits) > 0 and safe_count == len(audits)),
            },
            "subjects": list(audits),
        },
    )


def save_paper_match_report(output_dir: Path, rows: Sequence[Dict[str, Any]], cfg: PipelineConfig, n_subjects: int) -> None:
    statuses = sorted({row["status"] for row in rows})
    status_counts = {status: len({row["subject"] for row in rows if row["status"] == status}) for status in statuses}
    subjects = sorted({row["subject"] for row in rows})
    transformer_rows = [row for row in rows if row["model"] == "transformer"]
    quality_rows = transformer_rows
    if not quality_rows:
        seen_subjects = set()
        quality_rows = []
        for row in rows:
            if row["subject"] not in seen_subjects:
                seen_subjects.add(row["subject"])
                quality_rows.append(row)
    report = {
        "paper_match_summary": {
            "dataset_activity": "walking only (*mar.csv)",
            "input_channels": normalize_channels(cfg.channels),
            "target_channel": "FX knee angle",
            "feature_window_ms": cfg.feature_window,
            "feature_step_ms": cfg.feature_step,
            "feature_set": cfg.feature_set,
            "features_per_channel": feature_names_for_set(cfg.feature_set),
            "feature_dim": len(normalize_channels(cfg.channels)) * len(feature_names_for_set(cfg.feature_set)),
            "sequence_length": cfg.seq_len,
            "batch_size": cfg.batch_size,
            "subject_specific_training": True,
            "train_test_split": cfg.train_ratio,
            "validation_split": cfg.val_ratio,
            "test_split": cfg.test_ratio,
            "split_mode": cfg.split_mode,
            "scaler_scope": cfg.scaler_scope,
            "metrics": ["ME", "MAE", "RMSE"],
            "baseline": "MLP",
        },
        "transformer_defaults": {
            "d_model": cfg.d_model,
            "patch_size": cfg.patch_size,
            "encoder_layers": cfg.encoder_layers,
            "heads": cfg.heads,
            "ff_dim": cfg.ff_dim,
            "dropout": cfg.dropout,
            "scales": list(cfg.scales),
        },
        "training_defaults": {
            "epochs": cfg.epochs,
            "learning_rate": cfg.lr,
            "loss": cfg.loss,
            "weight_decay": cfg.weight_decay,
            "grad_clip": cfg.grad_clip,
            "early_stopping_patience": cfg.early_stopping_patience,
            "scheduler": cfg.scheduler_type,
            "normalization": f"{cfg.feature_scaler} feature scaler scope={cfg.scaler_scope}.",
        },
        "data_quality": {
            "requested_subjects": n_subjects,
            "completed_subjects": len(subjects),
            "status_counts": status_counts,
            "total_raw_rows": int(sum(row["total_rows"] for row in quality_rows)),
            "valid_numeric_rows": int(sum(row["valid_rows"] for row in quality_rows)),
            "skipped_malformed_rows": int(sum(row["skipped_rows"] for row in quality_rows)),
        },
        "reproduction_assumptions": [
            f"The paper does not specify the number of multi-scale branches M; this run uses scales {list(cfg.scales)}.",
            "The paper does not specify the initial learning rate, epoch count, number of attention heads, feed-forward width, dropout, or MLP hidden layers; these are configurable defaults.",
            (
                "This publication-oriented run fits Min-Max only on training input windows and uses purged temporal splits to avoid overlapping train/validation/test input windows."
                if cfg.split_mode == "purged_temporal" and cfg.scaler_scope == "train"
                else "The paper says Min-Max normalization is applied but does not state whether it is fit on train only or all data; this strict run fits it on training input windows only."
                if cfg.scaler_scope == "train"
                else "This paper_chasing run uses full-series Min-Max scaling and should not be interpreted as strict time-forward deployment performance."
            ),
            "Malformed CSV rows with missing EMG channels are skipped; valid rows preserve original time order.",
            "The paper's 'mean absolute error' time-domain feature is interpreted as the standard EMG mean absolute value feature.",
        ],
    }
    save_json(output_dir / "paper_match_report.json", report)
