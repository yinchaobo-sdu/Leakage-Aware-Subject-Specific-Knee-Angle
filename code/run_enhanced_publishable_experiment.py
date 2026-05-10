from __future__ import annotations

import argparse
import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from causalgait.data import SubjectRecord, describe_records, discover_subjects, find_dataset_root, load_subject_features, select_subjects
from causalgait.features import TargetScaler, feature_columns_for_channels, normalize_channels, split_and_scale
from causalgait.models import CausalGaitNet, MLPRegressor, MultiScaleTransformerRegressor
from causalgait.train import PipelineConfig, compute_metrics, leakage_audit_for_subject, save_prediction_csv, save_prediction_plot
from causalgait.utils import choose_device, ensure_dir, parse_int_list, save_json, set_seed, stable_seed, write_dict_csv


PAPER_MAE = 3.707
PAPER_RMSE = 4.691


@dataclass(frozen=True)
class EnhancedSpec:
    name: str
    model_name: str
    channels: Sequence[str]
    feature_set: str
    feature_scaler: str
    target_scaler: str
    horizons: Sequence[int]
    d_model: int
    heads: int
    ff_dim: int
    encoder_layers: int
    dropout: float
    role: str = "candidate"


@dataclass
class EnhancedPrepared:
    record: SubjectRecord
    features: np.ndarray
    angles: np.ndarray
    scaled_features: np.ndarray
    target_scaler: TargetScaler
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    feature_columns: List[str]
    n_feature_rows: int
    source_rows: int
    total_rows: int
    skipped_rows: int
    cfg: PipelineConfig


class MultiHorizonDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        angles: np.ndarray,
        indices: np.ndarray,
        seq_len: int,
        horizons: Sequence[int],
        target_scaler: TargetScaler,
    ) -> None:
        self.features = torch.as_tensor(np.ascontiguousarray(features), dtype=torch.float32)
        self.indices = torch.as_tensor(indices.astype(np.int64), dtype=torch.long)
        self.seq_len = int(seq_len)
        self.horizons = np.asarray(horizons, dtype=np.int64)
        targets = target_matrix(angles, indices, seq_len, horizons)
        self.targets = torch.as_tensor(target_scaler.transform(targets), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, item: int):
        start = int(self.indices[item].item())
        end = start + self.seq_len
        return self.features[start:end], self.targets[item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe enhanced four-channel sEMG experiments.")
    parser.add_argument("--data-root", default="SEMG_DB1")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "artifacts_enhanced_publishable"))
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--channels", default="RF,BF,VM,ST")
    parser.add_argument("--feature-set", choices=["original", "extended"], default="extended")
    parser.add_argument("--feature-scaler", choices=["minmax", "standard"], default="standard")
    parser.add_argument("--target-scaler", choices=["none", "standard"], default="standard")
    parser.add_argument("--horizons", default="1,5,10,20")
    parser.add_argument("--target-mae", type=float, default=PAPER_MAE)
    parser.add_argument("--target-rmse", type=float, default=PAPER_RMSE)
    parser.add_argument("--seeds", default="42,3407,2026,2027,7")
    parser.add_argument("--selection-seeds", default="42")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--feature-window", type=int, default=250)
    parser.add_argument("--feature-step", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ff-dim", type=int, default=384)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--scales", default="1,2,4,8")
    parser.add_argument("--input-noise-std", type=float, default=0.01)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-selection", action="store_true", help="Train only the enhanced multi-horizon CausalGaitNet.")
    return parser.parse_args()


def target_matrix(angles: np.ndarray, indices: np.ndarray, seq_len: int, horizons: Sequence[int]) -> np.ndarray:
    horizons_arr = np.asarray(horizons, dtype=np.int64)
    positions = indices.astype(np.int64)[:, None] + int(seq_len) + horizons_arr[None, :] - 1
    if positions.max(initial=0) >= len(angles):
        raise ValueError("A horizon label index exceeds the available angle sequence.")
    return angles[positions].astype(np.float32, copy=False)


def parse_seed_list(raw: str) -> List[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return seeds or [42]


def build_specs(args: argparse.Namespace) -> List[EnhancedSpec]:
    enhanced_channels = normalize_channels(args.channels)
    horizons = parse_int_list(args.horizons)
    scales = parse_int_list(args.scales)
    _ = scales
    specs = [
        EnhancedSpec(
            name="mlp_rf_vm_original",
            model_name="mlp",
            channels=("RF", "VM"),
            feature_set="original",
            feature_scaler="minmax",
            target_scaler="none",
            horizons=(1,),
            d_model=128,
            heads=4,
            ff_dim=256,
            encoder_layers=1,
            dropout=0.1,
            role="baseline",
        ),
        EnhancedSpec(
            name="transformer_rf_vm_original",
            model_name="transformer",
            channels=("RF", "VM"),
            feature_set="original",
            feature_scaler="minmax",
            target_scaler="none",
            horizons=(1,),
            d_model=128,
            heads=4,
            ff_dim=256,
            encoder_layers=1,
            dropout=0.1,
            role="candidate",
        ),
        EnhancedSpec(
            name="transformer_4ch_extended",
            model_name="transformer",
            channels=tuple(enhanced_channels),
            feature_set=args.feature_set,
            feature_scaler=args.feature_scaler,
            target_scaler=args.target_scaler,
            horizons=(1,),
            d_model=args.d_model,
            heads=args.heads,
            ff_dim=args.ff_dim,
            encoder_layers=args.encoder_layers,
            dropout=args.dropout,
            role="candidate",
        ),
        EnhancedSpec(
            name="causalgaitnet_4ch_extended",
            model_name="causalgaitnet",
            channels=tuple(enhanced_channels),
            feature_set=args.feature_set,
            feature_scaler=args.feature_scaler,
            target_scaler=args.target_scaler,
            horizons=(1,),
            d_model=args.d_model,
            heads=args.heads,
            ff_dim=args.ff_dim,
            encoder_layers=args.encoder_layers,
            dropout=args.dropout,
            role="candidate",
        ),
        EnhancedSpec(
            name="causalgaitnet_multihorizon_4ch_extended",
            model_name="causalgaitnet",
            channels=tuple(enhanced_channels),
            feature_set=args.feature_set,
            feature_scaler=args.feature_scaler,
            target_scaler=args.target_scaler,
            horizons=tuple(horizons),
            d_model=args.d_model,
            heads=args.heads,
            ff_dim=args.ff_dim,
            encoder_layers=args.encoder_layers,
            dropout=args.dropout,
            role="candidate",
        ),
    ]
    if args.skip_selection:
        return [specs[-1]]
    return specs


def make_cfg(args: argparse.Namespace, spec: EnhancedSpec, seed: int) -> PipelineConfig:
    return PipelineConfig(
        preset="enhanced_publishable",
        evaluation_mode="enhanced_publishable",
        split_mode="purged_temporal",
        scaler_scope="train",
        feature_window=args.feature_window,
        feature_step=args.feature_step,
        channels=tuple(spec.channels),
        feature_set=spec.feature_set,
        feature_scaler=spec.feature_scaler,
        forecast_horizon=max(spec.horizons),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_type="cosine",
        loss="smooth_l1",
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        best_checkpoint=True,
        seed=seed,
        seed_label=str(seed),
        num_workers=args.num_workers,
        d_model=spec.d_model,
        patch_size=8,
        encoder_layers=spec.encoder_layers,
        heads=spec.heads,
        ff_dim=spec.ff_dim,
        dropout=spec.dropout,
        scales=parse_int_list(args.scales),
        plots=not args.no_plots,
        no_cache=args.no_cache,
    )


def prepare_enhanced_subject(record: SubjectRecord, args: argparse.Namespace, spec: EnhancedSpec, seed: int, cache_dir: Path) -> EnhancedPrepared:
    cfg = make_cfg(args, spec, seed)
    bundle = load_subject_features(
        record=record,
        cache_dir=cache_dir,
        feature_window=cfg.feature_window,
        feature_step=cfg.feature_step,
        no_cache=cfg.no_cache,
        channels=spec.channels,
        feature_set=spec.feature_set,
    )
    scaled_features, angles, feature_scaler, train_indices, val_indices, test_indices = split_and_scale(
        bundle.features,
        bundle.angles,
        seq_len=cfg.seq_len,
        split_mode=cfg.split_mode,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        scaler_scope=cfg.scaler_scope,
        seed=stable_seed(seed, record.subject, spec.name),
        feature_scaler=spec.feature_scaler,
        forecast_horizon=max(spec.horizons),
    )
    train_targets = target_matrix(angles, train_indices, cfg.seq_len, spec.horizons)
    target_scaler = TargetScaler.fit(train_targets.reshape(-1), spec.target_scaler)
    cfg.feature_dim = len(bundle.feature_columns)
    return EnhancedPrepared(
        record=record,
        features=bundle.features,
        angles=angles,
        scaled_features=scaled_features,
        target_scaler=target_scaler,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        feature_columns=list(bundle.feature_columns),
        n_feature_rows=len(scaled_features),
        source_rows=bundle.source_rows,
        total_rows=bundle.total_rows,
        skipped_rows=bundle.skipped_rows,
        cfg=cfg,
    )


def make_dataset(prepared: EnhancedPrepared, indices: np.ndarray, spec: EnhancedSpec) -> MultiHorizonDataset:
    return MultiHorizonDataset(
        prepared.scaled_features,
        prepared.angles,
        indices,
        prepared.cfg.seq_len,
        spec.horizons,
        prepared.target_scaler,
    )


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator if shuffle else None,
        drop_last=False,
    )


def enhanced_leakage_audit(prepared: EnhancedPrepared, spec: EnhancedSpec) -> Dict[str, Any]:
    audit = leakage_audit_for_subject(prepared, prepared.cfg)
    audit.update(
        {
            "model_spec": spec.name,
            "channels": list(spec.channels),
            "feature_set": spec.feature_set,
            "target_scaler": spec.target_scaler,
            "horizons": list(spec.horizons),
        }
    )
    return audit


def build_model(spec: EnhancedSpec, feature_dim: int, seq_len: int, scales: Sequence[int]) -> nn.Module:
    output_dim = len(spec.horizons)
    kwargs = {
        "seq_len": seq_len,
        "feature_dim": feature_dim,
        "scales": scales,
        "patch_size": 8,
        "d_model": spec.d_model,
        "nhead": spec.heads,
        "num_layers": spec.encoder_layers,
        "ff_dim": spec.ff_dim,
        "dropout": spec.dropout,
        "output_dim": output_dim,
    }
    if spec.model_name == "transformer":
        return MultiScaleTransformerRegressor(**kwargs)
    if spec.model_name == "causalgaitnet":
        return CausalGaitNet(**kwargs)
    if spec.model_name == "mlp":
        return MLPRegressor(seq_len=seq_len, feature_dim=feature_dim, output_dim=output_dim)
    raise ValueError(f"Unsupported enhanced model: {spec.model_name}")


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)


def ensure_2d(pred: torch.Tensor) -> torch.Tensor:
    return pred.unsqueeze(1) if pred.ndim == 1 else pred


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip: float, input_noise_std: float) -> float:
    model.train()
    losses: List[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if input_noise_std > 0:
            x = x + torch.randn_like(x) * input_noise_std
        optimizer.zero_grad(set_to_none=True)
        pred = ensure_2d(model(x))
        loss = loss_fn(pred, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, target_scaler: TargetScaler) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y_device = y.to(device, non_blocking=True)
        pred = ensure_2d(model(x))
        loss = loss_fn(pred, y_device)
        losses.append(float(loss.detach().cpu().item()))
        preds.append(pred.detach().cpu().numpy())
        trues.append(y.numpy())
    pred_scaled = np.concatenate(preds, axis=0) if preds else np.empty((0, 1), dtype=np.float32)
    true_scaled = np.concatenate(trues, axis=0) if trues else np.empty((0, 1), dtype=np.float32)
    primary_pred = target_scaler.inverse_transform(pred_scaled[:, 0])
    primary_true = target_scaler.inverse_transform(true_scaled[:, 0])
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "pred_scaled": pred_scaled,
        "true_scaled": true_scaled,
        "pred": primary_pred,
        "true": primary_true,
        "metrics": compute_metrics(primary_true, primary_pred),
    }


def train_seed(
    prepared: EnhancedPrepared,
    spec: EnhancedSpec,
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
    device: torch.device,
    save_artifacts: bool,
) -> Dict[str, Any]:
    seed_value = stable_seed(seed, prepared.record.subject, spec.name)
    set_seed(seed_value)
    train_ds = make_dataset(prepared, prepared.train_indices, spec)
    val_ds = make_dataset(prepared, prepared.val_indices, spec)
    test_ds = make_dataset(prepared, prepared.test_indices, spec)
    train_loader = make_loader(train_ds, args, shuffle=True, seed=seed_value, device=device)
    val_loader = make_loader(val_ds, args, shuffle=False, seed=seed_value, device=device)
    test_loader = make_loader(test_ds, args, shuffle=False, seed=seed_value, device=device)
    model = build_model(spec, len(prepared.feature_columns), args.seq_len, parse_int_list(args.scales)).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    history: List[Dict[str, Any]] = []
    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, args.grad_clip, args.input_noise_std)
        val_result = evaluate(model, val_loader, loss_fn, device, prepared.target_scaler)
        val_loss = val_result["loss"]
        lr = float(optimizer.param_groups[0]["lr"])
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
        if epoch == 1 or epoch == args.epochs or epoch % max(10, args.epochs // 4) == 0:
            print(
                f"  {prepared.record.subject} {spec.name} seed={seed} epoch {epoch:03d}/{args.epochs}: "
                f"train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, lr={lr:.6g}",
                flush=True,
            )
        if args.early_stopping_patience > 0 and stale_epochs >= args.early_stopping_patience:
            print(f"  {prepared.record.subject} {spec.name} seed={seed} early stopping at epoch {epoch}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_result = evaluate(model, val_loader, loss_fn, device, prepared.target_scaler)
    test_result = evaluate(model, test_loader, loss_fn, device, prepared.target_scaler)
    result = {
        "history": history,
        "best_val_loss": best_val,
        "state_dict": copy.deepcopy(model.state_dict()),
        "val": val_result,
        "test": test_result,
    }
    if save_artifacts:
        seed_dir = ensure_dir(output_dir / "subjects" / prepared.record.subject / spec.name / f"seed_{seed}")
        write_dict_csv(seed_dir / "history.csv", history, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        save_prediction_csv(seed_dir / "val_predictions.csv", prepared.val_indices, args.seq_len, val_result["true"], val_result["pred"])
        save_prediction_csv(seed_dir / "test_predictions.csv", prepared.test_indices, args.seq_len, test_result["true"], test_result["pred"])
        torch.save(
            {
                "subject": prepared.record.subject,
                "status": prepared.record.status,
                "model_name": spec.name,
                "model_state_dict": result["state_dict"],
                "best_val_loss": best_val,
                "spec": asdict(spec),
                "config": asdict(prepared.cfg),
                "feature_columns": prepared.feature_columns,
                "feature_scaler": prepared.cfg.feature_scaler,
                "target_scaler": prepared.target_scaler.to_dict(),
                "leakage_audit": enhanced_leakage_audit(prepared, spec),
            },
            seed_dir / "checkpoint.pt",
        )
    return result


def row_for_result(prepared: EnhancedPrepared, spec: EnhancedSpec, split_name: str, metrics: Dict[str, float], seed_label: str, elapsed: float, main_result: bool) -> Dict[str, Any]:
    return {
        "subject": prepared.record.subject,
        "status": prepared.record.status,
        "model": spec.name,
        "split": split_name,
        "role": spec.role,
        "main_result": bool(main_result),
        "me": metrics["me"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "n_train": int(len(prepared.train_indices)),
        "n_val": int(len(prepared.val_indices)),
        "n_test": int(len(prepared.test_indices)),
        "total_rows": int(prepared.total_rows),
        "valid_rows": int(prepared.source_rows),
        "skipped_rows": int(prepared.skipped_rows),
        "feature_rows": int(prepared.n_feature_rows),
        "feature_dim": int(len(prepared.feature_columns)),
        "elapsed_seconds": float(elapsed),
        "evaluation_mode": "enhanced_publishable",
        "split_mode": "purged_temporal",
        "scaler_scope": "train",
        "feature_scaler": spec.feature_scaler,
        "target_scaler": spec.target_scaler,
        "channels": ",".join(spec.channels),
        "feature_set": spec.feature_set,
        "horizons": ",".join(str(h) for h in spec.horizons),
        "seed_or_ensemble": seed_label,
    }


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    test_rows = [row for row in rows if row.get("split") == "test"]
    for model in sorted({row["model"] for row in test_rows}):
        model_rows = [row for row in test_rows if row["model"] == model]
        main_rows = [row for row in model_rows if row.get("main_result")]
        if main_rows:
            model_rows = main_rows
        for group in ["overall", "healthy", "unhealthy"]:
            group_rows = model_rows if group == "overall" else [row for row in model_rows if row["status"] == group]
            if not group_rows:
                continue
            summaries.append(
                {
                    "model": model,
                    "group": group,
                    "n_subjects": len(group_rows),
                    "main_result": bool(any(row.get("main_result") for row in group_rows)),
                    "me": float(np.mean([row["me"] for row in group_rows])),
                    "mae": float(np.mean([row["mae"] for row in group_rows])),
                    "rmse": float(np.mean([row["rmse"] for row in group_rows])),
                }
            )
    return summaries


def save_metric_tables(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "subject",
        "status",
        "model",
        "split",
        "role",
        "main_result",
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
        "feature_dim",
        "elapsed_seconds",
        "evaluation_mode",
        "split_mode",
        "scaler_scope",
        "feature_scaler",
        "target_scaler",
        "channels",
        "feature_set",
        "horizons",
        "seed_or_ensemble",
    ]
    write_dict_csv(output_dir / "metrics_by_subject.csv", rows, fieldnames=fields)
    write_dict_csv(
        output_dir / "metrics_summary.csv",
        summarize_rows(rows),
        fieldnames=["model", "group", "n_subjects", "main_result", "me", "mae", "rmse"],
    )


def save_leakage(output_dir: Path, audits: Sequence[Dict[str, Any]]) -> None:
    unique = {}
    for audit in audits:
        unique[
            (
                audit["subject"],
                audit.get("model_spec"),
                tuple(audit.get("channels", [])),
                audit.get("feature_set"),
                audit.get("target_scaler"),
                tuple(audit.get("horizons", [])),
                audit["split_mode"],
                audit["scaler_scope"],
                audit.get("feature_scaler"),
                audit.get("seq_len"),
                audit.get("forecast_horizon"),
            )
        ] = audit
    audits_out = list(unique.values())
    safe_count = sum(1 for audit in audits_out if audit.get("leakage_safe_for_publication"))
    save_json(
        output_dir / "leakage_audit.json",
        {
            "summary": {
                "subjects": len({audit["subject"] for audit in audits_out}),
                "audited_subject_specs": len(audits_out),
                "leakage_safe_subject_specs": int(safe_count),
                "all_subject_specs_publication_safe": bool(audits_out and safe_count == len(audits_out)),
            },
            "subjects": audits_out,
        },
    )


def choose_selected_spec(specs: Sequence[EnhancedSpec], validation_rows: Sequence[Dict[str, Any]]) -> EnhancedSpec:
    candidate_rows = [row for row in validation_rows if row["role"] == "candidate" and row["split"] == "val"]
    if not candidate_rows:
        return specs[-1]
    means = []
    for name in sorted({row["model"] for row in candidate_rows}):
        rows = [row for row in candidate_rows if row["model"] == name]
        means.append((float(np.mean([row["mae"] for row in rows])), name, len(rows)))
    means.sort()
    selected_name = means[0][1]
    return next(spec for spec in specs if spec.name == selected_name)


def save_selection_report(output_dir: Path, specs: Sequence[EnhancedSpec], rows: Sequence[Dict[str, Any]], selected: EnhancedSpec) -> None:
    val_rows = [row for row in rows if row["split"] == "val"]
    candidates = []
    for name in sorted({row["model"] for row in val_rows}):
        model_rows = [row for row in val_rows if row["model"] == name]
        spec = next(item for item in specs if item.name == name)
        candidates.append(
            {
                "model": name,
                "role": spec.role,
                "n_subjects": len(model_rows),
                "mean_val_mae": float(np.mean([row["mae"] for row in model_rows])),
                "mean_val_rmse": float(np.mean([row["rmse"] for row in model_rows])),
                "eligible_for_main_selection": spec.role == "candidate",
            }
        )
    save_json(
        output_dir / "validation_selection_report.json",
        {
            "selection_metric": "mean validation MAE across subjects",
            "selected_model": selected.name,
            "selected_spec": asdict(selected),
            "candidates": candidates,
            "note": "Only validation rows are used for selecting the main configuration; test rows for non-selected models are exploratory.",
        },
    )


def save_statistical_report(output_dir: Path, rows: Sequence[Dict[str, Any]], target_mae: float, target_rmse: float) -> None:
    test_rows = [row for row in rows if row["split"] == "test"]
    report: Dict[str, Any] = {"target": {"mae": target_mae, "rmse": target_rmse}, "models": []}
    for name in sorted({row["model"] for row in test_rows}):
        model_rows = [row for row in test_rows if row["model"] == name]
        main_rows = [row for row in model_rows if row.get("main_result")]
        if main_rows:
            model_rows = main_rows
        maes = np.asarray([row["mae"] for row in model_rows], dtype=np.float64)
        rmses = np.asarray([row["rmse"] for row in model_rows], dtype=np.float64)
        report["models"].append(
            {
                "model": name,
                "n_subjects": int(len(model_rows)),
                "main_result": bool(any(row.get("main_result") for row in model_rows)),
                "mae_mean": float(maes.mean()),
                "mae_std": float(maes.std(ddof=1)) if len(maes) > 1 else 0.0,
                "rmse_mean": float(rmses.mean()),
                "rmse_std": float(rmses.std(ddof=1)) if len(rmses) > 1 else 0.0,
                "subjects_beating_paper_mae": int(np.sum(maes < target_mae)),
                "subjects_beating_paper_rmse": int(np.sum(rmses < target_rmse)),
            }
        )
    save_json(output_dir / "statistical_report.json", report)


def save_readiness_report(output_dir: Path, rows: Sequence[Dict[str, Any]], selected: EnhancedSpec, target_mae: float, target_rmse: float) -> None:
    main_rows = [row for row in rows if row["split"] == "test" and row.get("main_result") and row["model"] == selected.name]
    if main_rows:
        overall = {
            "model": selected.name,
            "n_subjects": len(main_rows),
            "me": float(np.mean([row["me"] for row in main_rows])),
            "mae": float(np.mean([row["mae"] for row in main_rows])),
            "rmse": float(np.mean([row["rmse"] for row in main_rows])),
        }
    else:
        overall = None
    beats = bool(overall and overall["mae"] < target_mae and overall["rmse"] < target_rmse)
    save_json(
        output_dir / "publication_readiness_report.json",
        {
            "publication_safe_protocol": True,
            "evaluation_mode": "enhanced_publishable",
            "selected_model": selected.name,
            "target": {"mae": target_mae, "rmse": target_rmse},
            "main_result_overall": overall,
            "beats_paper_table_i": beats,
            "paper_chasing_results_usable_as_main_claim": False,
            "notes": [
                "Main protocol uses purged temporal splits and train-only feature/target scalers.",
                "No historical knee angle input, full-series scaling, random-overlap test split, nearest-label lookup, or overlap-aware label postprocessing is used.",
                "Non-selected test rows are exploratory and must not be used for model selection.",
            ],
        },
    )


def run_single_seed_spec(
    record: SubjectRecord,
    args: argparse.Namespace,
    spec: EnhancedSpec,
    seed: int,
    cache_dir: Path,
    output_dir: Path,
    device: torch.device,
    save_artifacts: bool,
) -> tuple[EnhancedPrepared, Dict[str, Any], float]:
    start = time.time()
    prepared = prepare_enhanced_subject(record, args, spec, seed, cache_dir)
    result = train_seed(prepared, spec, args, seed, output_dir, device, save_artifacts=save_artifacts)
    elapsed = time.time() - start
    return prepared, result, elapsed


def run_final_ensemble(
    record: SubjectRecord,
    args: argparse.Namespace,
    spec: EnhancedSpec,
    seeds: Sequence[int],
    cache_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[EnhancedPrepared, Dict[str, Any], float]:
    start = time.time()
    prepared = prepare_enhanced_subject(record, args, spec, seeds[0], cache_dir)
    test_preds = []
    test_true = None
    val_metrics = []
    for seed in seeds:
        result = train_seed(prepared, spec, args, seed, output_dir, device, save_artifacts=True)
        test_preds.append(result["test"]["pred"])
        test_true = result["test"]["true"]
        val_metrics.append(result["val"]["metrics"])
    assert test_true is not None
    ensemble_pred = np.mean(np.stack(test_preds, axis=0), axis=0)
    metrics = compute_metrics(test_true, ensemble_pred)
    subject_dir = ensure_dir(output_dir / "subjects" / record.subject / spec.name)
    save_prediction_csv(subject_dir / "ensemble_predictions.csv", prepared.test_indices, args.seq_len, test_true, ensemble_pred)
    if not args.no_plots:
        save_prediction_plot(
            subject_dir / "ensemble_prediction_plot.png",
            record.subject,
            spec.name,
            test_true,
            ensemble_pred,
            3000,
        )
    save_json(
        subject_dir / "metadata.json",
        {
            "subject": record.subject,
            "status": record.status,
            "model": spec.name,
            "main_result": True,
            "seeds": list(seeds),
            "metrics": metrics,
            "mean_seed_val_metrics": {
                "mae": float(np.mean([item["mae"] for item in val_metrics])),
                "rmse": float(np.mean([item["rmse"] for item in val_metrics])),
                "me": float(np.mean([item["me"] for item in val_metrics])),
            },
            "spec": asdict(spec),
            "config": asdict(prepared.cfg),
            "feature_columns": prepared.feature_columns,
            "feature_scaler": prepared.cfg.feature_scaler,
            "target_scaler": prepared.target_scaler.to_dict(),
            "leakage_audit": enhanced_leakage_audit(prepared, spec),
            "elapsed_seconds": time.time() - start,
        },
    )
    return prepared, {"metrics": metrics, "true": test_true, "pred": ensemble_pred}, time.time() - start


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    cache_dir = ensure_dir(output_dir / "feature_cache")
    data_root = find_dataset_root(Path(args.data_root))
    records = select_subjects(discover_subjects(data_root), args.subjects)
    if args.max_subjects > 0:
        records = records[: args.max_subjects]
    device = choose_device(args.device)
    seeds = parse_seed_list(args.seeds)
    selection_seeds = parse_seed_list(args.selection_seeds)
    specs = build_specs(args)

    save_json(
        output_dir / "run_config.json",
        {
            "data_root": data_root,
            "output_dir": output_dir,
            "subjects": [record.subject for record in records],
            "device": str(device),
            "seeds": seeds,
            "selection_seeds": selection_seeds,
            "specs": [asdict(spec) for spec in specs],
            "args": vars(args),
        },
    )
    print(f"Dataset root: {data_root}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Subjects ({len(records)}): {describe_records(records)}", flush=True)

    metric_rows: List[Dict[str, Any]] = []
    leakage_audits: List[Dict[str, Any]] = []
    selection_seed = selection_seeds[0]

    for spec in specs:
        print(f"\n[selection] {spec.name} role={spec.role}", flush=True)
        for subject_i, record in enumerate(records, start=1):
            print(f"  [{subject_i}/{len(records)}] {record.subject} ({record.status})", flush=True)
            prepared, result, elapsed = run_single_seed_spec(
                record,
                args,
                spec,
                selection_seed,
                cache_dir,
                output_dir,
                device,
                save_artifacts=False,
            )
            leakage_audits.append(enhanced_leakage_audit(prepared, spec))
            val_row = row_for_result(prepared, spec, "val", result["val"]["metrics"], str(selection_seed), elapsed, False)
            test_row = row_for_result(prepared, spec, "test", result["test"]["metrics"], str(selection_seed), elapsed, False)
            metric_rows.extend([val_row, test_row])
            print(
                f"    val MAE={val_row['mae']:.4f}, RMSE={val_row['rmse']:.4f}; "
                f"test MAE={test_row['mae']:.4f}, RMSE={test_row['rmse']:.4f}",
                flush=True,
            )
            save_metric_tables(output_dir, metric_rows)
            save_leakage(output_dir, leakage_audits)

    selected = choose_selected_spec(specs, metric_rows)
    save_selection_report(output_dir, specs, metric_rows, selected)
    print(f"\nSelected main configuration by validation MAE: {selected.name}", flush=True)

    for subject_i, record in enumerate(records, start=1):
        print(f"\n[final ensemble {subject_i}/{len(records)}] {record.subject} ({record.status}) {selected.name}", flush=True)
        prepared, result, elapsed = run_final_ensemble(record, args, selected, seeds, cache_dir, output_dir, device)
        leakage_audits.append(enhanced_leakage_audit(prepared, selected))
        row = row_for_result(prepared, selected, "test", result["metrics"], ",".join(str(seed) for seed in seeds), elapsed, True)
        metric_rows.append(row)
        print(f"  ensemble test MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}", flush=True)
        save_metric_tables(output_dir, metric_rows)
        save_leakage(output_dir, leakage_audits)
        save_statistical_report(output_dir, metric_rows, args.target_mae, args.target_rmse)
        save_readiness_report(output_dir, metric_rows, selected, args.target_mae, args.target_rmse)

    save_metric_tables(output_dir, metric_rows)
    save_leakage(output_dir, leakage_audits)
    save_selection_report(output_dir, specs, metric_rows, selected)
    save_statistical_report(output_dir, metric_rows, args.target_mae, args.target_rmse)
    save_readiness_report(output_dir, metric_rows, selected, args.target_mae, args.target_rmse)
    print("\nDone. Enhanced publication-safe artifacts written to:")
    print(f"  {output_dir / 'metrics_summary.csv'}")
    print(f"  {output_dir / 'validation_selection_report.json'}")
    print(f"  {output_dir / 'leakage_audit.json'}")
    print(f"  {output_dir / 'publication_readiness_report.json'}")


if __name__ == "__main__":
    main()
