from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from causalgait.data import describe_records, discover_subjects, find_dataset_root, select_subjects
from causalgait.features import StandardScaler, _feature_rows_for_sequences
from causalgait.utils import choose_device, ensure_dir, parse_int_list, save_json, write_dict_csv
from run_enhanced_publishable_experiment import (
    EnhancedPrepared,
    EnhancedSpec,
    MultiHorizonDataset,
    build_model,
    compute_metrics,
    enhanced_leakage_audit,
    make_cfg,
    parse_seed_list,
    prepare_enhanced_subject,
    row_for_result,
    run_final_ensemble,
    save_leakage,
    save_metric_tables,
    save_prediction_plot,
    target_matrix,
)


MAIN_MODEL = "transformer_4ch_extended"
NO_MULTISCALE_MODEL = "transformer_4ch_extended_nomultiscale"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reviewer-requested multiscale ablation and robustness experiments.")
    parser.add_argument("--data-root", default="SEMG_DB1")
    parser.add_argument("--main-artifact-root", default=str(Path(__file__).resolve().parent / "artifacts_enhanced_publishable_remote"))
    parser.add_argument("--ablation-output-dir", default=str(Path(__file__).resolve().parent / "artifacts_nomultiscale"))
    parser.add_argument("--revision-dir", default=str(Path(__file__).resolve().parent / "revision_package"))
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--run", choices=["ablation", "robustness", "both"], default="both")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--feature-window", type=int, default=250)
    parser.add_argument("--feature-step", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument("--input-noise-std", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--robustness-batch-size", type=int, default=256)
    return parser.parse_args()


def transformer_spec(name: str = MAIN_MODEL) -> EnhancedSpec:
    return EnhancedSpec(
        name=name,
        model_name="transformer",
        channels=("RF", "BF", "VM", "ST"),
        feature_set="extended",
        feature_scaler="standard",
        target_scaler="standard",
        horizons=(1,),
        d_model=192,
        heads=6,
        ff_dim=384,
        encoder_layers=2,
        dropout=0.05,
        role="candidate",
    )


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def summarize_subject_rows(rows: Sequence[Dict[str, Any]], model: str, metric_prefix: str = "") -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for group in ["overall", "healthy", "unhealthy"]:
        group_rows = [row for row in rows if row["model"] == model and (group == "overall" or row["status"] == group)]
        if not group_rows:
            continue
        summaries.append(
            {
                "model": model,
                "group": group,
                "n_subjects": len(group_rows),
                f"{metric_prefix}ME": mean(float(row["ME"]) for row in group_rows),
                f"{metric_prefix}MAE": mean(float(row["MAE"]) for row in group_rows),
                f"{metric_prefix}RMSE": mean(float(row["RMSE"]) for row in group_rows),
            }
        )
    return summaries


def build_ablation_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=args.data_root,
        output_dir=str(output_dir),
        subjects=args.subjects,
        channels="RF,BF,VM,ST",
        feature_set="extended",
        feature_scaler="standard",
        target_scaler="standard",
        horizons="1",
        target_mae=3.707,
        target_rmse=4.691,
        seeds=args.seeds,
        selection_seeds=args.seeds.split(",")[0],
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        feature_window=args.feature_window,
        feature_step=args.feature_step,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        early_stopping_patience=args.early_stopping_patience,
        d_model=192,
        heads=6,
        ff_dim=384,
        encoder_layers=2,
        dropout=0.05,
        scales="1",
        input_noise_std=args.input_noise_std,
        device=args.device,
        num_workers=args.num_workers,
        max_subjects=args.max_subjects,
        no_cache=args.no_cache,
        no_plots=args.no_plots,
        skip_selection=False,
    )


def run_nomultiscale_ablation(args: argparse.Namespace, records: Sequence[Any], device: torch.device) -> Path:
    output_dir = ensure_dir(Path(args.ablation_output_dir).expanduser().resolve())
    cache_dir = ensure_dir(Path(args.main_artifact_root).expanduser().resolve() / "feature_cache")
    ablation_args = build_ablation_args(args, output_dir)
    spec = transformer_spec(NO_MULTISCALE_MODEL)
    seeds = parse_seed_list(args.seeds)
    metric_rows: List[Dict[str, Any]] = []
    leakage_audits: List[Dict[str, Any]] = []

    save_json(
        output_dir / "run_config.json",
        {
            "experiment": "reviewer_no_multiscale_ablation",
            "model": asdict(spec),
            "scales": [1],
            "seeds": seeds,
            "epochs": args.epochs,
            "data_root": str(Path(args.data_root).expanduser().resolve()),
        },
    )
    print(f"[ablation] Output: {output_dir}", flush=True)
    print(f"[ablation] Subjects ({len(records)}): {describe_records(records)}", flush=True)

    for subject_i, record in enumerate(records, start=1):
        print(f"\n[ablation {subject_i}/{len(records)}] {record.subject} ({record.status}) scales=1", flush=True)
        prepared, result, elapsed = run_final_ensemble(record, ablation_args, spec, seeds, cache_dir, output_dir, device)
        leakage_audits.append(enhanced_leakage_audit(prepared, spec))
        row = row_for_result(prepared, spec, "test", result["metrics"], ",".join(str(seed) for seed in seeds), elapsed, True)
        metric_rows.append(row)
        print(f"  no-multiscale test MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}", flush=True)
        save_metric_tables(output_dir, metric_rows)
        save_leakage(output_dir, leakage_audits)

    save_metric_tables(output_dir, metric_rows)
    save_leakage(output_dir, leakage_audits)
    return output_dir


def model_scale_string(row: Dict[str, str]) -> str:
    return row.get("seed_or_ensemble", "")


def write_multiscale_ablation_tables(main_artifact_root: Path, ablation_dir: Path, revision_dir: Path) -> None:
    table_dir = ensure_dir(revision_dir / "revision_tables")
    main_rows = read_csv(main_artifact_root / "metrics_by_subject.csv")
    ablation_rows = read_csv(ablation_dir / "metrics_by_subject.csv")
    no_rows = [row for row in ablation_rows if row["model"] == NO_MULTISCALE_MODEL and row["split"] == "test"]
    if not no_rows:
        raise RuntimeError("No no-multiscale rows found.")
    seed_label = no_rows[0].get("seed_or_ensemble", "")
    if "," in seed_label:
        multi_rows = [
            row
            for row in main_rows
            if row["model"] == MAIN_MODEL and row["split"] == "test" and row.get("main_result", "").lower() == "true"
        ]
        multi_label = "ensemble"
    else:
        multi_rows = [
            row
            for row in main_rows
            if row["model"] == MAIN_MODEL
            and row["split"] == "test"
            and row.get("main_result", "").lower() == "false"
            and row.get("seed_or_ensemble", "") == seed_label
        ]
        multi_label = f"seed_{seed_label}"
    no_by_subject = {row["subject"]: row for row in no_rows}
    multi_by_subject = {row["subject"]: row for row in multi_rows}
    by_subject: List[Dict[str, Any]] = []
    for subject, no_row in sorted(no_by_subject.items()):
        multi_row = multi_by_subject.get(subject)
        if not multi_row:
            continue
        by_subject.append(
            {
                "subject": subject,
                "status": no_row["status"],
                "comparison_basis": multi_label,
                "nomultiscale_MAE": float(no_row["mae"]),
                "nomultiscale_RMSE": float(no_row["rmse"]),
                "multiscale_MAE": float(multi_row["mae"]),
                "multiscale_RMSE": float(multi_row["rmse"]),
                "delta_MAE_nomulti_minus_multi": float(no_row["mae"]) - float(multi_row["mae"]),
                "delta_RMSE_nomulti_minus_multi": float(no_row["rmse"]) - float(multi_row["rmse"]),
            }
        )
    write_dict_csv(table_dir / "ablation_multiscale_by_subject.csv", by_subject)
    no_mae = mean(row["nomultiscale_MAE"] for row in by_subject)
    no_rmse = mean(row["nomultiscale_RMSE"] for row in by_subject)
    multi_mae = mean(row["multiscale_MAE"] for row in by_subject)
    multi_rmse = mean(row["multiscale_RMSE"] for row in by_subject)
    write_dict_csv(
        table_dir / "ablation_multiscale.csv",
        [
            {
                "ablation": "No-multiscale branch",
                "status": "completed",
                "implementation_note": "Transformer trained with scales=1 under purged temporal protocol.",
                "model": NO_MULTISCALE_MODEL,
                "scales": "1",
                "seed_or_ensemble": seed_label,
                "MAE": no_mae,
                "RMSE": no_rmse,
            },
            {
                "ablation": "Multi-scale branch",
                "status": "completed",
                "implementation_note": "Same Transformer family trained with scales=1,2,4,8.",
                "model": MAIN_MODEL,
                "scales": "1,2,4,8",
                "seed_or_ensemble": seed_label if "," not in seed_label else "42,3407,2026,2027,7",
                "MAE": multi_mae,
                "RMSE": multi_rmse,
            },
            {
                "ablation": "Delta no-multiscale minus multi-scale",
                "status": "completed",
                "implementation_note": "Positive delta means multi-scale is better.",
                "model": "delta",
                "scales": "",
                "seed_or_ensemble": seed_label,
                "MAE": no_mae - multi_mae,
                "RMSE": no_rmse - multi_rmse,
            },
        ],
    )


def channel_columns(feature_columns: Sequence[str], channel: str) -> List[int]:
    prefix = f"{channel}_"
    return [i for i, name in enumerate(feature_columns) if name.startswith(prefix)]


def amplitude_scaled_raw_features(features: np.ndarray, feature_columns: Sequence[str], factor: float) -> np.ndarray:
    scaled = features.copy()
    linear_suffixes = (
        "_max",
        "_mean_absolute_value",
        "_mean",
        "_min",
        "_rms",
        "_std",
        "_waveform_length",
        "_integrated_emg",
    )
    quadratic_suffixes = ("_variance",)
    for i, name in enumerate(feature_columns):
        if name.endswith(linear_suffixes):
            scaled[:, i] *= factor
        elif name.endswith(quadratic_suffixes):
            scaled[:, i] *= factor * factor
    return scaled


def smooth_features(values: np.ndarray, width: int = 5) -> np.ndarray:
    if width <= 1:
        return values.copy()
    pad = width // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(values)
    for i in range(values.shape[0]):
        out[i] = padded[i : i + width].mean(axis=0)
    return out


def train_standard_scaler(prepared: EnhancedPrepared) -> StandardScaler:
    fit_rows = _feature_rows_for_sequences(prepared.features, prepared.train_indices, prepared.cfg.seq_len)
    return StandardScaler.fit(fit_rows)


def make_perturbed_features(
    prepared: EnhancedPrepared,
    condition: str,
    rng: np.random.Generator,
) -> np.ndarray:
    base = prepared.scaled_features.copy()
    if condition == "clean":
        return base
    if condition.startswith("gaussian_noise_std_"):
        std = float(condition.rsplit("_", 1)[1])
        return (base + rng.normal(0.0, std, size=base.shape)).astype(np.float32)
    if condition.startswith("channel_dropout_"):
        channel = condition.rsplit("_", 1)[1]
        cols = channel_columns(prepared.feature_columns, channel)
        perturbed = base.copy()
        perturbed[:, cols] = 0.0
        return perturbed.astype(np.float32)
    if condition.startswith("amplitude_scale_"):
        factor = float(condition.rsplit("_", 1)[1])
        scaler = train_standard_scaler(prepared)
        raw = amplitude_scaled_raw_features(prepared.features, prepared.feature_columns, factor)
        return scaler.transform(raw)
    if condition == "feature_smoothing_5":
        return smooth_features(base, width=5).astype(np.float32)
    raise ValueError(f"Unknown robustness condition: {condition}")


@torch.no_grad()
def ensemble_predict(
    prepared: EnhancedPrepared,
    spec: EnhancedSpec,
    args: argparse.Namespace,
    scaled_features: np.ndarray,
    checkpoints: Sequence[Path],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    ds = MultiHorizonDataset(
        scaled_features,
        prepared.angles,
        prepared.test_indices,
        prepared.cfg.seq_len,
        spec.horizons,
        prepared.target_scaler,
    )
    loader = DataLoader(ds, batch_size=args.robustness_batch_size, shuffle=False, num_workers=0)
    all_seed_preds: List[np.ndarray] = []
    for checkpoint in checkpoints:
        model = build_model(spec, len(prepared.feature_columns), prepared.cfg.seq_len, parse_int_list("1,2,4,8")).to(device)
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        pred_chunks: List[np.ndarray] = []
        for x, _y in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x)
            if pred.ndim == 1:
                pred = pred.unsqueeze(1)
            pred_chunks.append(pred[:, 0].detach().cpu().numpy())
        pred_scaled = np.concatenate(pred_chunks, axis=0)
        all_seed_preds.append(prepared.target_scaler.inverse_transform(pred_scaled))
    pred = np.mean(np.stack(all_seed_preds, axis=0), axis=0)
    true_scaled = target_matrix(prepared.angles, prepared.test_indices, prepared.cfg.seq_len, spec.horizons)[:, 0]
    true = true_scaled.astype(np.float32, copy=False)
    return true, pred


def run_robustness(args: argparse.Namespace, records: Sequence[Any], device: torch.device) -> None:
    revision_dir = ensure_dir(Path(args.revision_dir).expanduser().resolve())
    table_dir = ensure_dir(revision_dir / "revision_tables")
    main_artifact_root = Path(args.main_artifact_root).expanduser().resolve()
    cache_dir = ensure_dir(main_artifact_root / "feature_cache")
    spec = transformer_spec(MAIN_MODEL)
    prep_args = build_ablation_args(args, main_artifact_root)
    prep_args.scales = "1,2,4,8"
    prep_args.input_noise_std = 0.0
    conditions = [
        "clean",
        "gaussian_noise_std_0.05",
        "gaussian_noise_std_0.10",
        "channel_dropout_RF",
        "channel_dropout_BF",
        "channel_dropout_VM",
        "channel_dropout_ST",
        "amplitude_scale_0.8",
        "amplitude_scale_1.2",
        "feature_smoothing_5",
    ]
    rows: List[Dict[str, Any]] = []
    for subject_i, record in enumerate(records, start=1):
        print(f"\n[robustness {subject_i}/{len(records)}] {record.subject} ({record.status})", flush=True)
        prepared = prepare_enhanced_subject(record, prep_args, spec, 42, cache_dir)
        subject_dir = main_artifact_root / "subjects" / record.subject / MAIN_MODEL
        checkpoints = sorted(subject_dir.glob("seed_*/checkpoint.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found under {subject_dir}")
        for condition in conditions:
            rng = np.random.default_rng(abs(hash((record.subject, condition))) % (2**32))
            scaled_features = make_perturbed_features(prepared, condition, rng)
            y_true, y_pred = ensemble_predict(prepared, spec, args, scaled_features, checkpoints, device)
            metrics = compute_metrics(y_true, y_pred)
            rows.append(
                {
                    "condition": condition,
                    "subject": record.subject,
                    "status": record.status,
                    "n_test": len(y_true),
                    "ME": metrics["me"],
                    "MAE": metrics["mae"],
                    "RMSE": metrics["rmse"],
                    "protocol": "checkpoint_reuse_purged_temporal_test_only_perturbation",
                }
            )
            print(f"  {condition}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}", flush=True)
    write_dict_csv(table_dir / "robustness_by_subject.csv", rows)

    clean_by_subject = {row["subject"]: row for row in rows if row["condition"] == "clean"}
    summary_rows: List[Dict[str, Any]] = []
    for condition in conditions:
        cond_rows = [row for row in rows if row["condition"] == condition]
        for group in ["overall", "healthy", "unhealthy"]:
            group_rows = cond_rows if group == "overall" else [row for row in cond_rows if row["status"] == group]
            if not group_rows:
                continue
            deltas_mae = []
            deltas_rmse = []
            for row in group_rows:
                clean = clean_by_subject[row["subject"]]
                deltas_mae.append(float(row["MAE"]) - float(clean["MAE"]))
                deltas_rmse.append(float(row["RMSE"]) - float(clean["RMSE"]))
            summary_rows.append(
                {
                    "condition": condition,
                    "group": group,
                    "n_subjects": len(group_rows),
                    "ME": mean(float(row["ME"]) for row in group_rows),
                    "MAE": mean(float(row["MAE"]) for row in group_rows),
                    "RMSE": mean(float(row["RMSE"]) for row in group_rows),
                    "delta_MAE_vs_clean": mean(deltas_mae),
                    "delta_RMSE_vs_clean": mean(deltas_rmse),
                    "protocol": "checkpoint_reuse_purged_temporal_test_only_perturbation",
                }
            )
    write_dict_csv(table_dir / "robustness_summary.csv", summary_rows)


def main() -> None:
    args = parse_args()
    data_root = find_dataset_root(Path(args.data_root))
    records = select_subjects(discover_subjects(data_root), args.subjects)
    if args.max_subjects > 0:
        records = records[: args.max_subjects]
    device = choose_device(args.device)
    print(f"Dataset root: {data_root}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Subjects ({len(records)}): {describe_records(records)}", flush=True)

    if args.run in {"ablation", "both"}:
        ablation_dir = run_nomultiscale_ablation(args, records, device)
        write_multiscale_ablation_tables(Path(args.main_artifact_root).expanduser().resolve(), ablation_dir, Path(args.revision_dir).expanduser().resolve())
    if args.run in {"robustness", "both"}:
        run_robustness(args, records, device)
    print("Reviewer numeric experiments completed.", flush=True)


if __name__ == "__main__":
    main()

