from __future__ import annotations

import argparse
import copy
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from causalgait.data import describe_records, discover_subjects, find_dataset_root, select_subjects
from causalgait.features import FEATURE_COLUMNS
from causalgait.train import (
    PipelineConfig,
    build_loss,
    build_pipeline_model,
    build_scheduler,
    compute_metrics,
    evaluate_loss,
    leakage_audit_for_subject,
    make_loader,
    predict,
    prepare_subject,
    save_leakage_audit,
    save_metrics_tables,
    save_paper_match_report,
    save_prediction_csv,
    save_prediction_plot,
    train_one_epoch,
    train_subject_model,
)
from causalgait.utils import choose_device, ensure_dir, parse_int_list, save_json, set_seed, stable_seed, write_dict_csv


PAPER_MAE = 3.707
PAPER_RMSE = 4.691


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run publication-safe CausalGait experiments.")
    parser.add_argument("--data-root", default="SEMG_DB1")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "artifacts_publishable"))
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--model", choices=["transformer", "mlp", "both"], default="both")
    parser.add_argument("--target-mae", type=float, default=PAPER_MAE)
    parser.add_argument("--target-rmse", type=float, default=PAPER_RMSE)
    parser.add_argument("--seeds", default="42,3407,2026")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--feature-window", type=int, default=250)
    parser.add_argument("--feature-step", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ff-dim", type=int, default=384)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--scales", default="1,2,4,8")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-subjects", type=int, default=0)
    return parser.parse_args()


def parse_seed_list(raw: str) -> List[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return seeds or [42]


def make_cfg(args: argparse.Namespace, seed: int, seed_label: str) -> PipelineConfig:
    return PipelineConfig(
        preset="publishable",
        evaluation_mode="publishable",
        split_mode="purged_temporal",
        scaler_scope="train",
        feature_window=args.feature_window,
        feature_step=args.feature_step,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_type="cosine",
        loss="huber",
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        best_checkpoint=True,
        seed=seed,
        seed_label=seed_label,
        num_workers=args.num_workers,
        d_model=args.d_model,
        patch_size=8,
        encoder_layers=args.encoder_layers,
        heads=args.heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        scales=parse_int_list(args.scales),
        plots=not args.no_plots,
        no_cache=args.no_cache,
    )


def train_transformer_seed(prepared, cfg: PipelineConfig, device: torch.device) -> Dict[str, Any]:
    seed = stable_seed(cfg.seed, prepared.record.subject, "publishable_transformer")
    set_seed(seed)
    model = build_pipeline_model("transformer", cfg).to(device)
    loss_fn = build_loss(cfg.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)
    train_loader = make_loader(prepared.train_dataset, cfg, shuffle=True, seed=seed, device=device)
    val_loader = make_loader(prepared.val_dataset, cfg, shuffle=False, seed=seed, device=device)
    test_loader = make_loader(prepared.test_dataset, cfg, shuffle=False, seed=seed, device=device)

    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, cfg.grad_clip)
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        lr = float(optimizer.param_groups[0]["lr"])
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if scheduler is not None:
            scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
        if epoch == 1 or epoch == cfg.epochs or epoch % max(10, cfg.epochs // 4) == 0:
            print(
                f"  {prepared.record.subject} seed={cfg.seed_label} epoch {epoch:03d}/{cfg.epochs}: "
                f"train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, lr={lr:.6g}",
                flush=True,
            )
        if cfg.early_stopping_patience > 0 and stale_epochs >= cfg.early_stopping_patience:
            print(
                f"  {prepared.record.subject} seed={cfg.seed_label} early stopping at epoch {epoch} "
                f"(best_val_loss={best_loss:.5f})",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred = predict(model, val_loader, device)
    test_pred = predict(model, test_loader, device)
    return {
        "history": history,
        "best_val_loss": best_loss,
        "val_true": val_pred["true"],
        "val_pred": val_pred["pred"],
        "test_true": test_pred["true"],
        "test_pred": test_pred["pred"],
        "state_dict": copy.deepcopy(model.state_dict()),
    }


def summarize_publication_readiness(output_dir: Path, rows: Sequence[Dict[str, Any]], target_mae: float, target_rmse: float) -> None:
    summary_path = output_dir / "metrics_summary.csv"
    overall_rows = [row for row in rows if row["model"] == "transformer_publishable_ensemble"]
    if overall_rows:
        overall = {
            "model": "transformer_publishable_ensemble",
            "n_subjects": len(overall_rows),
            "mae": float(np.mean([row["mae"] for row in overall_rows])),
            "rmse": float(np.mean([row["rmse"] for row in overall_rows])),
            "me": float(np.mean([row["me"] for row in overall_rows])),
        }
    else:
        overall = None
    beats = bool(overall and overall["mae"] < target_mae and overall["rmse"] < target_rmse)
    save_json(
        output_dir / "publication_readiness_report.json",
        {
            "publication_safe_protocol": True,
            "paper_chasing_results_usable_as_main_claim": False,
            "target": {"mae": target_mae, "rmse": target_rmse},
            "transformer_publishable_ensemble_overall": overall,
            "beats_paper_table_i": beats,
            "metrics_summary": summary_path,
            "notes": [
                "This run uses purged temporal splits and train-only Min-Max scaling.",
                "The ensemble averages independently trained model predictions only.",
                "No full-series scaler, random overlapping test split, nearest-label lookup, or overlap-aware label postprocessing is used.",
            ],
        },
    )


def run_transformer_ensemble(prepared, base_cfg: PipelineConfig, seeds: Sequence[int], output_dir: Path, device: torch.device) -> Dict[str, Any]:
    record = prepared.record
    subject_dir = ensure_dir(output_dir / "subjects" / record.subject / "transformer_publishable_ensemble")
    seed_label = ",".join(str(seed) for seed in seeds)
    test_preds = []
    test_true = None
    start_time = time.time()
    for seed in seeds:
        cfg = replace(base_cfg, seed=seed, seed_label=str(seed))
        result = train_transformer_seed(prepared, cfg, device)
        test_preds.append(result["test_pred"])
        test_true = result["test_true"]
        seed_dir = ensure_dir(subject_dir / f"seed_{seed}")
        save_prediction_csv(seed_dir / "val_predictions.csv", prepared.val_indices, cfg.seq_len, result["val_true"], result["val_pred"])
        save_prediction_csv(seed_dir / "test_predictions.csv", prepared.test_indices, cfg.seq_len, result["test_true"], result["test_pred"])
        write_dict_csv(seed_dir / "history.csv", result["history"], fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        torch.save(
            {
                "subject": record.subject,
                "status": record.status,
                "model_name": "transformer",
                "model_state_dict": result["state_dict"],
                "best_val_loss": result["best_val_loss"],
                "config": asdict(cfg),
                "scaler": prepared.scaler.to_dict(),
                "feature_columns": FEATURE_COLUMNS,
                "leakage_audit": leakage_audit_for_subject(prepared, cfg),
            },
            seed_dir / "checkpoint.pt",
        )

    ensemble_pred = np.mean(np.stack(test_preds, axis=0), axis=0)
    assert test_true is not None
    metrics = compute_metrics(test_true, ensemble_pred)
    save_prediction_csv(subject_dir / "ensemble_predictions.csv", prepared.test_indices, base_cfg.seq_len, test_true, ensemble_pred)
    if base_cfg.plots:
        save_prediction_plot(
            subject_dir / "ensemble_prediction_plot.png",
            record.subject,
            "transformer_publishable_ensemble",
            test_true,
            ensemble_pred,
            base_cfg.plot_max_points,
        )
    elapsed = time.time() - start_time
    save_json(
        subject_dir / "metadata.json",
        {
            "subject": record.subject,
            "status": record.status,
            "model": "transformer_publishable_ensemble",
            "seeds": list(seeds),
            "metrics": metrics,
            "config": asdict(replace(base_cfg, seed_label=seed_label)),
            "n_train": len(prepared.train_dataset),
            "n_val": len(prepared.val_dataset) if prepared.val_dataset is not None else 0,
            "n_test": len(prepared.test_dataset),
            "total_rows": prepared.total_rows,
            "valid_rows": prepared.source_rows,
            "skipped_rows": prepared.skipped_rows,
            "leakage_audit": leakage_audit_for_subject(prepared, base_cfg),
            "elapsed_seconds": elapsed,
        },
    )
    return {
        "subject": record.subject,
        "status": record.status,
        "model": "transformer_publishable_ensemble",
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
        "evaluation_mode": "publishable",
        "split_mode": base_cfg.split_mode,
        "scaler_scope": base_cfg.scaler_scope,
        "seed_or_ensemble": seed_label,
    }


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    seed_label = ",".join(str(seed) for seed in seeds)
    base_cfg = make_cfg(args, seeds[0], seed_label)
    set_seed(seeds[0])

    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    cache_dir = ensure_dir(output_dir / "feature_cache")
    data_root = find_dataset_root(Path(args.data_root))
    records = select_subjects(discover_subjects(data_root), args.subjects)
    if args.max_subjects > 0:
        records = records[: args.max_subjects]
    device = choose_device(args.device)

    save_json(
        output_dir / "run_config.json",
        {
            "data_root": data_root,
            "output_dir": output_dir,
            "subjects": [record.subject for record in records],
            "model": args.model,
            "seeds": seeds,
            "device": str(device),
            "config": asdict(base_cfg),
        },
    )

    print(f"Dataset root: {data_root}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Subjects ({len(records)}): {describe_records(records)}", flush=True)

    metric_rows: List[Dict[str, Any]] = []
    leakage_audits: List[Dict[str, Any]] = []
    for subject_i, record in enumerate(records, start=1):
        print(f"\n[publishable {subject_i}/{len(records)}] {record.subject} ({record.status})", flush=True)
        prepared = prepare_subject(record, base_cfg, cache_dir)
        leakage_audits.append(leakage_audit_for_subject(prepared, base_cfg))
        print(
            f"  split train/val/test={len(prepared.train_dataset)}/"
            f"{len(prepared.val_dataset) if prepared.val_dataset is not None else 0}/"
            f"{len(prepared.test_dataset)}",
            flush=True,
        )
        if args.model in {"transformer", "both"}:
            row = run_transformer_ensemble(prepared, base_cfg, seeds, output_dir, device)
            metric_rows.append(row)
            print(f"  transformer ensemble: MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}", flush=True)
        if args.model in {"mlp", "both"}:
            mlp_cfg = replace(base_cfg, seed=seeds[0], seed_label=str(seeds[0]))
            row = train_subject_model(prepared, "mlp", mlp_cfg, output_dir, device)
            metric_rows.append(row)
            print(f"  mlp: MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}", flush=True)

        save_metrics_tables(output_dir, metric_rows)
        save_leakage_audit(output_dir, leakage_audits)
        summarize_publication_readiness(output_dir, metric_rows, args.target_mae, args.target_rmse)

    save_metrics_tables(output_dir, metric_rows)
    save_leakage_audit(output_dir, leakage_audits)
    save_paper_match_report(output_dir, metric_rows, base_cfg, len(records))
    summarize_publication_readiness(output_dir, metric_rows, args.target_mae, args.target_rmse)
    print("\nDone. Publication-safe artifacts written to:")
    print(f"  {output_dir / 'metrics_summary.csv'}")
    print(f"  {output_dir / 'leakage_audit.json'}")
    print(f"  {output_dir / 'publication_readiness_report.json'}")


if __name__ == "__main__":
    main()
