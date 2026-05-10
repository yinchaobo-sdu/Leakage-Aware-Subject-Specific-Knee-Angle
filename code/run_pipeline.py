from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List

from causalgait.data import describe_records, discover_subjects, find_dataset_root, select_subjects
from causalgait.train import (
    PipelineConfig,
    leakage_audit_for_subject,
    prepare_subject,
    save_leakage_audit,
    save_metrics_tables,
    save_paper_match_report,
    train_subject_model,
)
from causalgait.utils import choose_device, ensure_dir, parse_int_list, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transformer-based knee angle prediction from sEMG.")
    parser.add_argument("--data-root", default="SEMG_DB1", help="Path to SEMG_DB1 dataset root.")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "artifacts"))
    parser.add_argument("--preset", choices=["strict", "paper_chasing", "publishable"], default="strict")
    parser.add_argument("--split-mode", choices=["temporal", "random_overlap", "purged_temporal"], default=None)
    parser.add_argument("--scaler-scope", choices=["train", "full"], default=None)
    parser.add_argument("--model", choices=["transformer", "mlp", "both"], default="both")
    parser.add_argument("--subjects", default="all", help="all or comma-separated subject stems, e.g. 1Nmar,1Amar.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--feature-window", type=int, default=250)
    parser.add_argument("--feature-step", type=int, default=1)
    parser.add_argument("--scales", default="1,2,4")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--scheduler-step", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-type", choices=["step", "cosine", "none"], default=None)
    parser.add_argument("--loss", choices=["mse", "huber", "smooth_l1"], default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--best-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-max-points", type=int, default=3000)
    parser.add_argument("--max-subjects", type=int, default=0, help="Optional debug limit; 0 means no limit.")
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "strict":
        if args.split_mode is None:
            args.split_mode = "temporal"
        if args.scaler_scope is None:
            args.scaler_scope = "train"
        if args.scheduler_type is None:
            args.scheduler_type = "step"
        if args.loss is None:
            args.loss = "mse"
        if args.val_ratio is None:
            args.val_ratio = 0.0
        if args.test_ratio is None:
            args.test_ratio = 1.0 - args.train_ratio
        return args

    if args.preset == "publishable":
        if args.split_mode is None:
            args.split_mode = "purged_temporal"
        if args.scaler_scope is None:
            args.scaler_scope = "train"
        if args.scheduler_type is None:
            args.scheduler_type = "cosine"
        if args.loss is None:
            args.loss = "huber"
        if args.val_ratio is None:
            args.val_ratio = 0.1
        if args.test_ratio is None:
            args.test_ratio = 0.2
        if args.epochs == 30:
            args.epochs = 120
        if args.lr == 1e-3:
            args.lr = 5e-4
        if args.d_model == 128:
            args.d_model = 192
        if args.heads == 4:
            args.heads = 6
        if args.ff_dim == 256:
            args.ff_dim = 384
        if args.encoder_layers == 1:
            args.encoder_layers = 2
        if abs(args.dropout - 0.1) < 1e-12:
            args.dropout = 0.05
        if args.scales == "1,2,4":
            args.scales = "1,2,4,8"
        if args.grad_clip == 0.0:
            args.grad_clip = 1.0
        if args.early_stopping_patience == 0:
            args.early_stopping_patience = 20
        args.best_checkpoint = True
        return args

    if args.split_mode is None:
        args.split_mode = "random_overlap"
    if args.scaler_scope is None:
        args.scaler_scope = "full"
    if args.scheduler_type is None:
        args.scheduler_type = "cosine"
    if args.loss is None:
        args.loss = "huber"
    if args.val_ratio is None:
        args.val_ratio = 0.1
    if args.test_ratio is None:
        args.test_ratio = 0.2
    if args.epochs == 30:
        args.epochs = 120
    if args.lr == 1e-3:
        args.lr = 5e-4
    if args.d_model == 128:
        args.d_model = 192
    if args.heads == 4:
        args.heads = 6
    if args.ff_dim == 256:
        args.ff_dim = 384
    if args.encoder_layers == 1:
        args.encoder_layers = 2
    if abs(args.dropout - 0.1) < 1e-12:
        args.dropout = 0.05
    if args.scales == "1,2,4":
        args.scales = "1,2,4,8"
    args.best_checkpoint = True
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.preset == "publishable":
        if args.split_mode != "purged_temporal":
            raise ValueError("--preset publishable requires --split-mode purged_temporal.")
        if args.scaler_scope != "train":
            raise ValueError("--preset publishable requires --scaler-scope train.")
    if args.preset != "paper_chasing" and args.scaler_scope == "full":
        raise ValueError("Full-series scaling is only allowed for --preset paper_chasing.")


def model_list(choice: str) -> List[str]:
    if choice == "both":
        return ["transformer", "mlp"]
    return [choice]


def main() -> None:
    args = parse_args()
    args = apply_preset(args)
    validate_args(args)
    set_seed(args.seed)

    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    cache_dir = ensure_dir(output_dir / "feature_cache")
    data_root = find_dataset_root(Path(args.data_root))
    records = select_subjects(discover_subjects(data_root), args.subjects)
    if args.max_subjects and args.max_subjects > 0:
        records = records[: args.max_subjects]

    cfg = PipelineConfig(
        preset=args.preset,
        evaluation_mode=args.preset,
        split_mode=args.split_mode,
        scaler_scope=args.scaler_scope,
        feature_window=args.feature_window,
        feature_step=args.feature_step,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_type=args.scheduler_type,
        loss=args.loss,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        best_checkpoint=args.best_checkpoint,
        seed=args.seed,
        seed_label=str(args.seed),
        num_workers=args.num_workers,
        d_model=args.d_model,
        patch_size=args.patch_size,
        encoder_layers=args.encoder_layers,
        heads=args.heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        scales=parse_int_list(args.scales),
        plots=not args.no_plots,
        plot_max_points=args.plot_max_points,
        no_cache=args.no_cache,
    )
    device = choose_device(args.device)

    save_json(
        output_dir / "run_config.json",
        {
            "data_root": data_root,
            "output_dir": output_dir,
            "subjects": [record.subject for record in records],
            "models": model_list(args.model),
            "device": str(device),
            "config": asdict(cfg),
        },
    )

    print(f"Dataset root: {data_root}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Subjects ({len(records)}): {describe_records(records)}", flush=True)
    print(f"Models: {', '.join(model_list(args.model))}", flush=True)

    metric_rows = []
    leakage_audits = []
    for subject_idx, record in enumerate(records, start=1):
        print(f"\n[{subject_idx}/{len(records)}] Preparing {record.subject} ({record.status})", flush=True)
        prepared = prepare_subject(record, cfg, cache_dir=cache_dir)
        leakage_audits.append(leakage_audit_for_subject(prepared, cfg))
        print(
            f"  rows={prepared.source_rows}/{prepared.total_rows} valid, skipped={prepared.skipped_rows}, "
            f"features={prepared.n_feature_rows}, "
            f"train={len(prepared.train_dataset)}, "
            f"val={len(prepared.val_dataset) if prepared.val_dataset is not None else 0}, "
            f"test={len(prepared.test_dataset)}",
            flush=True,
        )
        for model_name in model_list(args.model):
            print(f"  Training {model_name}", flush=True)
            row = train_subject_model(prepared, model_name, cfg, output_dir, device)
            metric_rows.append(row)
            save_metrics_tables(output_dir, metric_rows)
            print(
                f"  {model_name} metrics: ME={row['me']:.4f}, "
                f"MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}",
                flush=True,
            )

    save_metrics_tables(output_dir, metric_rows)
    save_leakage_audit(output_dir, leakage_audits)
    save_paper_match_report(output_dir, metric_rows, cfg, n_subjects=len(records))
    print("\nDone. Metrics written to:")
    print(f"  {output_dir / 'metrics_by_subject.csv'}")
    print(f"  {output_dir / 'metrics_summary.csv'}")
    print(f"  {output_dir / 'leakage_audit.json'}")
    print(f"  {output_dir / 'paper_match_report.json'}")


if __name__ == "__main__":
    main()
