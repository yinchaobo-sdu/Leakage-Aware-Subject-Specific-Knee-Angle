from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from causalgait.data import discover_subjects, find_dataset_root, read_emg_csv
from causalgait.features import feature_columns_for_channels
from causalgait.utils import ensure_dir, save_json, write_dict_csv


PAPER_MAE = 3.707
PAPER_RMSE = 4.691
MAIN_MODEL = "transformer_4ch_extended"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reviewer-response tables, figures, and statistics.")
    parser.add_argument("--data-root", default="SEMG_DB1")
    parser.add_argument("--artifact-root", default=str(Path(__file__).resolve().parent / "artifacts_enhanced_publishable_remote"))
    parser.add_argument("--strict-artifact-root", default=str(Path(__file__).resolve().parent / "artifacts_publishable_remote"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "revision_package"))
    parser.add_argument("--loso-step", type=int, default=10)
    parser.add_argument("--traditional-step", type=int, default=20)
    parser.add_argument("--traditional-max-train", type=int, default=2500)
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def me(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def copy_core_artifacts(artifact_root: Path, output_dir: Path) -> None:
    for name in [
        "metrics_summary.csv",
        "metrics_by_subject.csv",
        "validation_selection_report.json",
        "leakage_audit.json",
        "publication_readiness_report.json",
        "statistical_report.json",
    ]:
        src = artifact_root / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def log_path_for_subject(data_root: Path, subject: str, status: str) -> Path:
    folder = "N_LOG" if status == "healthy" else "A_LOG"
    return data_root / folder / f"{subject}.log"


def parse_log_header(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_bytes()[:2048].decode("latin-1", errors="ignore")
    result: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("Start="):
            result["start"] = line.split("=", 1)[1]
        elif line.startswith("End="):
            result["end"] = line.split("=", 1)[1]
        elif line.startswith("Recorded="):
            result["recorded"] = line.split("=", 1)[1]
    return result


def main_rows(artifact_root: Path) -> List[Dict[str, str]]:
    rows = read_csv(artifact_root / "metrics_by_subject.csv")
    return [
        row
        for row in rows
        if row["model"] == MAIN_MODEL and row["split"] == "test" and row.get("main_result", "").lower() == "true"
    ]


def generate_dataset_summary(data_root: Path, artifact_root: Path, table_dir: Path) -> None:
    metric_rows = {row["subject"]: row for row in main_rows(artifact_root)}
    records = discover_subjects(data_root)
    rows: List[Dict[str, Any]] = []
    for record in records:
        raw = read_emg_csv(record.path)
        metric = metric_rows.get(record.subject, {})
        log_header = parse_log_header(log_path_for_subject(data_root, record.subject, record.status))
        rows.append(
            {
                "subject": record.subject,
                "status": record.status,
                "sex": "male",
                "source_file": record.path.name,
                "duration_s_1000hz": round(raw.valid_rows / 1000.0, 3),
                "valid_rows": raw.valid_rows,
                "skipped_rows": raw.skipped_rows,
                "start_time_from_log": log_header.get("start", ""),
                "end_time_from_log": log_header.get("end", ""),
                "n_train_windows": metric.get("n_train", ""),
                "n_val_windows": metric.get("n_val", ""),
                "n_test_windows": metric.get("n_test", ""),
                "feature_rows": metric.get("feature_rows", ""),
                "main_mae": metric.get("mae", ""),
                "main_rmse": metric.get("rmse", ""),
            }
        )
    write_dict_csv(table_dir / "dataset_summary.csv", rows)


def generate_model_tables(artifact_root: Path, strict_artifact_root: Path, table_dir: Path) -> None:
    summary_rows = read_csv(artifact_root / "metrics_summary.csv")
    rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        if row["group"] != "overall":
            continue
        rows.append(
            {
                "protocol": "enhanced_publishable_purged_temporal",
                "model": row["model"],
                "n_subjects": row["n_subjects"],
                "main_result": row.get("main_result", "False"),
                "ME": row["me"],
                "MAE": row["mae"],
                "RMSE": row["rmse"],
                "notes": "Validation-selected main result" if row.get("main_result", "").lower() == "true" else "Exploratory candidate/baseline",
            }
        )
    if (strict_artifact_root / "metrics_summary.csv").exists():
        for row in read_csv(strict_artifact_root / "metrics_summary.csv"):
            if row["group"] == "overall":
                rows.append(
                    {
                        "protocol": "strict_publishable_purged_temporal",
                        "model": row["model"],
                        "n_subjects": row["n_subjects"],
                        "main_result": "historical_baseline",
                        "ME": row["me"],
                        "MAE": row["mae"],
                        "RMSE": row["rmse"],
                        "notes": "Earlier two-channel publishable run",
                    }
                )
    rows.append(
        {
            "protocol": "reviewer_requested_traditional_baseline",
            "model": "RF/SVR/XGBoost",
            "n_subjects": "",
            "main_result": "see_traditional_baseline_summary",
            "ME": "",
            "MAE": "",
            "RMSE": "",
            "notes": "See traditional_baseline_summary.csv when generated.",
        }
    )
    write_dict_csv(table_dir / "baseline_comparison.csv", rows)

    ablation_rows = [
        {
            "comparison": "2-channel original Transformer vs 4-channel extended Transformer",
            "baseline_model": "transformer_rf_vm_original",
            "enhanced_model": "transformer_4ch_extended",
        },
        {
            "comparison": "4-channel Transformer vs CausalGaitNet",
            "baseline_model": "transformer_4ch_extended",
            "enhanced_model": "causalgaitnet_4ch_extended",
        },
        {
            "comparison": "single-horizon vs multi-horizon CausalGaitNet",
            "baseline_model": "causalgaitnet_4ch_extended",
            "enhanced_model": "causalgaitnet_multihorizon_4ch_extended",
        },
    ]
    by_model = {row["model"]: row for row in summary_rows if row["group"] == "overall"}
    for row in ablation_rows:
        base = by_model.get(row["baseline_model"], {})
        enh = by_model.get(row["enhanced_model"], {})
        row.update(
            {
                "baseline_MAE": base.get("mae", ""),
                "baseline_RMSE": base.get("rmse", ""),
                "enhanced_MAE": enh.get("mae", ""),
                "enhanced_RMSE": enh.get("rmse", ""),
                "delta_MAE": float(enh["mae"]) - float(base["mae"]) if base and enh else "",
                "delta_RMSE": float(enh["rmse"]) - float(base["rmse"]) if base and enh else "",
            }
        )
    write_dict_csv(table_dir / "ablation_model_components.csv", ablation_rows)

    write_dict_csv(
        table_dir / "ablation_multiscale.csv",
        [
            {
                "ablation": "No-multiscale branch",
                "status": "planned_required_experiment",
                "implementation_note": "Run Transformer with --scales 1 only under purged temporal protocol; compare with scales 1,2,4,8.",
            },
            {
                "ablation": "Multi-scale branch",
                "status": "completed",
                "model": MAIN_MODEL,
                "scales": "1,2,4,8",
                "MAE": by_model.get(MAIN_MODEL, {}).get("mae", ""),
                "RMSE": by_model.get(MAIN_MODEL, {}).get("rmse", ""),
            },
        ],
    )


def bootstrap_ci(values: Sequence[float], rng: np.random.Generator, iters: int) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    samples = rng.choice(arr, size=(iters, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def permutation_p(left: Sequence[float], right: Sequence[float], rng: np.random.Generator, iters: int) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    observed = abs(left_arr.mean() - right_arr.mean())
    pooled = np.concatenate([left_arr, right_arr])
    count = 0
    for _ in range(iters):
        rng.shuffle(pooled)
        diff = abs(pooled[: len(left_arr)].mean() - pooled[len(left_arr) :].mean())
        count += int(diff >= observed)
    return float((count + 1) / (iters + 1))


def cliffs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    comparisons = np.sign(left_arr[:, None] - right_arr[None, :])
    return float(comparisons.sum() / comparisons.size)


def generate_statistics(artifact_root: Path, table_dir: Path, output_dir: Path, rng: np.random.Generator, bootstrap_iters: int) -> None:
    rows = main_rows(artifact_root)
    for row in rows:
        row["mae_f"] = float(row["mae"])
        row["rmse_f"] = float(row["rmse"])
        row["me_f"] = float(row["me"])

    stats: Dict[str, Any] = {"main_model": MAIN_MODEL, "target": {"MAE": PAPER_MAE, "RMSE": PAPER_RMSE}}
    for metric_key, label in [("mae_f", "MAE"), ("rmse_f", "RMSE")]:
        values = [float(row[metric_key]) for row in rows]
        lo, hi = bootstrap_ci(values, rng, bootstrap_iters)
        stats[label] = {
            "mean": mean(values),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)),
            "bootstrap_95ci": [lo, hi],
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "subjects_below_paper_threshold": int(sum(v < (PAPER_MAE if label == "MAE" else PAPER_RMSE) for v in values)),
        }

    group_rows: List[Dict[str, Any]] = []
    for status in ["healthy", "unhealthy"]:
        subset = [row for row in rows if row["status"] == status]
        for metric_key, label in [("mae_f", "MAE"), ("rmse_f", "RMSE")]:
            values = [float(row[metric_key]) for row in subset]
            lo, hi = bootstrap_ci(values, rng, bootstrap_iters)
            group_rows.append(
                {
                    "group": status,
                    "metric": label,
                    "n_subjects": len(values),
                    "mean": mean(values),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=1)),
                    "bootstrap_95ci_low": lo,
                    "bootstrap_95ci_high": hi,
                }
            )
    write_dict_csv(table_dir / "healthy_unhealthy_statistics.csv", group_rows)

    comparisons = []
    for metric_key, label in [("mae_f", "MAE"), ("rmse_f", "RMSE")]:
        healthy = [float(row[metric_key]) for row in rows if row["status"] == "healthy"]
        unhealthy = [float(row[metric_key]) for row in rows if row["status"] == "unhealthy"]
        comparisons.append(
            {
                "metric": label,
                "healthy_mean": mean(healthy),
                "unhealthy_mean": mean(unhealthy),
                "difference_healthy_minus_unhealthy": mean(healthy) - mean(unhealthy),
                "permutation_p_two_sided": permutation_p(healthy, unhealthy, rng, bootstrap_iters),
                "cliffs_delta_healthy_vs_unhealthy": cliffs_delta(healthy, unhealthy),
                "test": "two-sided permutation test on subject-level means",
            }
        )
    write_dict_csv(table_dir / "statistical_tests.csv", comparisons)
    stats["healthy_unhealthy_tests"] = comparisons
    save_json(output_dir / "statistical_tests.json", stats)

    extremes = []
    sorted_rows = sorted(rows, key=lambda r: float(r["mae"]))
    for label, selected in [("best", sorted_rows[:5]), ("worst", sorted_rows[-5:])]:
        for row in selected:
            extremes.append(
                {
                    "rank_group": label,
                    "subject": row["subject"],
                    "status": row["status"],
                    "ME": row["me"],
                    "MAE": row["mae"],
                    "RMSE": row["rmse"],
                }
            )
    write_dict_csv(table_dir / "subject_performance_extremes.csv", extremes)


def load_feature_cache(artifact_root: Path, subject: str) -> Tuple[np.ndarray, np.ndarray]:
    matches = list((artifact_root / "feature_cache").glob(f"{subject}_chRF-BF-VM-ST_setextended_win250_step1.npz"))
    if not matches:
        raise FileNotFoundError(f"Feature cache missing for {subject}")
    with np.load(matches[0], allow_pickle=False) as cached:
        return cached["features"].astype(np.float32), cached["angles"].astype(np.float32)


def generate_feature_correlation(artifact_root: Path, records: Sequence[Any], table_dir: Path, fig_dir: Path, rng: np.random.Generator) -> None:
    chunks = []
    for record in records:
        features, _ = load_feature_cache(artifact_root, record.subject)
        n = min(800, len(features))
        indices = rng.choice(len(features), size=n, replace=False)
        chunks.append(features[indices])
    values = np.concatenate(chunks, axis=0)
    columns = feature_columns_for_channels(["RF", "BF", "VM", "ST"], "extended")
    corr = np.corrcoef(values, rowvar=False)
    rows = []
    for i, left in enumerate(columns):
        row = {"feature": left}
        for j, right in enumerate(columns):
            row[right] = float(corr[i, j])
        rows.append(row)
    write_dict_csv(table_dir / "feature_correlation.csv", rows, fieldnames=["feature", *columns])

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(label="Pearson r")
    plt.xticks(range(len(columns)), columns, rotation=90, fontsize=5)
    plt.yticks(range(len(columns)), columns, fontsize=5)
    plt.title("Extended four-channel feature correlation matrix")
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_correlation_matrix.png", dpi=220)
    plt.close()


def generate_loso_ridge(artifact_root: Path, records: Sequence[Any], table_dir: Path, loso_step: int) -> None:
    subject_data = {}
    for record in records:
        features, angles = load_feature_cache(artifact_root, record.subject)
        idx = np.arange(0, min(len(features), len(angles)), max(1, loso_step), dtype=np.int64)
        subject_data[record.subject] = {
            "status": record.status,
            "x": features[idx].astype(np.float64),
            "y": angles[idx].astype(np.float64),
        }

    rows = []
    lam = 1.0
    for heldout in records:
        train_x = np.concatenate([subject_data[r.subject]["x"] for r in records if r.subject != heldout.subject], axis=0)
        train_y = np.concatenate([subject_data[r.subject]["y"] for r in records if r.subject != heldout.subject], axis=0)
        test_x = subject_data[heldout.subject]["x"]
        test_y = subject_data[heldout.subject]["y"]
        mean_x = train_x.mean(axis=0)
        scale_x = train_x.std(axis=0)
        scale_x = np.where(scale_x == 0, 1.0, scale_x)
        train_xs = (train_x - mean_x) / scale_x
        test_xs = (test_x - mean_x) / scale_x
        design = np.concatenate([np.ones((len(train_xs), 1)), train_xs], axis=1)
        reg = np.eye(design.shape[1]) * lam
        reg[0, 0] = 0.0
        coef = np.linalg.solve(design.T @ design + reg, design.T @ train_y)
        pred = np.concatenate([np.ones((len(test_xs), 1)), test_xs], axis=1) @ coef
        rows.append(
            {
                "held_out_subject": heldout.subject,
                "status": heldout.status,
                "model": "LOSO_ridge_4ch_extended_nonsequence",
                "n_train_samples_downsampled": int(len(train_y)),
                "n_test_samples_downsampled": int(len(test_y)),
                "downsample_step": int(loso_step),
                "ME": me(test_y, pred),
                "MAE": mae(test_y, pred),
                "RMSE": rmse(test_y, pred),
                "note": "Cross-subject ridge baseline using extended feature rows; not sequence Transformer.",
            }
        )
    write_dict_csv(table_dir / "loso_metrics.csv", rows)
    summary = []
    for group in ["overall", "healthy", "unhealthy"]:
        subset = rows if group == "overall" else [row for row in rows if row["status"] == group]
        summary.append(
            {
                "model": "LOSO_ridge_4ch_extended_nonsequence",
                "group": group,
                "n_subjects": len(subset),
                "ME": mean(float(row["ME"]) for row in subset),
                "MAE": mean(float(row["MAE"]) for row in subset),
                "RMSE": mean(float(row["RMSE"]) for row in subset),
            }
        )
    write_dict_csv(table_dir / "loso_summary.csv", summary)


def feature_cache_path(artifact_root: Path, subject: str, channels: str = "RF-BF-VM-ST", feature_set: str = "extended") -> Path:
    return artifact_root / "feature_cache" / f"{subject}_ch{channels}_set{feature_set}_win250_step1.npz"


def load_subject_split_audit(artifact_root: Path, subject: str, model: str = MAIN_MODEL) -> Dict[str, Any]:
    path = artifact_root / "subjects" / subject / model / "metadata.json"
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata["leakage_audit"]


def rows_from_audit_subset(audit: Dict[str, Any], subset: str, step: int) -> Tuple[np.ndarray, np.ndarray]:
    info = audit["subsets"][subset]
    labels = np.arange(int(info["label_min"]), int(info["label_max"]) + 1, dtype=np.int64)
    labels = labels[:: max(1, step)]
    x_rows = labels - 1
    return x_rows, labels


def select_train_rows(
    x_rows: np.ndarray,
    y_rows: np.ndarray,
    max_train: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_train <= 0 or x_rows.size <= max_train:
        return x_rows, y_rows
    chosen = np.sort(rng.choice(x_rows.size, size=max_train, replace=False))
    return x_rows[chosen], y_rows[chosen]


def generate_traditional_baselines(
    artifact_root: Path,
    records: Sequence[Any],
    table_dir: Path,
    rng: np.random.Generator,
    step: int,
    max_train: int,
) -> None:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover - optional dependency guard
        write_dict_csv(
            table_dir / "traditional_baseline_summary.csv",
            [
                {
                    "model": "RF/SVR/XGBoost",
                    "group": "overall",
                    "n_subjects": 0,
                    "ME": "",
                    "MAE": "",
                    "RMSE": "",
                    "status": f"skipped: {exc}",
                }
            ],
        )
        return

    tuning_ranges = {
        "random_forest": [
            {"max_depth": 8, "min_samples_leaf": 2},
            {"max_depth": 14, "min_samples_leaf": 1},
        ],
        "svr_rbf": [
            {"C": 1.0, "epsilon": 0.5},
            {"C": 10.0, "epsilon": 0.5},
        ],
        "xgboost": [
            {"max_depth": 3, "learning_rate": 0.05},
            {"max_depth": 5, "learning_rate": 0.05},
        ],
    }
    per_subject_rows: List[Dict[str, Any]] = []

    for record in records:
        cache_path = feature_cache_path(artifact_root, record.subject)
        if not cache_path.exists():
            continue
        data = np.load(cache_path, allow_pickle=True)
        features = data["features"].astype(np.float32)
        angles = data["angles"].astype(np.float32)
        audit = load_subject_split_audit(artifact_root, record.subject)

        train_x_rows, train_y_rows = rows_from_audit_subset(audit, "train", step)
        val_x_rows, val_y_rows = rows_from_audit_subset(audit, "val", step)
        test_x_rows, test_y_rows = rows_from_audit_subset(audit, "test", step)
        train_x_rows, train_y_rows = select_train_rows(train_x_rows, train_y_rows, max_train, rng)

        x_train, y_train = features[train_x_rows], angles[train_y_rows]
        x_val, y_val = features[val_x_rows], angles[val_y_rows]
        x_test, y_test = features[test_x_rows], angles[test_y_rows]

        model_candidates: Dict[str, List[Tuple[str, Any]]] = {
            "random_forest": [
                (
                    f"depth={params['max_depth']},leaf={params['min_samples_leaf']}",
                    make_pipeline(
                        StandardScaler(),
                        RandomForestRegressor(
                            n_estimators=120,
                            max_depth=params["max_depth"],
                            min_samples_leaf=params["min_samples_leaf"],
                            random_state=2026,
                            n_jobs=-1,
                        ),
                    ),
                )
                for params in tuning_ranges["random_forest"]
            ],
            "svr_rbf": [
                (
                    f"C={params['C']},eps={params['epsilon']}",
                    make_pipeline(StandardScaler(), SVR(C=params["C"], epsilon=params["epsilon"], kernel="rbf")),
                )
                for params in tuning_ranges["svr_rbf"]
            ],
            "xgboost": [
                (
                    f"depth={params['max_depth']},lr={params['learning_rate']}",
                    make_pipeline(
                        StandardScaler(),
                        XGBRegressor(
                            n_estimators=160,
                            max_depth=params["max_depth"],
                            learning_rate=params["learning_rate"],
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="reg:squarederror",
                            random_state=2026,
                            n_jobs=2,
                            verbosity=0,
                        ),
                    ),
                )
                for params in tuning_ranges["xgboost"]
            ],
        }

        for model_name, candidates in model_candidates.items():
            best_model = None
            best_label = ""
            best_val_mae = float("inf")
            for label, candidate in candidates:
                candidate.fit(x_train, y_train)
                val_pred = candidate.predict(x_val)
                val_mae = float(mean_absolute_error(y_val, val_pred))
                if val_mae < best_val_mae:
                    best_model = candidate
                    best_label = label
                    best_val_mae = val_mae
            if best_model is None:
                continue
            pred = np.asarray(best_model.predict(x_test), dtype=np.float64)
            err = pred - y_test.astype(np.float64)
            per_subject_rows.append(
                {
                    "model": model_name,
                    "subject": record.subject,
                    "status": record.status,
                    "n_train": int(len(y_train)),
                    "n_val": int(len(y_val)),
                    "n_test": int(len(y_test)),
                    "selected_params": best_label,
                    "val_mae": best_val_mae,
                    "ME": float(np.mean(err)),
                    "MAE": float(mean_absolute_error(y_test, pred)),
                    "RMSE": float(math.sqrt(mean_squared_error(y_test, pred))),
                }
            )

    write_dict_csv(table_dir / "traditional_baselines_by_subject.csv", per_subject_rows)
    summary_rows: List[Dict[str, Any]] = []
    for model_name in sorted({row["model"] for row in per_subject_rows}):
        model_rows = [row for row in per_subject_rows if row["model"] == model_name]
        for group in ["overall", "healthy", "unhealthy"]:
            group_rows = model_rows if group == "overall" else [row for row in model_rows if row["status"] == group]
            if not group_rows:
                continue
            summary_rows.append(
                {
                    "model": model_name,
                    "group": group,
                    "n_subjects": len(group_rows),
                    "ME": mean(float(row["ME"]) for row in group_rows),
                    "MAE": mean(float(row["MAE"]) for row in group_rows),
                    "RMSE": mean(float(row["RMSE"]) for row in group_rows),
                    "protocol": "purged_temporal_train_only_scaler_single_feature_row",
                    "tuning": json.dumps(tuning_ranges[model_name], sort_keys=True),
                    "sample_step": step,
                    "max_train_per_subject": max_train,
                }
            )
    write_dict_csv(table_dir / "traditional_baseline_summary.csv", summary_rows)


def generate_robustness_summary(table_dir: Path) -> None:
    rows = [
        {
            "perturbation": "Gaussian EMG feature noise",
            "status": "planned_required_experiment",
            "protocol": "Apply noise to test input features only after train-only scaling; evaluate selected checkpoint without retraining.",
        },
        {
            "perturbation": "single-channel dropout",
            "status": "planned_required_experiment",
            "protocol": "Set one channel feature block to zero at test time; report MAE/RMSE degradation per channel.",
        },
        {
            "perturbation": "amplitude scaling",
            "status": "planned_required_experiment",
            "protocol": "Scale raw sEMG channels by 0.8x and 1.2x before feature extraction; reuse train-only scaler.",
        },
        {
            "perturbation": "filter/preprocessing variation",
            "status": "planned_required_experiment",
            "protocol": "Compare no-filter baseline with a standard EMG band-pass/envelope preprocessing pipeline.",
        },
    ]
    write_dict_csv(table_dir / "robustness_summary.csv", rows)


def plot_prediction(path: Path, out_path: Path, title: str) -> None:
    rows = read_csv(path)
    y_true = np.asarray([float(row["y_true"]) for row in rows], dtype=np.float64)
    y_pred = np.asarray([float(row["y_pred"]) for row in rows], dtype=np.float64)
    n = min(1500, len(y_true))
    plt.figure(figsize=(11, 4))
    plt.plot(y_true[:n], label="Measured FX", linewidth=1.4)
    plt.plot(y_pred[:n], label="Predicted", linewidth=1.0)
    plt.xlabel("Test window")
    plt.ylabel("Knee angle (deg)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def generate_prediction_figures(artifact_root: Path, fig_dir: Path) -> None:
    rows = main_rows(artifact_root)
    for status in ["healthy", "unhealthy"]:
        candidates = [row for row in rows if row["status"] == status]
        chosen = sorted(candidates, key=lambda row: float(row["mae"]))[0]
        pred_path = artifact_root / "subjects" / chosen["subject"] / MAIN_MODEL / "ensemble_predictions.csv"
        plot_prediction(
            pred_path,
            fig_dir / f"prediction_{status}_{chosen['subject']}.png",
            f"{chosen['subject']} ({status}) - {MAIN_MODEL}",
        )


def generate_schematics(fig_dir: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    ax = plt.gca()
    ax.axis("off")
    boxes = [
        (0.05, 0.62, "RF\nrectus femoris"),
        (0.05, 0.38, "BF\nbiceps femoris"),
        (0.32, 0.62, "VM\nvastus medialis"),
        (0.32, 0.38, "ST\nsemitendinosus"),
        (0.62, 0.5, "FX goniometer\nknee flexion"),
        (0.82, 0.5, "Model\nangle prediction"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.16, 0.14, fill=False, linewidth=1.6))
        ax.text(x + 0.08, y + 0.07, text, ha="center", va="center", fontsize=10)
    for x1, y1, x2, y2 in [(0.21, 0.69, 0.62, 0.57), (0.21, 0.45, 0.62, 0.57), (0.48, 0.69, 0.62, 0.57), (0.48, 0.45, 0.62, 0.57), (0.78, 0.57, 0.82, 0.57)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.text(0.5, 0.9, "Four-channel sEMG and knee goniometry setup", ha="center", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "experimental_setup_schematic.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4.8))
    ax = plt.gca()
    ax.axis("off")
    ax.text(0.25, 0.9, "Leaky random-overlap split", ha="center", fontsize=12, weight="bold")
    ax.text(0.75, 0.9, "Purged temporal split", ha="center", fontsize=12, weight="bold")
    for i in range(6):
        ax.add_patch(plt.Rectangle((0.05 + i * 0.055, 0.55), 0.18, 0.08, fill=False, edgecolor="tab:red", linewidth=1.2))
    ax.text(0.25, 0.42, "Highly overlapping windows can enter\ntrain and test simultaneously.", ha="center", fontsize=10)
    ax.add_patch(plt.Rectangle((0.55, 0.58), 0.18, 0.08, color="tab:blue", alpha=0.3))
    ax.add_patch(plt.Rectangle((0.74, 0.58), 0.07, 0.08, color="gray", alpha=0.25))
    ax.add_patch(plt.Rectangle((0.82, 0.58), 0.13, 0.08, color="tab:orange", alpha=0.35))
    ax.text(0.64, 0.72, "train", ha="center", fontsize=10)
    ax.text(0.775, 0.72, "purge", ha="center", fontsize=10)
    ax.text(0.885, 0.72, "test", ha="center", fontsize=10)
    ax.text(0.75, 0.42, "A temporal gap prevents input-window\noverlap across subsets.", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_dir / "protocol_leakage_diagram.png", dpi=200)
    plt.close()


def generate_literature_table(table_dir: Path) -> None:
    rows = [
        {
            "reference": "Sanchez and Sotelo, EMG Dataset in Lower Limb",
            "doi": "10.24432/C5ZW3P",
            "task_or_data": "Public lower-limb sEMG and knee-goniometry dataset",
            "subjects_or_data": "22 male subjects; 11 healthy and 11 pathological; four sEMG channels RF/BF/VM/ST plus knee angle FX",
            "reported_result": "Dataset source; no age/body mass/height/pathology subtype metadata provided",
            "relevance_to_revision": "Dataset source and protocol/demographic context for the revised manuscript.",
        },
        {
            "reference": "Li et al., Sensors 2023, Estimation of Knee Joint Angle from Surface EMG Using Multiple Kernels Relevance Vector Regression",
            "doi": "10.3390/s23104934",
            "task_or_data": "sEMG-based knee-angle estimation with MKRVR and classical baselines",
            "subjects_or_data": "Five healthy volunteers; separate local dataset",
            "reported_result": "MAE 3.27 +/- 1.20 deg; RMSE 4.81 +/- 1.37 deg",
            "relevance_to_revision": "Quantitative comparator showing classical-kernel performance under a different dataset/protocol.",
        },
        {
            "reference": "Wang et al., ROBIO 2024, Accurate Prediction of Knee Joint Angles Using a Hybrid CNN-LSTM-Attention Network from Surface Electromyography",
            "doi": "10.1109/ROBIO64047.2024.10907528",
            "task_or_data": "Continuous knee-angle prediction from raw sEMG using CNN-LSTM-Attention",
            "subjects_or_data": "Five able-bodied subjects; six sEMG channels; separate local dataset",
            "reported_result": "RMSE 3.09 +/- 0.90 deg; rho 0.9911 +/- 0.0052; R2 0.9817 +/- 0.0108",
            "relevance_to_revision": "Recent deep temporal-model comparator; useful for discussion but not directly comparable to UCI protocol.",
        },
        {
            "reference": "Wang et al., IEEE JBHI 2025, Dual Transformer Network for Predicting Joint Angles and Torques from Multi-Channel EMG Signals in the Lower Limbs",
            "doi": "10.1109/JBHI.2025.3555255",
            "task_or_data": "Multi-output lower-limb joint angle and torque prediction from multi-channel EMG",
            "subjects_or_data": "Separate multi-channel lower-limb dataset",
            "reported_result": "Knee-angle RMSE 1.4312 deg; hip 1.1827 deg; ankle 0.8113 deg",
            "relevance_to_revision": "Recent Transformer-based lower-limb comparator under a different richer protocol.",
        },
        {
            "reference": "This revised study",
            "doi": "N/A",
            "task_or_data": "Leakage-aware subject-specific knee-angle prediction from UCI Lower Limb EMG",
            "subjects_or_data": "22 male subjects; 11 healthy and 11 pathological; four sEMG channels RF/BF/VM/ST",
            "reported_result": "MAE 6.30 deg; RMSE 9.03 deg under purged temporal validation",
            "relevance_to_revision": "Main within-dataset result; emphasizes leakage-safe protocol rather than optimistic overlapping-window accuracy.",
        },
    ]
    write_dict_csv(table_dir / "literature_comparison.csv", rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    strict_artifact_root = Path(args.strict_artifact_root).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    table_dir = ensure_dir(output_dir / "revision_tables")
    fig_dir = ensure_dir(output_dir / "revision_figures")
    data_root = find_dataset_root(Path(args.data_root))
    records = discover_subjects(data_root)

    copy_core_artifacts(artifact_root, output_dir)
    generate_dataset_summary(data_root, artifact_root, table_dir)
    generate_model_tables(artifact_root, strict_artifact_root, table_dir)
    generate_statistics(artifact_root, table_dir, output_dir, rng, args.bootstrap_iters)
    generate_feature_correlation(artifact_root, records, table_dir, fig_dir, rng)
    generate_loso_ridge(artifact_root, records, table_dir, args.loso_step)
    generate_traditional_baselines(
        artifact_root,
        records,
        table_dir,
        rng,
        step=args.traditional_step,
        max_train=args.traditional_max_train,
    )
    generate_robustness_summary(table_dir)
    generate_prediction_figures(artifact_root, fig_dir)
    generate_schematics(fig_dir)
    generate_literature_table(table_dir)
    save_json(
        output_dir / "reviewer_artifact_manifest.json",
        {
            "artifact_root": artifact_root,
            "data_root": data_root,
            "main_model": MAIN_MODEL,
            "paper_threshold": {"mae": PAPER_MAE, "rmse": PAPER_RMSE},
            "generated_tables": sorted(path.name for path in table_dir.glob("*")),
            "generated_figures": sorted(path.name for path in fig_dir.glob("*")),
            "notes": [
                "RF/SVR/XGBoost baselines are generated when scikit-learn and xgboost are installed.",
                "LOSO ridge is a lightweight cross-subject baseline and does not replace a full cross-subject Transformer experiment.",
            ],
        },
    )
    print(f"Reviewer artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
