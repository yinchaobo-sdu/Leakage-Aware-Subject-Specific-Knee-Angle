from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict


MAIN_MODEL = "transformer_4ch_extended"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify reproduced main experiment metrics and leakage audit.")
    parser.add_argument("--artifact-dir", required=True, help="Directory containing metrics_summary.csv and leakage_audit.json.")
    parser.add_argument("--mae", type=float, default=6.296599355610934, help="Reference overall MAE for the main model.")
    parser.add_argument("--rmse", type=float, default=9.033915257953181, help="Reference overall RMSE for the main model.")
    parser.add_argument("--tolerance", type=float, default=0.25, help="Allowed absolute metric difference in degrees.")
    parser.add_argument("--model", default=MAIN_MODEL, help="Expected main model name.")
    return parser.parse_args()


def load_main_row(path: Path, model: str) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model") == model and row.get("group") == "overall" and row.get("main_result") == "True":
                return row
    raise ValueError(f"Could not find the overall main-result row for {model} in {path}.")


def load_audit_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing leakage audit: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Missing summary object in {path}.")
    return summary


def assert_close(name: str, observed: float, expected: float, tolerance: float) -> None:
    delta = abs(observed - expected)
    if math.isnan(observed) or delta > tolerance:
        raise AssertionError(
            f"{name} differs from reference: observed={observed:.6f}, "
            f"expected={expected:.6f}, delta={delta:.6f}, tolerance={tolerance:.6f}"
        )


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()

    row = load_main_row(artifact_dir / "metrics_summary.csv", args.model)
    observed_mae = float(row["mae"])
    observed_rmse = float(row["rmse"])
    assert_close("MAE", observed_mae, args.mae, args.tolerance)
    assert_close("RMSE", observed_rmse, args.rmse, args.tolerance)

    summary = load_audit_summary(artifact_dir / "leakage_audit.json")
    if not summary.get("all_subject_specs_publication_safe", False):
        raise AssertionError("Leakage audit did not mark all subject/model checks as publication safe.")
    safe = int(summary.get("leakage_safe_subject_specs", -1))
    audited = int(summary.get("audited_subject_specs", -1))
    if audited <= 0 or safe != audited:
        raise AssertionError(f"Leakage audit count mismatch: safe={safe}, audited={audited}.")

    print("Reproduction check passed.")
    print(f"  model: {args.model}")
    print(f"  MAE:  observed={observed_mae:.4f}, reference={args.mae:.4f}")
    print(f"  RMSE: observed={observed_rmse:.4f}, reference={args.rmse:.4f}")
    print(f"  leakage audit: {safe}/{audited} safe")


if __name__ == "__main__":
    main()
