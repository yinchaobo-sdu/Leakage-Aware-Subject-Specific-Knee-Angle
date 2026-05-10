from __future__ import annotations

import csv
import json
import os
import random
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def natural_key(text: str) -> List[Any]:
    parts = re.split(r"(\d+)", text)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def parse_int_list(value: str) -> List[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    if any(v <= 0 for v in values):
        raise ValueError("All integer list values must be positive.")
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def stable_seed(base_seed: int, *parts: str) -> int:
    total = base_seed
    for part in parts:
        for ch in part:
            total = (total * 131 + ord(ch)) % 2_147_483_647
    return total


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return dict(vars(obj))


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)


def write_dict_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean_or_nan(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def relpath_for_log(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


try:
    from tqdm.auto import tqdm as tqdm
except Exception:  # pragma: no cover - optional dependency fallback.
    def tqdm(iterable=None, **_: Any):
        return iterable if iterable is not None else []


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
