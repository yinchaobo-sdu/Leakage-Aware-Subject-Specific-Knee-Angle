from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import DEFAULT_CHANNELS, FeatureBundle, load_or_compute_features, normalize_channels
from .utils import natural_key


@dataclass(frozen=True)
class SubjectRecord:
    subject: str
    status: str
    path: Path


@dataclass(frozen=True)
class RawEMGData:
    channels: Dict[str, np.ndarray]
    rf: np.ndarray
    bf: np.ndarray
    vm: np.ndarray
    st: np.ndarray
    fx: np.ndarray
    total_rows: int
    valid_rows: int
    skipped_rows: int


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, angles: np.ndarray, indices: np.ndarray, seq_len: int) -> None:
        self.features = torch.as_tensor(np.ascontiguousarray(features), dtype=torch.float32)
        self.angles = torch.as_tensor(np.ascontiguousarray(angles), dtype=torch.float32)
        self.indices = torch.as_tensor(indices.astype(np.int64), dtype=torch.long)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = int(self.indices[item].item())
        end = start + self.seq_len
        return self.features[start:end], self.angles[end]


def find_dataset_root(data_root: Path) -> Path:
    data_root = data_root.expanduser().resolve()
    candidates = [
        data_root,
        data_root / "SEMG_DB1",
        data_root / "SEMG_DB1" / "SEMG_DB1",
    ]
    for candidate in candidates:
        if (candidate / "A_CSV").is_dir() and (candidate / "N_CSV").is_dir():
            return candidate

    matches = []
    for candidate in data_root.rglob("A_CSV"):
        parent = candidate.parent
        if (parent / "N_CSV").is_dir():
            matches.append(parent)
    if matches:
        return sorted(matches, key=lambda p: len(str(p)))[0]

    raise FileNotFoundError(f"Could not find A_CSV/N_CSV under {data_root}.")


def discover_subjects(data_root: Path) -> List[SubjectRecord]:
    root = find_dataset_root(data_root)
    records: List[SubjectRecord] = []
    for folder_name, status in (("N_CSV", "healthy"), ("A_CSV", "unhealthy")):
        folder = root / folder_name
        for path in sorted(folder.glob("*mar.csv"), key=lambda p: natural_key(p.name)):
            records.append(SubjectRecord(subject=path.stem, status=status, path=path))
    records.sort(key=lambda r: (r.status != "healthy", natural_key(r.subject)))
    if not records:
        raise FileNotFoundError(f"No walking CSV files matching *mar.csv found in {root}.")
    return records


def select_subjects(records: Sequence[SubjectRecord], subjects: str) -> List[SubjectRecord]:
    if subjects.strip().lower() == "all":
        return list(records)

    requested = set()
    for part in subjects.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name.endswith(".csv"):
            name = name[:-4]
        requested.add(name)
    selected = [record for record in records if record.subject.lower() in requested]
    missing = requested - {record.subject.lower() for record in selected}
    if missing:
        available = ", ".join(record.subject for record in records)
        raise ValueError(f"Unknown subject(s): {', '.join(sorted(missing))}. Available: {available}")
    return selected


def read_emg_csv(path: Path) -> RawEMGData:
    rows = []
    total = 0
    skipped = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            total += 1
            if len(row) < 5:
                skipped += 1
                continue
            try:
                values = [float(row[i]) for i in range(5)]
            except ValueError:
                skipped += 1
                continue
            rows.append(values)

    if not rows:
        raise ValueError(f"No valid numeric rows found in {path}.")
    if skipped:
        print(f"  warning: skipped {skipped} malformed rows in {path.name}", flush=True)
    data = np.asarray(rows, dtype=np.float32)
    channels = {
        "RF": data[:, 0],
        "BF": data[:, 1],
        "VM": data[:, 2],
        "ST": data[:, 3],
    }
    fx = data[:, 4]
    return RawEMGData(
        channels=channels,
        rf=channels["RF"],
        bf=channels["BF"],
        vm=channels["VM"],
        st=channels["ST"],
        fx=fx,
        total_rows=total,
        valid_rows=len(rows),
        skipped_rows=skipped,
    )


def load_subject_features(
    record: SubjectRecord,
    cache_dir: Optional[Path],
    feature_window: int,
    feature_step: int,
    no_cache: bool,
    channels: Sequence[str] | str = DEFAULT_CHANNELS,
    feature_set: str = "original",
) -> FeatureBundle:
    channels_norm = normalize_channels(channels)
    raw = read_emg_csv(record.path)
    return load_or_compute_features(
        source_path=record.path,
        signals=raw.channels,
        fx=raw.fx,
        cache_dir=cache_dir,
        window=feature_window,
        step=feature_step,
        channels=channels_norm,
        feature_set=feature_set,
        total_rows=raw.total_rows,
        skipped_rows=raw.skipped_rows,
        no_cache=no_cache,
    )


def describe_records(records: Iterable[SubjectRecord]) -> str:
    return ", ".join(f"{r.subject}({r.status})" for r in records)
