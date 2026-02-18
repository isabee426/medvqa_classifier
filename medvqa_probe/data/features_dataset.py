"""On‑disk feature dataset: saving, loading, and a PyTorch Dataset wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from medvqa_probe.utils.logging import setup_logging

logger = setup_logging(name=__name__)


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------

@dataclass
class FeatureRecord:
    id: str
    dataset_name: str
    split: str
    features: np.ndarray  # shape: (feature_dim,)
    label: int
    label_type: Literal["corruption", "hallucination"]
    corruption_type: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------

class FeatureStore:
    """Append‑friendly storage backed by .npz (features) + .jsonl (index)."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._index_path = self.directory / "index.jsonl"
        self._features_path = self.directory / "features.npz"
        self._buffer_features: list[np.ndarray] = []
        self._buffer_index: list[dict] = []
        self._existing_ids: set[str] = set()
        # Load existing IDs for resumability.
        if self._index_path.exists():
            with open(self._index_path) as f:
                for line in f:
                    rec = json.loads(line)
                    self._existing_ids.add(rec["id"])

    @property
    def n_existing(self) -> int:
        return len(self._existing_ids)

    def already_processed(self, example_id: str) -> bool:
        return example_id in self._existing_ids

    def append(self, record: FeatureRecord) -> None:
        """Buffer a single record."""
        self._buffer_index.append({
            "id": record.id,
            "dataset_name": record.dataset_name,
            "split": record.split,
            "label": record.label,
            "label_type": record.label_type,
            "corruption_type": record.corruption_type,
        })
        self._buffer_features.append(record.features)
        self._existing_ids.add(record.id)

    def flush(self) -> None:
        """Write buffered records to disk (append‑safe)."""
        if not self._buffer_index:
            return
        # Append index lines.
        with open(self._index_path, "a") as f:
            for rec in self._buffer_index:
                f.write(json.dumps(rec) + "\n")
        # Append features — we accumulate in a single .npz keyed by ID.
        existing: dict[str, np.ndarray] = {}
        if self._features_path.exists():
            existing = dict(np.load(self._features_path, allow_pickle=False))
        for idx_rec, feat in zip(self._buffer_index, self._buffer_features):
            existing[idx_rec["id"]] = feat
        np.savez(self._features_path, **existing)
        logger.info("Flushed %d records to %s", len(self._buffer_index), self.directory)
        self._buffer_index.clear()
        self._buffer_features.clear()

    def finalize(self) -> None:
        """Flush any remaining buffer."""
        self.flush()


def load_feature_records(
    directory: str | Path,
    splits: list[str] | None = None,
    label_type: str | None = None,
) -> list[FeatureRecord]:
    """Load all feature records from a :class:`FeatureStore` directory."""
    directory = Path(directory)
    index_path = directory / "index.jsonl"
    features_path = directory / "features.npz"
    if not index_path.exists() or not features_path.exists():
        raise FileNotFoundError(f"Feature store not found at {directory}")

    data = np.load(features_path, allow_pickle=False)
    records: list[FeatureRecord] = []
    with open(index_path) as f:
        for line in f:
            meta = json.loads(line)
            if splits and meta["split"] not in splits:
                continue
            if label_type and meta["label_type"] != label_type:
                continue
            records.append(FeatureRecord(
                id=meta["id"],
                dataset_name=meta["dataset_name"],
                split=meta["split"],
                features=data[meta["id"]],
                label=meta["label"],
                label_type=meta["label_type"],
                corruption_type=meta.get("corruption_type"),
            ))
    logger.info(
        "Loaded %d feature records from %s (splits=%s, label_type=%s)",
        len(records), directory, splits, label_type,
    )
    return records


# ---------------------------------------------------------------------------
# PyTorch dataset wrapper
# ---------------------------------------------------------------------------

class FeatureDataset(Dataset):
    """Thin PyTorch Dataset over a list of :class:`FeatureRecord`."""

    def __init__(
        self,
        records: list[FeatureRecord],
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        noise_std: float = 0.0,
    ) -> None:
        self.ids = [r.id for r in records]
        self.features = torch.from_numpy(
            np.stack([r.features for r in records])
        ).float()
        self.labels = torch.tensor(
            [r.label for r in records], dtype=torch.float32,
        )
        self.corruption_types = [r.corruption_type for r in records]
        self.noise_std = noise_std  # Gaussian noise applied at __getitem__ time

        # Standardize features (zero mean, unit variance).
        if mean is None:
            self.mean = self.features.mean(dim=0)
            self.std = self.features.std(dim=0).clamp(min=1e-8)
        else:
            self.mean = torch.from_numpy(mean).float()
            self.std = torch.from_numpy(std).float().clamp(min=1e-8)
        self.features = (self.features - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        feat = self.features[idx]
        if self.noise_std > 0.0:
            feat = feat + torch.randn_like(feat) * self.noise_std
        return {
            "id": self.ids[idx],
            "features": feat,
            "label": self.labels[idx],
        }

    @property
    def feature_dim(self) -> int:
        return self.features.shape[1]

    def label_distribution(self) -> dict[int, int]:
        labels = self.labels.numpy().astype(int)
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
