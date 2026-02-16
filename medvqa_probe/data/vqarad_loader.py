"""VQA-RAD dataset loader — supports HuggingFace Hub and local files."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from medvqa_probe.utils.config import DataConfig
from medvqa_probe.utils.logging import setup_logging

logger = setup_logging(name=__name__)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class Example:
    id: str
    image_path: str  # path on disk *or* placeholder when image kept in memory
    question: str
    answer: str
    split: Literal["train", "val", "test"]
    dataset_name: str = "vqa-rad"
    meta: dict[str, Any] = field(default_factory=dict)
    # When loaded from HF, the PIL image lives here (avoids extra disk I/O).
    _pil_image: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_pil_image", None)
        return d


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _stable_id(dataset_name: str, split: str, index: int) -> str:
    """Deterministic short hash so IDs are reproducible."""
    raw = f"{dataset_name}:{split}:{index}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def load_vqarad(cfg: DataConfig, image_cache_dir: str | None = None) -> list[Example]:
    """Load VQA‑RAD examples.

    If *cfg.use_local* is True and *cfg.local_data_dir* is set, load from
    local files.  Otherwise pull from the HuggingFace Hub.

    When *image_cache_dir* is given, PIL images are saved to that directory
    and ``image_path`` is set accordingly; otherwise images are kept in memory.
    """
    examples: list[Example] = []

    if cfg.use_local and cfg.local_data_dir:
        examples = _load_local(cfg)
    else:
        examples = _load_hf(cfg, image_cache_dir=image_cache_dir)

    logger.info(
        "Loaded %d examples from %s (splits: %s)",
        len(examples), cfg.dataset_name, cfg.splits,
    )
    return examples


def _load_hf(cfg: DataConfig, image_cache_dir: str | None = None) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset(cfg.hf_dataset_id)
    examples: list[Example] = []

    # VQA-RAD has "train" and "test" splits on HF.
    split_map = {"train": "train", "test": "test", "val": "test"}

    cache_dir: Path | None = Path(image_cache_dir) if image_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for split_name in cfg.splits:
        hf_split = split_map.get(split_name, split_name)
        if hf_split not in ds:
            logger.warning("Split '%s' not found in HF dataset; skipping.", hf_split)
            continue
        subset = ds[hf_split]
        max_n = cfg.max_examples_per_split
        n = len(subset) if max_n is None else min(max_n, len(subset))

        for i in range(n):
            row = subset[i]
            eid = _stable_id(cfg.dataset_name, split_name, i)
            pil_img = row["image"]

            img_path = ""
            if cache_dir is not None:
                img_path = str(cache_dir / f"{eid}.png")
                if not Path(img_path).exists():
                    pil_img.save(img_path)

            examples.append(Example(
                id=eid,
                image_path=img_path,
                question=row["question"],
                answer=row["answer"],
                split=split_name,  # type: ignore[arg-type]
                dataset_name=cfg.dataset_name,
                _pil_image=pil_img if cache_dir is None else None,
            ))

    return examples


def _load_local(cfg: DataConfig) -> list[Example]:
    """Load from a local directory with structure:
        <local_data_dir>/<split>/images/  and  <local_data_dir>/<split>/qa.jsonl
    Each line in qa.jsonl: {"image": "xxx.png", "question": "...", "answer": "..."}
    """
    import json

    base = Path(cfg.local_data_dir)  # type: ignore[arg-type]
    examples: list[Example] = []
    for split_name in cfg.splits:
        qa_file = base / split_name / "qa.jsonl"
        if not qa_file.exists():
            logger.warning("Local QA file not found: %s", qa_file)
            continue
        with open(qa_file) as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                eid = _stable_id(cfg.dataset_name, split_name, i)
                examples.append(Example(
                    id=eid,
                    image_path=str(base / split_name / "images" / row["image"]),
                    question=row["question"],
                    answer=row["answer"],
                    split=split_name,  # type: ignore[arg-type]
                    dataset_name=cfg.dataset_name,
                ))
                if cfg.max_examples_per_split and i + 1 >= cfg.max_examples_per_split:
                    break
    return examples


def iter_examples_by_split(
    examples: list[Example],
    split: str,
) -> Iterator[Example]:
    """Yield examples belonging to a given split."""
    return (e for e in examples if e.split == split)
