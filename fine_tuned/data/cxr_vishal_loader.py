"""CXR-VisHal loader — chest X-ray hallucination benchmark from MedVH/MedHEval.

Since CXR-VisHal is gated on PhysioNet, this loader supports:
  1. Local files: if ``cfg.local_data_dir`` points to a directory with
     ``qa.jsonl`` and ``images/``, those are used directly.
  2. Synthetic fallback: constructs hallucination-labeled data from
     VQA-RAD (HuggingFace) using HALT-style corruptions, filtered to
     chest-related questions where possible.
"""

from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from pathlib import Path

from medvqa_probe.data.vqarad_loader import Example, load_vqarad
from medvqa_probe.utils.config import DataConfig
from medvqa_probe.utils.logging import setup_logging

logger = setup_logging(name=__name__)

# Chest-related keywords for filtering VQA-RAD to CXR-like examples.
CXR_KEYWORDS = {"chest", "lung", "pulmonary", "cardiac", "heart", "rib",
                 "mediastin", "pleural", "diaphragm", "thorax", "thoracic",
                 "pneumo", "atelectasis", "consolidation", "cardiomegaly"}


def _stable_id(prefix: str, index: int, suffix: str = "") -> str:
    raw = f"{prefix}:{index}:{suffix}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _is_cxr_related(ex: Example) -> bool:
    text = (ex.question + " " + ex.answer).lower()
    return any(kw in text for kw in CXR_KEYWORDS)


def load_cxr_vishal(
    cfg: DataConfig,
    image_cache_dir: str | None = None,
) -> list[tuple[Example, int]]:
    """Load CXR-VisHal with hallucination labels.

    Returns (Example, label) pairs: label=0 faithful, label=1 hallucinated.
    """
    if cfg.use_local and cfg.local_data_dir:
        return _load_local(cfg)
    return _load_synthetic(cfg, image_cache_dir)


def _load_local(cfg: DataConfig) -> list[tuple[Example, int]]:
    """Load from local CXR-VisHal files (PhysioNet download)."""
    import json

    base = Path(cfg.local_data_dir)  # type: ignore[arg-type]
    pairs: list[tuple[Example, int]] = []

    for split_name in cfg.splits:
        qa_file = base / split_name / "qa.jsonl"
        if not qa_file.exists():
            logger.warning("CXR-VisHal QA file not found: %s", qa_file)
            continue
        with open(qa_file) as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                eid = _stable_id("cxr-vishal", i, split_name)
                label = int(row.get("is_hallucinated", row.get("label", 0)))
                pairs.append((
                    Example(
                        id=eid,
                        image_path=str(base / split_name / "images" / row["image"]),
                        question=row["question"],
                        answer=row["answer"],
                        split=split_name,
                        dataset_name="cxr-vishal",
                        meta={"hallucination_type": "dataset_label"},
                    ),
                    label,
                ))
                if cfg.max_examples_per_split and i + 1 >= cfg.max_examples_per_split:
                    break
    logger.info("Loaded %d CXR-VisHal pairs from local", len(pairs))
    return pairs


def _load_synthetic(
    cfg: DataConfig,
    image_cache_dir: str | None = None,
) -> list[tuple[Example, int]]:
    """Synthetic CXR-VisHal from VQA-RAD chest-related questions."""
    logger.info("CXR-VisHal not available locally — using synthetic from VQA-RAD chest subset")
    img_dir = image_cache_dir or "outputs/cxr_cache/images"
    examples = load_vqarad(cfg, image_cache_dir=img_dir)

    # Filter to chest-related questions.
    cxr_examples = [ex for ex in examples if _is_cxr_related(ex)]
    if len(cxr_examples) < 20:
        logger.warning("Only %d CXR-related examples found; using all VQA-RAD", len(cxr_examples))
        cxr_examples = examples

    rng = random.Random(42)
    pairs: list[tuple[Example, int]] = []

    for i, ex in enumerate(cxr_examples):
        faithful = deepcopy(ex)
        faithful.id = _stable_id("cxr-vishal-synth", i, "faithful")
        faithful.dataset_name = "cxr-vishal"
        faithful.meta["hallucination_type"] = "none"
        pairs.append((faithful, 0))

        # Hallucinated: swap answer.
        donor = rng.choice(cxr_examples)
        while donor.answer == ex.answer and len(cxr_examples) > 1:
            donor = rng.choice(cxr_examples)
        halluc = deepcopy(ex)
        halluc.id = _stable_id("cxr-vishal-synth", i, "halluc")
        halluc.answer = donor.answer
        halluc.dataset_name = "cxr-vishal"
        halluc.meta["hallucination_type"] = "swap_answer"
        pairs.append((halluc, 1))

    if cfg.max_examples_per_split:
        train = [(e, l) for e, l in pairs if e.split == "train"]
        test = [(e, l) for e, l in pairs if e.split == "test"]
        rng.shuffle(train)
        rng.shuffle(test)
        pairs = train[:cfg.max_examples_per_split] + test[:cfg.max_examples_per_split]

    label_counts = {0: sum(1 for _, l in pairs if l == 0), 1: sum(1 for _, l in pairs if l == 1)}
    logger.info("Synthetic CXR-VisHal: %d pairs, labels=%s", len(pairs), label_counts)
    return pairs
