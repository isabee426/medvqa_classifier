"""HALT-MedVQA loader — downloads from GitHub, filters to VQA-RAD (synpic) images.

HALT-MedVQA hallucination scenarios:
  - FAKE: nonsensical questions; correct answer = "I do not know"
  - SWAP: image swapped with unrelated one; correct answer = "I do not know"

Labeling scheme (bidirectional hallucination detection):
  label=0 (correct):
    - HALT impossible question + "I do not know"   (correctly abstains)
    - Non-HALT VQA-RAD question + correct answer    (correctly answers)
  label=1 (hallucinated):
    - HALT impossible question + specific answer     (hallucinates instead of IDK)
    - Non-HALT VQA-RAD question + "I do not know"   (refuses when it should know)

This gives a perfectly balanced 50/50 dataset (1:1 per source) and forces the
classifier to detect BOTH hallucination directions:
  1. Model makes up answers when it should say "I do not know"
  2. Model says "I do not know" when it should provide a real answer
Factual accuracy (wrong answers) is handled separately at RL training time.

If VQA-RAD images are not locally available, falls back to synthetic
hallucination labels using VQA-RAD from HuggingFace.
"""

from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from dataclasses import field
from pathlib import Path
from typing import Any

from medvqa_probe.data.vqarad_loader import Example
from medvqa_probe.utils.config import DataConfig
from medvqa_probe.utils.logging import setup_logging

logger = setup_logging(name=__name__)

# GitHub raw URLs for HALT-MedVQA data files.
HALT_GITHUB_URLS = {
    "fake": "https://raw.githubusercontent.com/knowlab/halt-medvqa/main/data/fake_qa_shuffle.json",
    "swap": "https://raw.githubusercontent.com/knowlab/halt-medvqa/main/data/swap_img_shuffle.json",
}

# Original VQA-RAD dataset JSON (correct Q/A pairs with image filenames).
VQARAD_JSON_URL = "https://files.osf.io/v1/resources/89kps/providers/osfstorage/5b213a9886d8510012c26c09"


def _stable_id(prefix: str, scenario: str, index: int, suffix: str = "") -> str:
    raw = f"{prefix}:{scenario}:{index}:{suffix}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _download_halt_json(url: str, cache_dir: Path) -> list[dict]:
    """Download a HALT-MedVQA JSON file, caching locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = url.split("/")[-1]
    cached = cache_dir / fname

    if cached.exists():
        logger.info("Using cached HALT data: %s", cached)
        with open(cached) as f:
            data = json.load(f)
    else:
        import urllib.request
        logger.info("Downloading HALT data: %s", url)
        with urllib.request.urlopen(url) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        with open(cached, "w") as f:
            f.write(raw)

    # HALT uses dict with string keys "0", "1", ... → convert to list.
    if isinstance(data, dict):
        data = [data[str(i)] for i in range(len(data))]
    return data


def _filter_vqarad(records: list[dict]) -> list[dict]:
    """Keep only VQA-RAD derived records (synpic images)."""
    filtered = [r for r in records if r.get("img", "").startswith("synpic")]
    logger.info("Filtered to %d VQA-RAD (synpic) records out of %d total", len(filtered), len(records))
    return filtered


def _load_vqarad_pairs(
    cache_dir: Path,
    images_dir: Path,
    rng: random.Random,
    max_questions: int | None = None,
) -> list[tuple[Example, int]]:
    """Load original VQA-RAD Q/A pairs with bidirectional labels.

    For each VQA-RAD question (capped to max_questions):
      - (question, correct_answer)  → label=0  (correctly answers)
      - (question, "I do not know") → label=1  (wrongly abstains)

    Downloads VQA_RAD Dataset Public.json from OSF.
    """
    cached = cache_dir / "VQA_RAD_Dataset_Public.json"
    if cached.exists():
        logger.info("Using cached VQA-RAD JSON: %s", cached)
        with open(cached) as f:
            data = json.load(f)
    else:
        import urllib.request
        logger.info("Downloading VQA-RAD dataset JSON from OSF...")
        with urllib.request.urlopen(VQARAD_JSON_URL) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        with open(cached, "w") as f:
            f.write(raw)

    # Shuffle and cap upfront — no point running BLIP on examples we'll discard.
    rng.shuffle(data)
    if max_questions is not None:
        data = data[:max_questions]

    pairs: list[tuple[Example, int]] = []
    for i, rec in enumerate(data):
        img_name = rec.get("image_name", rec.get("image", ""))
        img_path = images_dir / img_name
        if not img_path.exists():
            img_path = images_dir / img_name.replace(".jpg", ".png")
        if not img_path.exists():
            continue

        question = str(rec.get("question", ""))
        answer = str(rec.get("answer", ""))
        if not question or not answer:
            continue

        split = "train" if rng.random() < 0.7 else "test"

        # label=0: correct answer (model should know and does)
        pairs.append((
            Example(
                id=_stable_id("vqarad", "correct", i),
                image_path=str(img_path),
                question=question,
                answer=answer,
                split=split,
                dataset_name="halt-medvqa",
                meta={"halt_scenario": "original", "hallucination_type": "none"},
            ),
            0,
        ))

        # label=1: "I do not know" (model should know but falsely abstains)
        pairs.append((
            Example(
                id=_stable_id("vqarad", "idk", i),
                image_path=str(img_path),
                question=question,
                answer="I do not know",
                split=split,
                dataset_name="halt-medvqa",
                meta={"halt_scenario": "original", "hallucination_type": "false_abstention"},
            ),
            1,
        ))

    logger.info("Loaded %d VQA-RAD pairs: %d correct (0) + %d IDK (1)", len(pairs), len(pairs) // 2, len(pairs) // 2)
    return pairs


def _pick_wrong_option(record: dict) -> str:
    """Pick a random wrong answer option (not the correct one, not 'I do not know')."""
    correct = record["answer_id"]
    options = record.get("option", {})
    wrong = [
        v for k, v in options.items()
        if k != correct and v.lower() != "i do not know" and v.lower() != "none of the above"
    ]
    if not wrong:
        # All options are "I do not know" or correct — use any non-correct option.
        wrong = [v for k, v in options.items() if k != correct]
    return random.choice(wrong) if wrong else record["answer"]


def load_halt_medvqa(
    cfg: DataConfig,
    image_cache_dir: str | None = None,
    vqa_rad_images_dir: str | None = None,
) -> list[tuple[Example, int]]:
    """Load HALT-MedVQA VQA-RAD subset with hallucination labels.

    Returns list of (Example, label) pairs:
      - label=0: faithful (correct answer, typically "I do not know")
      - label=1: hallucinated (wrong option given as answer)

    Args:
        cfg: DataConfig with dataset settings.
        image_cache_dir: directory to cache downloaded images.
        vqa_rad_images_dir: path to VQA-RAD images with original filenames
            (synpic*.jpg). If None, falls back to synthetic approach.
    """
    cache_dir = Path(image_cache_dir or "outputs/halt_cache")

    # Check for local VQA-RAD images.
    images_dir = Path(vqa_rad_images_dir) if vqa_rad_images_dir else None
    if images_dir and not images_dir.exists():
        logger.warning("VQA-RAD images dir not found: %s — using fallback", images_dir)
        images_dir = None

    if images_dir is not None:
        return _load_from_halt_github(cfg, cache_dir, images_dir)
    else:
        return _load_synthetic_fallback(cfg, cache_dir)


def _load_from_halt_github(
    cfg: DataConfig,
    cache_dir: Path,
    images_dir: Path,
) -> list[tuple[Example, int]]:
    """Load actual HALT-MedVQA data with local VQA-RAD images."""
    rng = random.Random(42)
    pairs: list[tuple[Example, int]] = []

    for scenario, url in HALT_GITHUB_URLS.items():
        records = _download_halt_json(url, cache_dir)
        records = _filter_vqarad(records)

        for i, rec in enumerate(records):
            img_path = images_dir / rec["img"]
            if not img_path.exists():
                # Try .png variant.
                img_path = images_dir / rec["img"].replace(".jpg", ".png")
            if not img_path.exists():
                logger.debug("Image not found: %s — skipping", rec["img"])
                continue

            question = rec["question"]
            correct_answer = rec["answer"]  # typically "I do not know"

            # Faithful triple (label=0): correct answer.
            faithful_id = _stable_id("halt-medvqa", scenario, i, "faithful")
            pairs.append((
                Example(
                    id=faithful_id,
                    image_path=str(img_path),
                    question=question,
                    answer=correct_answer,
                    split="train" if rng.random() < 0.7 else "test",
                    dataset_name="halt-medvqa",
                    meta={"halt_scenario": scenario, "hallucination_type": "none"},
                ),
                0,
            ))

            # Hallucinated triple (label=1): wrong option.
            wrong_answer = _pick_wrong_option(rec)
            halluc_id = _stable_id("halt-medvqa", scenario, i, "hallucinated")
            pairs.append((
                Example(
                    id=halluc_id,
                    image_path=str(img_path),
                    question=question,
                    answer=wrong_answer,
                    split="train" if rng.random() < 0.7 else "test",
                    dataset_name="halt-medvqa",
                    meta={"halt_scenario": scenario, "hallucination_type": scenario},
                ),
                1,
            ))

    # Add original VQA-RAD correct Q/A pairs as faithful (label=0).
    # This ensures the classifier sees real medical answers on both sides,
    # not just "I do not know" vs specific medical terms.
    # Cap VQA-RAD questions to match HALT count — no need to process more.
    n_halt = len(pairs) // 2  # pairs so far are 1:1 HALT faithful/hallucinated
    correct_pairs = _load_vqarad_pairs(cache_dir, images_dir, rng, max_questions=n_halt)
    pairs.extend(correct_pairs)

    # Data is balanced by construction: each source generates 1 label=0 + 1 label=1 per question.
    n0 = sum(1 for _, l in pairs if l == 0)
    n1 = sum(1 for _, l in pairs if l == 1)
    logger.info("Dataset: %d faithful (0) + %d hallucinated (1)", n0, n1)

    # Subsample if configured.
    if cfg.max_examples_per_split:
        train = [(ex, l) for ex, l in pairs if ex.split == "train"]
        test = [(ex, l) for ex, l in pairs if ex.split == "test"]
        rng.shuffle(train)
        rng.shuffle(test)
        train = train[:cfg.max_examples_per_split]
        test = test[:cfg.max_examples_per_split]
        pairs = train + test

    logger.info("Loaded %d HALT-MedVQA pairs (faithful + hallucinated)", len(pairs))
    return pairs


def _load_synthetic_fallback(
    cfg: DataConfig,
    cache_dir: Path,
) -> list[tuple[Example, int]]:
    """Fallback: construct HALT-style hallucination data from VQA-RAD on HuggingFace.

    Applies the same hallucination scenarios (SWAP, FAKE) synthetically:
      - SWAP: swap image with a random other → answer becomes wrong for this image
      - FAKE: swap answer with a random other → answer is fabricated
    """
    from medvqa_probe.data.vqarad_loader import load_vqarad

    logger.info("Using synthetic fallback: constructing HALT-style data from VQA-RAD HF")
    image_dir = str(cache_dir / "images")
    examples = load_vqarad(cfg, image_cache_dir=image_dir)
    if not examples:
        logger.error("No VQA-RAD examples loaded for synthetic fallback")
        return []

    rng = random.Random(42)
    pairs: list[tuple[Example, int]] = []

    for i, ex in enumerate(examples):
        # Faithful triple (label=0): original correct answer.
        faithful = deepcopy(ex)
        faithful.id = _stable_id("halt-medvqa-synth", "faithful", i)
        faithful.dataset_name = "halt-medvqa"
        faithful.meta["hallucination_type"] = "none"
        pairs.append((faithful, 0))

        # SWAP hallucination (label=1): keep original question + answer but swap image.
        donor = rng.choice(examples)
        while donor.id == ex.id:
            donor = rng.choice(examples)
        swapped = deepcopy(ex)
        swapped.id = _stable_id("halt-medvqa-synth", "swap", i)
        swapped.image_path = donor.image_path
        swapped._pil_image = donor._pil_image
        swapped.dataset_name = "halt-medvqa"
        swapped.meta["hallucination_type"] = "swap"
        pairs.append((swapped, 1))

        # FAKE hallucination (label=1): keep image but give wrong answer.
        wrong_donor = rng.choice(examples)
        while wrong_donor.answer == ex.answer:
            wrong_donor = rng.choice(examples)
        faked = deepcopy(ex)
        faked.id = _stable_id("halt-medvqa-synth", "fake", i)
        faked.answer = wrong_donor.answer
        faked.dataset_name = "halt-medvqa"
        faked.meta["hallucination_type"] = "fake"
        pairs.append((faked, 1))

    # Subsample if configured.
    if cfg.max_examples_per_split:
        train = [(ex, l) for ex, l in pairs if ex.split == "train"]
        test = [(ex, l) for ex, l in pairs if ex.split == "test"]
        rng.shuffle(train)
        rng.shuffle(test)
        train = train[:cfg.max_examples_per_split]
        test = test[:cfg.max_examples_per_split]
        pairs = train + test

    label_counts = {0: sum(1 for _, l in pairs if l == 0), 1: sum(1 for _, l in pairs if l == 1)}
    logger.info("Synthetic HALT data: %d pairs, labels=%s", len(pairs), label_counts)
    return pairs
