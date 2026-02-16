"""Razvan‑style corruption strategies for creating negative (misaligned) examples."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Literal

from medvqa_probe.data.vqarad_loader import Example
from medvqa_probe.utils.config import CorruptionConfig
from medvqa_probe.utils.logging import setup_logging

logger = setup_logging(name=__name__)

CorruptionType = Literal["swap_answer", "swap_image", "empty_answer", "distort_answer"]


# ---------------------------------------------------------------------------
# Individual corruption functions
# ---------------------------------------------------------------------------

def swap_answer(example: Example, pool: list[Example], rng: random.Random) -> Example:
    """Replace the answer with an answer from a *different* example."""
    candidates = [e for e in pool if e.id != example.id and e.answer != example.answer]
    if not candidates:
        candidates = [e for e in pool if e.id != example.id]
    donor = rng.choice(candidates)
    out = deepcopy(example)
    out.answer = donor.answer
    out.meta["corruption_type"] = "swap_answer"
    out.meta["donor_id"] = donor.id
    return out


def swap_image(example: Example, pool: list[Example], rng: random.Random) -> Example:
    """Pair the question+answer with a different image."""
    candidates = [e for e in pool if e.image_path != example.image_path]
    if not candidates:
        candidates = [e for e in pool if e.id != example.id]
    donor = rng.choice(candidates)
    out = deepcopy(example)
    out.image_path = donor.image_path
    out._pil_image = donor._pil_image
    out.meta["corruption_type"] = "swap_image"
    out.meta["donor_id"] = donor.id
    return out


_ANATOMY_SWAPS = {
    "lung": "liver", "liver": "lung", "heart": "kidney", "kidney": "heart",
    "left": "right", "right": "left", "upper": "lower", "lower": "upper",
    "anterior": "posterior", "posterior": "anterior", "chest": "abdomen",
    "abdomen": "chest", "brain": "spine", "spine": "brain",
    "normal": "abnormal", "abnormal": "normal", "yes": "no", "no": "yes",
    "benign": "malignant", "malignant": "benign", "fracture": "effusion",
    "effusion": "fracture", "pneumonia": "cardiomegaly", "cardiomegaly": "pneumonia",
}


def distort_answer(example: Example, pool: list[Example], rng: random.Random) -> Example:
    """Perturb answer by swapping anatomy/finding terms."""
    out = deepcopy(example)
    words = out.answer.split()
    swapped = False
    for i, w in enumerate(words):
        key = w.lower().strip(".,;:!?")
        if key in _ANATOMY_SWAPS:
            replacement = _ANATOMY_SWAPS[key]
            # Preserve original casing
            if w[0].isupper():
                replacement = replacement.capitalize()
            words[i] = w.replace(key, replacement).replace(key.capitalize(), replacement.capitalize())
            swapped = True
    if swapped:
        out.answer = " ".join(words)
    else:
        # Fallback: negate or prepend "not" to make it wrong
        out.answer = "not " + out.answer if not out.answer.lower().startswith("not") else out.answer[4:]
    out.meta["corruption_type"] = "distort_answer"
    return out


def empty_answer(example: Example, pool: list[Example], rng: random.Random) -> Example:
    """Remove or blank out the answer text."""
    out = deepcopy(example)
    out.answer = rng.choice(["", "N/A", "[BLANK]"])
    out.meta["corruption_type"] = "empty_answer"
    return out


_STRATEGY_FNS = {
    "swap_answer": swap_answer,
    "swap_image": swap_image,
    "empty_answer": empty_answer,
    "distort_answer": distort_answer,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_corrupted_examples(
    clean_examples: list[Example],
    cfg: CorruptionConfig,
) -> list[Example]:
    """Generate corrupted (negative) examples from a list of clean examples.

    Returns a list of corrupted :class:`Example` objects with
    ``meta["corruption_type"]`` set.  The number of negatives per clean
    example is controlled by *cfg.negatives_per_positive*.
    """
    if not cfg.enabled:
        return []

    rng = random.Random(cfg.seed)

    # Normalise strategy probabilities.
    active: list[tuple[str, float]] = [
        (name, weight)
        for name, weight in cfg.strategies.items()
        if name in _STRATEGY_FNS and weight > 0
    ]
    if not active:
        logger.warning("No active corruption strategies configured.")
        return []

    names, weights = zip(*active)
    total = sum(weights)
    probs = [w / total for w in weights]

    corrupted: list[Example] = []
    for example in clean_examples:
        for _ in range(cfg.negatives_per_positive):
            strategy_name: str = rng.choices(names, weights=probs, k=1)[0]
            fn = _STRATEGY_FNS[strategy_name]
            corrupted.append(fn(example, clean_examples, rng))

    logger.info(
        "Generated %d corrupted examples (%d per positive, strategies: %s)",
        len(corrupted), cfg.negatives_per_positive, list(names),
    )
    return corrupted
