"""Feature extraction pipeline.

Usage:
    python -m medvqa_probe.extract_features --config configs/extract_vqarad_stage1.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from medvqa_probe.data.corruptions import generate_corrupted_examples
from medvqa_probe.data.features_dataset import FeatureRecord, FeatureStore
from medvqa_probe.data.vqarad_loader import Example, load_vqarad
from medvqa_probe.models.base_vqa_model import HFVLMModel, pool_activations, pool_attention_maps
from medvqa_probe.utils.config import FullConfig, load_config
from medvqa_probe.utils.logging import setup_logging

logger = setup_logging(name=__name__)


def _get_image(example: Example, image_cache_dir: Path | None = None) -> Image.Image:
    """Load a PIL image from an Example."""
    if example._pil_image is not None:
        img = example._pil_image
    elif example.image_path and Path(example.image_path).exists():
        img = Image.open(example.image_path)
    else:
        raise ValueError(f"No image available for example {example.id}")
    return img.convert("RGB")


def _corrupted_id(base_id: str, corruption_type: str, idx: int) -> str:
    raw = f"{base_id}:corrupt:{corruption_type}:{idx}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def run(cfg: FullConfig) -> None:
    ext = cfg.extraction
    store = FeatureStore(ext.output_dir)
    logger.info("Output directory: %s (already have %d records)", ext.output_dir, store.n_existing)

    # --- Load dataset -----------------------------------------------------
    image_cache = str(Path(ext.output_dir) / "image_cache")
    examples = load_vqarad(cfg.data, image_cache_dir=image_cache)

    # --- Build clean + corrupted work list --------------------------------
    work: list[tuple[Example, int, str | None]] = []  # (example, label, corruption_type)

    for ex in examples:
        work.append((ex, 1, None))  # clean → label 1

    if cfg.corruption.enabled:
        corrupted = generate_corrupted_examples(examples, cfg.corruption)
        # Assign unique IDs to corrupted examples.
        corruption_counters: Counter[str] = Counter()
        for c_ex in corrupted:
            ctype = c_ex.meta.get("corruption_type", "unknown")
            corruption_counters[ctype] += 1
            c_ex.id = _corrupted_id(c_ex.id, ctype, corruption_counters[ctype])
            work.append((c_ex, 0, ctype))

    logger.info("Total work items: %d (clean: %d, corrupted: %d)",
                len(work),
                sum(1 for _, l, _ in work if l == 1),
                sum(1 for _, l, _ in work if l == 0))

    # --- Load model -------------------------------------------------------
    model = HFVLMModel()
    model.load(ext)

    # --- Extract features -------------------------------------------------
    skipped = 0
    for i, (example, label, ctype) in enumerate(tqdm(work, desc="Extracting features")):
        if store.already_processed(example.id):
            skipped += 1
            continue

        try:
            image = _get_image(example)
        except Exception as e:
            logger.warning("Skipping %s: %s", example.id, e)
            continue

        out = model.forward(image, example.question, example.answer)

        hidden_features = pool_activations(
            activations=out.activations,
            vision_mask=out.vision_mask,
            question_mask=out.question_mask,
            answer_mask=out.answer_mask,
            layers=ext.layers,
            segments=ext.segments,
            pooling=ext.pooling,
        )

        if getattr(ext, "use_attention", False) and out.attentions:
            attention_features = pool_attention_maps(
                attentions=out.attentions,
                vision_mask=out.vision_mask,
                question_mask=out.question_mask,
                answer_mask=out.answer_mask,
                layers=ext.layers,
                reduction=getattr(ext, "attention_reduction", "mean"),
            )
            features = np.concatenate([hidden_features, attention_features])
        else:
            features = hidden_features

        store.append(FeatureRecord(
            id=example.id,
            dataset_name=example.dataset_name,
            split=example.split,
            features=features,
            label=label,
            label_type="corruption",
            corruption_type=ctype,
        ))

        if (i + 1) % ext.save_every_n == 0:
            store.flush()

    store.finalize()

    if skipped:
        logger.info("Skipped %d already‑processed examples (resuming).", skipped)

    # --- Sanity checks ----------------------------------------------------
    from medvqa_probe.data.features_dataset import load_feature_records
    records = load_feature_records(ext.output_dir)
    shapes = set(r.features.shape for r in records)
    label_dist = Counter(r.label for r in records)
    logger.info("Feature shapes: %s", shapes)
    logger.info("Label distribution: %s", dict(label_dist))
    if len(shapes) != 1:
        logger.error("INCONSISTENT feature shapes detected: %s", shapes)
    if len(label_dist) < 2:
        logger.warning("Only one label value found — check corruption config.")
    logger.info("Feature extraction complete. %d total records.", len(records))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract internal‑state features from a VQA backbone.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
