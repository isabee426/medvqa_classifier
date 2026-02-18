"""Hallucination feature extraction pipeline (Stage-2).

Reuses Stage-1 BLIP backbone and pooling, but loads hallucination-labeled
data from HALT-MedVQA / CXR-VisHal instead of corruption data.

Usage:
    python -m fine_tuned.extract_hallucination_features \
        --config fine_tuned/configs/extract_halt_medvqa.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from medvqa_probe.data.features_dataset import FeatureRecord, FeatureStore
from medvqa_probe.data.vqarad_loader import Example
from medvqa_probe.models.base_vqa_model import HFVLMModel, pool_activations
from medvqa_probe.utils.config import FullConfig, load_config
from medvqa_probe.utils.logging import setup_logging

logger = setup_logging(name=__name__)


def _get_image(example: Example) -> Image.Image:
    if example._pil_image is not None:
        return example._pil_image.convert("RGB")
    if example.image_path and Path(example.image_path).exists():
        return Image.open(example.image_path).convert("RGB")
    raise ValueError(f"No image available for example {example.id}")


def _load_examples(cfg: FullConfig) -> list[tuple[Example, int]]:
    """Dispatch to the correct hallucination dataset loader."""
    dataset = cfg.data.dataset_name
    image_cache = str(Path(cfg.extraction.output_dir) / "image_cache")

    if dataset in ("halt-medvqa", "halt_medvqa"):
        from fine_tuned.data.halt_medvqa_loader import load_halt_medvqa
        vqa_rad_images_dir = getattr(cfg.data, "local_data_dir", None)
        return load_halt_medvqa(cfg.data, image_cache_dir=image_cache,
                                vqa_rad_images_dir=vqa_rad_images_dir)
    elif dataset in ("cxr-vishal", "cxr_vishal"):
        from fine_tuned.data.cxr_vishal_loader import load_cxr_vishal
        return load_cxr_vishal(cfg.data, image_cache_dir=image_cache)
    else:
        logger.error("Unknown dataset: %s (expected halt-medvqa or cxr-vishal)", dataset)
        sys.exit(1)


def run(cfg: FullConfig) -> None:
    ext = cfg.extraction
    store = FeatureStore(ext.output_dir)
    logger.info("Output directory: %s (already have %d records)",
                ext.output_dir, store.n_existing)

    # Load hallucination-labeled examples.
    pairs = _load_examples(cfg)
    if not pairs:
        logger.error("No examples loaded — check dataset config.")
        sys.exit(1)

    label_counts = Counter(l for _, l in pairs)
    logger.info("Total examples: %d (faithful=%d, hallucinated=%d)",
                len(pairs), label_counts.get(0, 0), label_counts.get(1, 0))

    # Load BLIP backbone.
    model = HFVLMModel()
    model.load(ext)

    # Extract features.
    skipped = 0
    for i, (example, label) in enumerate(tqdm(pairs, desc="Extracting hallucination features")):
        if store.already_processed(example.id):
            skipped += 1
            continue

        try:
            image = _get_image(example)
        except Exception as e:
            logger.warning("Skipping %s: %s", example.id, e)
            continue

        out = model.forward(image, example.question, example.answer)

        features = pool_activations(
            activations=out.activations,
            vision_mask=out.vision_mask,
            question_mask=out.question_mask,
            answer_mask=out.answer_mask,
            layers=ext.layers,
            segments=ext.segments,
            pooling=ext.pooling,
        )

        halluc_type = example.meta.get("hallucination_type")
        store.append(FeatureRecord(
            id=example.id,
            dataset_name=example.dataset_name,
            split=example.split,
            features=features,
            label=label,
            label_type="hallucination",
            corruption_type=halluc_type,
        ))

        if (i + 1) % ext.save_every_n == 0:
            store.flush()

    store.finalize()

    if skipped:
        logger.info("Skipped %d already-processed examples.", skipped)

    # Sanity checks.
    from medvqa_probe.data.features_dataset import load_feature_records
    records = load_feature_records(ext.output_dir, label_type="hallucination")
    shapes = set(r.features.shape for r in records)
    label_dist = Counter(r.label for r in records)
    logger.info("Feature shapes: %s", shapes)
    logger.info("Label distribution: %s", dict(label_dist))
    if len(shapes) != 1:
        logger.error("INCONSISTENT feature shapes: %s", shapes)
    logger.info("Hallucination feature extraction complete. %d records.", len(records))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract hallucination features for Stage-2 fine-tuning.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
