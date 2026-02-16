"""Tests for data structures, corruptions, and feature storage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from medvqa_probe.data.corruptions import generate_corrupted_examples
from medvqa_probe.data.features_dataset import FeatureDataset, FeatureRecord, FeatureStore, load_feature_records
from medvqa_probe.data.vqarad_loader import Example
from medvqa_probe.utils.config import CorruptionConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_examples(n: int = 20) -> list[Example]:
    return [
        Example(
            id=f"ex_{i:03d}",
            image_path=f"/tmp/fake_img_{i}.png",
            question=f"Is there a lesion in region {i}?",
            answer="yes" if i % 2 == 0 else "no",
            split="train",
            dataset_name="vqa-rad",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Corruption tests
# ---------------------------------------------------------------------------

class TestCorruptions:
    def test_generates_expected_count(self):
        examples = _make_examples(10)
        cfg = CorruptionConfig(seed=0, negatives_per_positive=2)
        corrupted = generate_corrupted_examples(examples, cfg)
        assert len(corrupted) == 20  # 10 * 2

    def test_all_have_corruption_type(self):
        examples = _make_examples(10)
        cfg = CorruptionConfig(seed=0)
        corrupted = generate_corrupted_examples(examples, cfg)
        for c in corrupted:
            assert "corruption_type" in c.meta
            assert c.meta["corruption_type"] in {"swap_answer", "swap_image", "empty_answer"}

    def test_swap_answer_changes_answer(self):
        examples = _make_examples(20)
        cfg = CorruptionConfig(seed=42, strategies={"swap_answer": 1.0}, negatives_per_positive=1)
        corrupted = generate_corrupted_examples(examples, cfg)
        changed = sum(1 for orig, cor in zip(examples, corrupted) if orig.answer != cor.answer)
        assert changed > 0

    def test_empty_answer_blanks(self):
        examples = _make_examples(10)
        cfg = CorruptionConfig(seed=1, strategies={"empty_answer": 1.0}, negatives_per_positive=1)
        corrupted = generate_corrupted_examples(examples, cfg)
        for c in corrupted:
            assert c.answer in {"", "N/A", "[BLANK]"}

    def test_deterministic_with_seed(self):
        examples = _make_examples(10)
        c1 = generate_corrupted_examples(examples, CorruptionConfig(seed=99))
        c2 = generate_corrupted_examples(examples, CorruptionConfig(seed=99))
        for a, b in zip(c1, c2):
            assert a.answer == b.answer
            assert a.image_path == b.image_path
            assert a.meta == b.meta


# ---------------------------------------------------------------------------
# Feature store tests
# ---------------------------------------------------------------------------

class TestFeatureStore:
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(tmpdir)
            records = [
                FeatureRecord(
                    id=f"r_{i}",
                    dataset_name="vqa-rad",
                    split="train",
                    features=np.random.randn(128).astype(np.float32),
                    label=i % 2,
                    label_type="corruption",
                    corruption_type="swap_answer" if i % 2 == 0 else None,
                )
                for i in range(10)
            ]
            for r in records:
                store.append(r)
            store.finalize()

            loaded = load_feature_records(tmpdir)
            assert len(loaded) == 10
            for orig, back in zip(records, loaded):
                assert orig.id == back.id
                assert orig.label == back.label
                np.testing.assert_allclose(orig.features, back.features, atol=1e-6)

    def test_resumability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(tmpdir)
            store.append(FeatureRecord(
                id="a", dataset_name="d", split="train",
                features=np.zeros(8, dtype=np.float32), label=0, label_type="corruption",
            ))
            store.finalize()

            # Reopen store — "a" should be recognized.
            store2 = FeatureStore(tmpdir)
            assert store2.already_processed("a")
            assert not store2.already_processed("b")


# ---------------------------------------------------------------------------
# FeatureDataset tests
# ---------------------------------------------------------------------------

class TestFeatureDataset:
    def test_shape_and_len(self):
        records = [
            FeatureRecord(
                id=f"r_{i}", dataset_name="d", split="train",
                features=np.random.randn(64).astype(np.float32),
                label=i % 2, label_type="corruption",
            )
            for i in range(5)
        ]
        ds = FeatureDataset(records)
        assert len(ds) == 5
        assert ds.feature_dim == 64
        item = ds[0]
        assert item["features"].shape == (64,)
        assert "label" in item

    def test_label_distribution(self):
        records = [
            FeatureRecord(id=f"r_{i}", dataset_name="d", split="train",
                          features=np.zeros(8, dtype=np.float32),
                          label=1 if i < 3 else 0, label_type="corruption")
            for i in range(5)
        ]
        ds = FeatureDataset(records)
        dist = ds.label_distribution()
        assert dist[1] == 3
        assert dist[0] == 2
