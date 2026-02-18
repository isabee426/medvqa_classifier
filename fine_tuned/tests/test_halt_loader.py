"""Sanity tests for HALT-MedVQA loader."""

import pytest
from medvqa_probe.utils.config import DataConfig


def test_halt_synthetic_fallback_loads():
    """Test that the synthetic fallback produces valid examples."""
    from fine_tuned.data.halt_medvqa_loader import load_halt_medvqa

    cfg = DataConfig(
        dataset_name="halt-medvqa",
        hf_dataset_id="flaviagiammarino/vqa-rad",
        splits=["train", "test"],
        max_examples_per_split=5,
        label_type="hallucination",
    )
    pairs = load_halt_medvqa(cfg, image_cache_dir="/tmp/halt_test_cache")

    assert len(pairs) > 0, "Should produce at least some examples"

    for ex, label in pairs:
        assert label in (0, 1), f"Label must be 0 or 1, got {label}"
        assert ex.dataset_name == "halt-medvqa"
        assert ex.question, "Question should not be empty"
        assert ex.answer, "Answer should not be empty"
        assert ex.split in ("train", "test")

    labels = [l for _, l in pairs]
    assert 0 in labels, "Should have at least one faithful (label=0) example"
    assert 1 in labels, "Should have at least one hallucinated (label=1) example"


def test_halt_download_json():
    """Test that HALT JSON files can be downloaded from GitHub."""
    from pathlib import Path
    from fine_tuned.data.halt_medvqa_loader import _download_halt_json, HALT_GITHUB_URLS

    cache_dir = Path("/tmp/halt_json_cache")
    for scenario, url in HALT_GITHUB_URLS.items():
        records = _download_halt_json(url, cache_dir)
        assert len(records) > 0, f"Should download {scenario} records"
        assert "question" in records[0], "Records must have 'question' field"
        assert "option" in records[0], "Records must have 'option' field"
        assert "img" in records[0], "Records must have 'img' field"


def test_halt_filter_vqarad():
    """Test that VQA-RAD filtering works (synpic images only)."""
    from fine_tuned.data.halt_medvqa_loader import _filter_vqarad

    records = [
        {"img": "synpic12345.jpg", "question": "q1"},
        {"img": "test_0001.jpg", "question": "q2"},
        {"img": "synpic67890.jpg", "question": "q3"},
        {"img": "PMC123_Fig1.jpg", "question": "q4"},
    ]
    filtered = _filter_vqarad(records)
    assert len(filtered) == 2
    assert all(r["img"].startswith("synpic") for r in filtered)
