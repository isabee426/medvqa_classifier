"""Tests for config loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from medvqa_probe.utils.config import FullConfig, load_config


class TestConfigLoading:
    def test_load_yaml(self):
        raw = {
            "data": {"dataset_name": "test-ds", "splits": ["train"]},
            "extraction": {"layers": [1, 2, 3], "pooling": "max"},
            "training": {"epochs": 5, "lr": 0.01},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(raw, f)
            path = f.name

        cfg = load_config(path)
        assert cfg.data.dataset_name == "test-ds"
        assert cfg.extraction.layers == [1, 2, 3]
        assert cfg.extraction.pooling == "max"
        assert cfg.training.epochs == 5
        assert cfg.training.lr == 0.01

    def test_defaults_when_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            path = f.name

        cfg = load_config(path)
        assert isinstance(cfg, FullConfig)
        assert cfg.data.dataset_name == "vqa-rad"
        assert cfg.training.epochs == 30

    def test_overrides(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"training": {"epochs": 10}}, f)
            path = f.name

        cfg = load_config(path, overrides={"training": {"epochs": 99}})
        assert cfg.training.epochs == 99
