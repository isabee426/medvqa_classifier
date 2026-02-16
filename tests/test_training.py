"""Lightweight training sanity checks — runs on CPU, no real model needed."""

from __future__ import annotations

import numpy as np
import torch

from medvqa_probe.data.features_dataset import FeatureDataset, FeatureRecord
from medvqa_probe.models.mlp_classifier import FocalLoss, MLPClassifier, build_loss_fn
from medvqa_probe.utils.config import ClassifierConfig, TrainingConfig
from medvqa_probe.utils.metrics import compute_binary_metrics


def _make_dataset(n: int = 40, dim: int = 64) -> FeatureDataset:
    records = [
        FeatureRecord(
            id=f"s_{i}", dataset_name="test", split="train",
            features=np.random.randn(dim).astype(np.float32),
            label=i % 2, label_type="corruption",
        )
        for i in range(n)
    ]
    return FeatureDataset(records)


class TestMLPClassifier:
    def test_forward_shape(self):
        cfg = ClassifierConfig(input_dim=64, hidden_dim=32, num_layers=2, num_classes=1)
        model = MLPClassifier(cfg)
        x = torch.randn(8, 64)
        out = model(x)
        assert out.shape == (8, 1)

    def test_multiclass_output(self):
        cfg = ClassifierConfig(input_dim=64, hidden_dim=32, num_layers=1, num_classes=3)
        model = MLPClassifier(cfg)
        x = torch.randn(4, 64)
        out = model(x)
        assert out.shape == (4, 3)


class TestOverfit:
    """Verify that the classifier can overfit a tiny batch."""

    def test_overfit_tiny_batch(self):
        torch.manual_seed(0)
        ds = _make_dataset(n=8, dim=32)
        cfg = ClassifierConfig(input_dim=32, hidden_dim=64, num_layers=2, num_classes=1)
        model = MLPClassifier(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        feats = ds.features
        labels = ds.labels

        initial_loss = None
        for step in range(200):
            logits = model(feats).squeeze(-1)
            loss = loss_fn(logits, labels)
            if step == 0:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.1, (
            f"Model did not overfit: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )


class TestLossFunctions:
    def test_bce_loss(self):
        tcfg = TrainingConfig(loss="bce")
        fn = build_loss_fn(tcfg)
        logits = torch.randn(4)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
        loss = fn(logits, labels)
        assert loss.item() > 0

    def test_focal_loss(self):
        fl = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(4)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
        loss = fl(logits, labels)
        assert loss.item() > 0

    def test_ce_loss(self):
        tcfg = TrainingConfig(loss="ce")
        tcfg.class_weights = None
        fn = build_loss_fn(tcfg)
        # For CE, logits shape = (batch, num_classes), labels = long
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 2, 0])
        loss = fn(logits, labels)
        assert loss.item() > 0


class TestMetrics:
    def test_perfect_binary(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.9, 0.8])
        m = compute_binary_metrics(labels, probs)
        assert m.accuracy == 1.0
        assert m.roc_auc == 1.0

    def test_random_binary(self):
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 2, size=100)
        probs = rng.rand(100)
        m = compute_binary_metrics(labels, probs)
        assert 0 <= m.accuracy <= 1
        assert 0 <= m.roc_auc <= 1
        assert m.n_samples == 100
