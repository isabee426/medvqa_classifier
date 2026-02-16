"""Metrics computation for the hallucination classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    ece: float
    n_samples: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "ece": self.ece,
            "n_samples": self.n_samples,
        }

    def summary(self) -> str:
        lines = [f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
                 for k, v in self.to_dict().items()]
        return "\n".join(lines)


def expected_calibration_error(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE)."""
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, probs, n_bins=n_bins, strategy="uniform",
        )
    except ValueError:
        return 0.0
    # Bin counts via histogram.
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts, _ = np.histogram(probs, bins=bin_edges)
    non_empty = bin_counts[bin_counts > 0]
    if len(non_empty) == 0:
        return 0.0
    # ECE = weighted average of |accuracy - confidence| per bin.
    ece = np.sum(
        non_empty[: len(fraction_of_positives)]
        * np.abs(fraction_of_positives - mean_predicted_value)
    ) / len(labels)
    return float(ece)


def compute_binary_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Compute a full set of binary classification metrics.

    Args:
        labels: Ground‑truth binary labels (0 or 1).
        probs: Predicted probabilities in [0, 1].
        threshold: Decision threshold for hard predictions.
    """
    preds = (probs >= threshold).astype(int)
    return BinaryMetrics(
        accuracy=float(accuracy_score(labels, preds)),
        precision=float(precision_score(labels, preds, zero_division=0)),
        recall=float(recall_score(labels, preds, zero_division=0)),
        f1=float(f1_score(labels, preds, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        pr_auc=float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        ece=expected_calibration_error(labels, probs),
        n_samples=len(labels),
    )
