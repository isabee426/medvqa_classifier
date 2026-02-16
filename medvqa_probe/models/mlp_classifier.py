"""Simple configurable MLP probe for binary / multi-class classification."""

from __future__ import annotations

import torch
import torch.nn as nn

from medvqa_probe.utils.config import ClassifierConfig


class ResidualBlock(nn.Module):
    """Linear -> LayerNorm -> activation -> dropout, with a residual skip."""

    def __init__(self, dim: int, dropout: float, act_fn: nn.Module) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            act_fn,
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MLPClassifier(nn.Module):
    """MLP probe with residual connections after the first projection layer.

    - ``num_classes=1`` -> single logit output (binary, use BCEWithLogitsLoss).
    - ``num_classes>1`` -> multi-class (use CrossEntropyLoss).
    """

    def __init__(self, cfg: ClassifierConfig) -> None:
        super().__init__()
        assert cfg.input_dim is not None, "input_dim must be set before building the classifier."
        act_fn = nn.GELU() if cfg.activation == "gelu" else nn.ReLU()

        layers: list[nn.Module] = [
            # First layer projects from input_dim -> hidden_dim.
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            act_fn,
            nn.Dropout(cfg.dropout),
        ]

        # Remaining hidden layers use residual connections (same dim -> same dim).
        for _ in range(cfg.num_layers - 1):
            layers.append(ResidualBlock(cfg.hidden_dim, cfg.dropout, act_fn))

        layers.append(nn.Linear(cfg.hidden_dim, cfg.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits -- shape (batch, num_classes) or (batch, 1)."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary focal loss (sigmoid version)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def build_loss_fn(cfg) -> nn.Module:
    """Build a loss function from config."""
    if cfg.loss == "bce":
        weight = None
        if cfg.class_weights:
            # For BCE, use pos_weight (ratio of neg/pos).
            weight = torch.tensor([cfg.class_weights[1] / cfg.class_weights[0]])
        return nn.BCEWithLogitsLoss(pos_weight=weight)
    elif cfg.loss == "focal":
        return FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    elif cfg.loss == "ce":
        weight = torch.tensor(cfg.class_weights, dtype=torch.float32) if cfg.class_weights else None
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Unknown loss: {cfg.loss}")
