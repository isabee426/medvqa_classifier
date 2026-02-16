"""Classifier evaluation script.

Usage:
    python -m medvqa_probe.eval_classifier --config configs/eval_stage1_corruption.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from medvqa_probe.data.features_dataset import FeatureDataset, load_feature_records
from medvqa_probe.models.mlp_classifier import MLPClassifier
from medvqa_probe.utils.config import FullConfig, load_config
from medvqa_probe.utils.logging import setup_logging
from medvqa_probe.utils.metrics import compute_binary_metrics

logger = setup_logging(name=__name__)


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Learn a single temperature scalar to minimize NLL (Platt scaling)."""
    from scipy.optimize import minimize_scalar

    def nll(t):
        scaled = logits / t
        # Numerically stable sigmoid + BCE.
        log_p = -np.logaddexp(0, -scaled)
        log_1mp = -np.logaddexp(0, scaled)
        return -np.mean(labels * log_p + (1 - labels) * log_1mp)

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def run(cfg: FullConfig) -> None:
    ecfg = cfg.eval
    device = torch.device(ecfg.device if torch.cuda.is_available() else "cpu")

    records = load_feature_records(ecfg.features_dir, splits=ecfg.splits, label_type=ecfg.label_type)
    if not records:
        logger.error("No records found for splits=%s, label_type=%s in %s",
                      ecfg.splits, ecfg.label_type, ecfg.features_dir)
        sys.exit(1)

    # Load normalization stats from training if available.
    norm_path = Path(ecfg.checkpoint_path).parent / "norm_stats.npz"
    if norm_path.exists():
        norm = np.load(norm_path)
        ds = FeatureDataset(records, mean=norm["mean"], std=norm["std"])
        logger.info("Using train normalization stats from %s", norm_path)
    else:
        ds = FeatureDataset(records)
    loader = DataLoader(ds, batch_size=ecfg.batch_size, shuffle=False)
    logger.info("Evaluating on %d examples (feature_dim=%d, labels=%s)",
                len(ds), ds.feature_dim, ds.label_distribution())

    # --- Load model -------------------------------------------------------
    ccfg = cfg.classifier
    ccfg.input_dim = ds.feature_dim
    model = MLPClassifier(ccfg).to(device)

    ckpt = ecfg.checkpoint_path
    if not ckpt or not Path(ckpt).exists():
        logger.error("Checkpoint not found: %s", ckpt)
        sys.exit(1)

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    logger.info("Loaded checkpoint: %s", ckpt)

    # --- Inference --------------------------------------------------------
    all_ids: list[str] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            logits = model(feats).squeeze(-1)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["label"].numpy())
            all_ids.extend(batch["id"])

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)

    # --- Temperature scaling (post-hoc calibration) -----------------------
    temperature = _fit_temperature(logits_np, labels_np)
    logger.info("Learned temperature: %.4f", temperature)
    calibrated_logits = logits_np / temperature
    probs_np = 1.0 / (1.0 + np.exp(-calibrated_logits))

    # Save temperature alongside checkpoint for future use.
    temp_path = Path(ecfg.checkpoint_path).parent / "temperature.json"
    with open(temp_path, "w") as f:
        json.dump({"temperature": float(temperature)}, f)

    metrics = compute_binary_metrics(labels_np, probs_np)

    # --- Report -----------------------------------------------------------
    out_dir = Path(ecfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Metrics:\n%s", metrics.summary())

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    logger.info("Saved metrics to %s", out_dir / "metrics.json")

    # --- Per-example scores -----------------------------------------------
    if ecfg.dump_scores:
        scores_path = out_dir / "per_example_scores.csv"
        corruption_types = ds.corruption_types
        with open(scores_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "score", "corruption_type"])
            for eid, label, prob, ctype in zip(all_ids, labels_np, probs_np, corruption_types):
                writer.writerow([eid, int(label), f"{prob:.6f}", ctype or ""])
        logger.info("Per-example scores saved to %s", scores_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the hallucination classifier probe.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
