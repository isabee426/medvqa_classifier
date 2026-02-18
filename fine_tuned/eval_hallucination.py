"""Stage-2 hallucination classifier evaluation with per-dataset breakdown.

Usage:
    python -m fine_tuned.eval_hallucination \
        --config fine_tuned/configs/eval_stage2_hallucination.yaml
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

from medvqa_probe.data.features_dataset import FeatureDataset, FeatureRecord, load_feature_records
from medvqa_probe.models.mlp_classifier import MLPClassifier
from medvqa_probe.utils.config import FullConfig, load_config
from medvqa_probe.utils.logging import setup_logging
from medvqa_probe.utils.metrics import compute_binary_metrics

logger = setup_logging(name=__name__)

BASELINE_ACCURACY = 0.82


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    def nll(t):
        scaled = logits / t
        log_p = -np.logaddexp(0, -scaled)
        log_1mp = -np.logaddexp(0, scaled)
        return -np.mean(labels * log_p + (1 - labels) * log_1mp)

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def _load_records_multi(dirs: list[str], splits: list[str], label_type: str):
    all_records: list[FeatureRecord] = []
    for d in dirs:
        try:
            all_records.extend(load_feature_records(d, splits=splits, label_type=label_type))
        except FileNotFoundError:
            logger.warning("Feature store not found at %s — skipping.", d)
    return all_records


def run(cfg: FullConfig) -> None:
    ecfg = cfg.eval
    device = torch.device(ecfg.device if torch.cuda.is_available() else "cpu")

    # Load from multiple dirs if configured.
    features_dirs = ecfg.features_dirs or ([ecfg.features_dir] if ecfg.features_dir else [])
    records = _load_records_multi(features_dirs, ecfg.splits, ecfg.label_type)

    if not records:
        logger.error("No records found for splits=%s, label_type=%s in %s",
                      ecfg.splits, ecfg.label_type, features_dirs)
        sys.exit(1)

    # Track dataset names for per-dataset breakdown.
    dataset_names = [r.dataset_name for r in records]

    # Normalization.
    norm_path = Path(ecfg.checkpoint_path).parent / "norm_stats.npz"
    if norm_path.exists():
        norm = np.load(norm_path)
        ds = FeatureDataset(records, mean=norm["mean"], std=norm["std"])
        logger.info("Using training normalization stats from %s", norm_path)
    else:
        ds = FeatureDataset(records)
    loader = DataLoader(ds, batch_size=ecfg.batch_size, shuffle=False)
    logger.info("Evaluating on %d examples (feature_dim=%d, labels=%s)",
                len(ds), ds.feature_dim, ds.label_distribution())

    # Load model.
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

    # Inference.
    all_ids: list[str] = []
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            logits = model(feats).squeeze(-1)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["label"].numpy())
            all_ids.extend(batch["id"])

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)

    # Temperature scaling.
    temperature = _fit_temperature(logits_np, labels_np)
    logger.info("Learned temperature: %.4f", temperature)
    calibrated_logits = logits_np / temperature
    probs_np = 1.0 / (1.0 + np.exp(-calibrated_logits))

    temp_path = Path(ecfg.checkpoint_path).parent / "temperature.json"
    with open(temp_path, "w") as f:
        json.dump({"temperature": float(temperature)}, f)

    # Global metrics.
    metrics = compute_binary_metrics(labels_np, probs_np)
    out_dir = Path(ecfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== GLOBAL METRICS ===")
    logger.info("\n%s", metrics.summary())

    # Baseline comparison.
    logger.info("=== vs BASELINE ===")
    logger.info("  Current accuracy:  %.4f", metrics.accuracy)
    logger.info("  Baseline accuracy: %.4f", BASELINE_ACCURACY)
    if metrics.accuracy > BASELINE_ACCURACY:
        logger.info("  BEATS BASELINE by %.2f pp", (metrics.accuracy - BASELINE_ACCURACY) * 100)
    else:
        logger.info("  Below baseline by %.2f pp", (BASELINE_ACCURACY - metrics.accuracy) * 100)

    # Per-dataset metrics.
    dataset_names_arr = np.array(dataset_names)
    per_dataset = {}
    unique_datasets = sorted(set(dataset_names))
    if len(unique_datasets) > 1:
        logger.info("=== PER-DATASET METRICS ===")
        for ds_name in unique_datasets:
            mask = dataset_names_arr == ds_name
            if mask.sum() < 5:
                continue
            ds_metrics = compute_binary_metrics(labels_np[mask], probs_np[mask])
            per_dataset[ds_name] = ds_metrics.to_dict()
            logger.info("  %s (n=%d): AUC=%.4f  Acc=%.4f  F1=%.4f",
                        ds_name, mask.sum(), ds_metrics.roc_auc,
                        ds_metrics.accuracy, ds_metrics.f1)

    # Save results.
    results = {
        **metrics.to_dict(),
        "baseline_accuracy": BASELINE_ACCURACY,
        "beats_baseline": bool(metrics.accuracy > BASELINE_ACCURACY),
        "per_dataset": per_dataset,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved metrics to %s", out_dir / "metrics.json")

    # Per-example scores.
    if ecfg.dump_scores:
        scores_path = out_dir / "per_example_scores.csv"
        with open(scores_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "dataset", "label", "score", "hallucination_type"])
            for eid, ds, label, prob, ctype in zip(
                all_ids, dataset_names, labels_np, probs_np, ds.corruption_types
            ):
                writer.writerow([eid, ds, int(label), f"{prob:.6f}", ctype or ""])
        logger.info("Per-example scores saved to %s", scores_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage-2 hallucination classifier.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
