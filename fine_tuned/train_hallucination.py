"""Stage-2 hallucination classifier training with warm-start from Stage-1.

Usage:
    python -m fine_tuned.train_hallucination \
        --config fine_tuned/configs/train_stage2_hallucination.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from medvqa_probe.data.features_dataset import FeatureDataset, load_feature_records
from medvqa_probe.models.mlp_classifier import MLPClassifier, build_loss_fn
from medvqa_probe.train_classifier import _build_scheduler, _set_seed
from medvqa_probe.utils.config import FullConfig, load_config
from medvqa_probe.utils.logging import setup_logging
from medvqa_probe.utils.metrics import compute_binary_metrics

logger = setup_logging(name=__name__)


def _load_records_multi(dirs: list[str], splits: list[str], label_type: str):
    """Load feature records from multiple directories."""
    all_records = []
    for d in dirs:
        try:
            records = load_feature_records(d, splits=splits, label_type=label_type)
            all_records.extend(records)
        except FileNotFoundError:
            logger.warning("Feature store not found at %s — skipping.", d)
    return all_records


def run(cfg: FullConfig) -> None:
    tcfg = cfg.training
    _set_seed(tcfg.seed)
    device = torch.device(tcfg.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load from multiple feature directories.
    features_dirs = tcfg.features_dirs or [cfg.extraction.output_dir]
    train_records = _load_records_multi(features_dirs, ["train"], tcfg.label_type)
    val_records = _load_records_multi(features_dirs, ["test", "val"], tcfg.label_type)

    if not train_records:
        logger.error("No training records found. Run extract_hallucination_features first.")
        sys.exit(1)

    noise_std = getattr(tcfg, "noise_std", 0.0) or 0.0
    train_ds = FeatureDataset(train_records, noise_std=noise_std)
    val_ds = FeatureDataset(val_records, mean=train_ds.mean.numpy(),
                            std=train_ds.std.numpy()) if val_records else None

    logger.info("Train: %d examples, feature_dim=%d, labels=%s",
                len(train_ds), train_ds.feature_dim, train_ds.label_distribution())
    if val_ds:
        logger.info("Val:   %d examples, labels=%s", len(val_ds), val_ds.label_distribution())

    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True,
                              num_workers=tcfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False,
                            num_workers=tcfg.num_workers) if val_ds else None

    # Build model.
    ccfg = cfg.classifier
    ccfg.input_dim = train_ds.feature_dim
    model = MLPClassifier(ccfg).to(device)

    # Warm-start from Stage-1 if configured.
    pretrained = getattr(tcfg, "pretrained_checkpoint", None)
    if pretrained and Path(pretrained).exists():
        state = torch.load(pretrained, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        logger.info("Warm-started from Stage-1 checkpoint: %s", pretrained)
    elif pretrained:
        logger.warning("Pretrained checkpoint not found: %s — training from scratch", pretrained)

    # Optionally freeze early layers.
    freeze_n = getattr(tcfg, "freeze_first_n_layers", 0) or 0
    if freeze_n > 0:
        frozen = 0
        for name, param in model.named_parameters():
            if frozen < freeze_n and "layers" in name:
                param.requires_grad = False
                frozen += 1
        logger.info("Froze %d layer parameters.", frozen)

    logger.info("Classifier:\n%s", model)
    loss_fn = build_loss_fn(tcfg).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tcfg.lr, weight_decay=tcfg.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, tcfg, len(train_loader))

    # Training loop.
    ckpt_dir = Path(tcfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -float("inf")
    best_epoch = -1
    no_improve = 0
    patience = getattr(tcfg, "patience", 0) or 0
    history: list[dict] = []

    # Track per-dataset info for val records.
    val_dataset_names = [r.dataset_name for r in val_records] if val_records else []

    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            feats = batch["features"].to(device)
            labels = batch["label"].to(device)
            logits = model(feats).squeeze(-1)

            if tcfg.loss == "ce":
                loss = loss_fn(logits, labels.long())
            else:
                loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validate.
        val_metrics_dict = {}
        if val_loader is not None:
            model.eval()
            all_labels, all_probs = [], []
            val_loss = 0.0
            vn = 0
            with torch.no_grad():
                for batch in val_loader:
                    feats = batch["features"].to(device)
                    labels = batch["label"].to(device)
                    logits = model(feats).squeeze(-1)
                    if tcfg.loss == "ce":
                        val_loss += loss_fn(logits, labels.long()).item()
                    else:
                        val_loss += loss_fn(logits, labels).item()
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.append(probs)
                    all_labels.append(labels.cpu().numpy())
                    vn += 1

            all_labels_np = np.concatenate(all_labels)
            all_probs_np = np.concatenate(all_probs)
            metrics = compute_binary_metrics(all_labels_np, all_probs_np)
            val_metrics_dict = metrics.to_dict()
            val_metrics_dict["val_loss"] = val_loss / max(vn, 1)

            # Per-dataset metrics.
            dataset_names_arr = np.array(val_dataset_names)
            for ds_name in set(val_dataset_names):
                mask = dataset_names_arr == ds_name
                if mask.sum() >= 10:
                    ds_metrics = compute_binary_metrics(all_labels_np[mask], all_probs_np[mask])
                    val_metrics_dict[f"{ds_name}_auc"] = ds_metrics.roc_auc

        record = {"epoch": epoch, "train_loss": avg_train_loss, **val_metrics_dict}
        history.append(record)

        metric_val = val_metrics_dict.get(
            tcfg.checkpoint_metric.replace("val_", ""),
            val_metrics_dict.get("roc_auc", -avg_train_loss),
        )
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_auc=%.4f  val_f1=%.4f",
            epoch, tcfg.epochs, avg_train_loss,
            val_metrics_dict.get("roc_auc", 0), val_metrics_dict.get("f1", 0),
        )

        if metric_val > best_metric:
            best_metric = metric_val
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            logger.info("  -> New best (%.4f) — saved checkpoint.", best_metric)
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                logger.info("Early stopping: no improvement for %d epochs.", patience)
                break

    # Save artifacts.
    torch.save(model.state_dict(), ckpt_dir / "final_model.pt")
    np.savez(ckpt_dir / "norm_stats.npz",
             mean=train_ds.mean.numpy(), std=train_ds.std.numpy())
    summary = {
        "stage": 2,
        "best_epoch": best_epoch,
        "best_metric": float(best_metric),
        "checkpoint_metric": tcfg.checkpoint_metric,
        "total_epochs": tcfg.epochs,
        "pretrained_checkpoint": pretrained or "none",
        "classifier": {
            "input_dim": ccfg.input_dim,
            "hidden_dim": ccfg.hidden_dim,
            "num_layers": ccfg.num_layers,
            "dropout": ccfg.dropout,
            "activation": ccfg.activation,
            "num_classes": ccfg.num_classes,
        },
        "training": {
            "lr": tcfg.lr,
            "batch_size": tcfg.batch_size,
            "loss": tcfg.loss,
            "scheduler": tcfg.scheduler,
            "seed": tcfg.seed,
            "features_dirs": features_dirs,
        },
        "history": history,
    }
    with open(ckpt_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training complete. Best epoch: %d (%.4f). Artifacts in %s",
                best_epoch, best_metric, ckpt_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage-2 hallucination classifier fine-tuning.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
