"""Classifier training pipeline.

Usage:
    python -m medvqa_probe.train_classifier --config configs/train_stage1_corruption.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from medvqa_probe.data.features_dataset import FeatureDataset, load_feature_records
from medvqa_probe.models.mlp_classifier import MLPClassifier, build_loss_fn
from medvqa_probe.utils.config import FullConfig, load_config
from medvqa_probe.utils.logging import setup_logging
from medvqa_probe.utils.metrics import compute_binary_metrics

logger = setup_logging(name=__name__)


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_scheduler(optimizer, cfg, steps_per_epoch: int):
    import math
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = getattr(cfg, "warmup_steps", 0) or 0

    if cfg.scheduler == "cosine" and warmup_steps > 0:
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.step_gamma)
    return None


def run(cfg: FullConfig) -> None:
    tcfg = cfg.training
    _set_seed(tcfg.seed)
    device = torch.device(tcfg.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    features_dir = cfg.extraction.output_dir

    # --- Load feature records ---------------------------------------------
    train_records = load_feature_records(features_dir, splits=["train"], label_type=tcfg.label_type)
    val_records = load_feature_records(features_dir, splits=["test", "val"], label_type=tcfg.label_type)

    if not train_records:
        logger.error("No training records found in %s. Run extract_features first.", features_dir)
        sys.exit(1)

    train_ds = FeatureDataset(train_records)
    val_ds = FeatureDataset(val_records, mean=train_ds.mean.numpy(), std=train_ds.std.numpy()) if val_records else None
    logger.info("Train: %d examples, feature_dim=%d, labels=%s",
                len(train_ds), train_ds.feature_dim, train_ds.label_distribution())
    if val_ds:
        logger.info("Val:   %d examples, labels=%s", len(val_ds), val_ds.label_distribution())

    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, num_workers=tcfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, num_workers=tcfg.num_workers) if val_ds else None

    # --- Build model and optimizer ----------------------------------------
    ccfg = cfg.classifier
    ccfg.input_dim = train_ds.feature_dim
    model = MLPClassifier(ccfg).to(device)
    logger.info("Classifier:\n%s", model)

    loss_fn = build_loss_fn(tcfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = _build_scheduler(optimizer, tcfg, len(train_loader))

    # --- Training loop ----------------------------------------------------
    ckpt_dir = Path(tcfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -float("inf")
    best_epoch = -1
    history: list[dict] = []

    for epoch in range(1, tcfg.epochs + 1):
        # ---- train ----
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

        # ---- validate ----
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

        record = {"epoch": epoch, "train_loss": avg_train_loss, **val_metrics_dict}
        history.append(record)

        metric_val = val_metrics_dict.get(tcfg.checkpoint_metric.replace("val_", ""), val_metrics_dict.get("roc_auc", -avg_train_loss))
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_auc=%.4f  val_f1=%.4f",
            epoch, tcfg.epochs, avg_train_loss,
            val_metrics_dict.get("roc_auc", 0), val_metrics_dict.get("f1", 0),
        )

        if metric_val > best_metric:
            best_metric = metric_val
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            logger.info("  -> New best (%.4f) — saved checkpoint.", best_metric)

    # ---- Save final artifacts -------------------------------------------
    torch.save(model.state_dict(), ckpt_dir / "final_model.pt")
    np.savez(ckpt_dir / "norm_stats.npz", mean=train_ds.mean.numpy(), std=train_ds.std.numpy())
    summary = {
        "best_epoch": best_epoch,
        "best_metric": float(best_metric),
        "checkpoint_metric": tcfg.checkpoint_metric,
        "total_epochs": tcfg.epochs,
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
        },
        "history": history,
    }
    with open(ckpt_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training complete. Best epoch: %d (%.4f). Artifacts in %s", best_epoch, best_metric, ckpt_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the hallucination classifier probe.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
