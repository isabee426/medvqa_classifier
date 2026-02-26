"""Zero-shot evaluation: run BLIP on unseen VQA-RAD questions and test the probe.

Pipeline:
  1. Load VQA-RAD questions NOT used in training (HuggingFace test split).
  2. Run BLIP .generate() to get the model's actual answer for each question.
  3. Score each answer vs gold using normalized exact-match + token F1:
       label=0 (correct)       if exact_match OR token_f1 >= threshold
       label=1 (hallucinated)  otherwise
  4. Extract hidden states from those same (image, question, blip_answer) triples.
  5. Apply the Stage-2 probe → report AUC.

Usage:
    python -m fine_tuned.test.blip_inference_eval \
        --checkpoint outputs/checkpoints/stage2_hallucination/best_model.pt \
        --output outputs/eval/blip_inference \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score

from medvqa_probe.utils.logging import setup_logging

logger = setup_logging(name=__name__)

MODEL_ID = "Salesforce/blip-image-captioning-base"
LAYERS = [2, 6, 10]
TOKEN_F1_THRESHOLD = 0.5  # token-level F1 threshold for "correct"


# ---------------------------------------------------------------------------
# Text-based answer scoring (normalized exact-match + token F1)
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    """Normalize answer text: lowercase, strip articles/punctuation/whitespace."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def _token_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 between predicted and gold answers."""
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _score_answer(pred: str, gold: str) -> tuple[bool, float]:
    """Score predicted answer against gold.

    Returns (is_correct, token_f1).
    is_correct = True if normalized exact match OR token_f1 >= threshold.
    """
    norm_pred = _normalize_answer(pred)
    norm_gold = _normalize_answer(gold)
    exact = (norm_pred == norm_gold)
    f1 = _token_f1(pred, gold)
    is_correct = exact or (f1 >= TOKEN_F1_THRESHOLD)
    return is_correct, f1


# ---------------------------------------------------------------------------
# BLIP inference + judging
# ---------------------------------------------------------------------------

def _run_blip_inference(
    records: list[dict],
    image_cache_dir: Path,
    device: str,
    dtype: torch.dtype,
) -> list[dict]:
    """Run BLIP .generate() on each VQA-RAD question and score with text matching.

    Returns list of dicts with keys:
        question, gold_answer, blip_answer, token_f1,
        label (0=correct, 1=hallucinated), image_path
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    image_cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading BLIP model for inference...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    blip = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    ).to(device).eval()

    results = []
    skipped = 0
    for i, rec in enumerate(records):
        pil_image = rec.get("image")
        if pil_image is None:
            skipped += 1
            continue

        question = str(rec.get("question", ""))
        gold = str(rec.get("answer", ""))
        if not question or not gold:
            skipped += 1
            continue

        # Cache image to disk for feature extraction later.
        img_cache_path = image_cache_dir / f"vqarad_test_{i}.jpg"
        if not img_cache_path.exists():
            pil_image.convert("RGB").save(str(img_cache_path))

        try:
            image = pil_image.convert("RGB")
            prompt = f"Question: {question} Answer:"
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = blip.generate(**inputs, max_new_tokens=32)
            blip_answer = processor.decode(generated[0], skip_special_tokens=True)
            if "Answer:" in blip_answer:
                blip_answer = blip_answer.split("Answer:")[-1].strip()

            is_correct, f1 = _score_answer(blip_answer, gold)
            label = 0 if is_correct else 1

            results.append({
                "image_path": str(img_cache_path),
                "question": question,
                "gold_answer": gold,
                "blip_answer": blip_answer,
                "token_f1": round(f1, 4),
                "label": label,
            })

            if (i + 1) % 50 == 0:
                n_correct = sum(1 for r in results if r["label"] == 0)
                logger.info(
                    "Processed %d / %d  |  BLIP accuracy: %.1f%%  |  avg token F1: %.3f",
                    i + 1, len(records),
                    100 * n_correct / len(results),
                    sum(r["token_f1"] for r in results) / len(results),
                )

        except Exception as e:
            logger.warning("Error on record %d: %s", i, e)
            continue

    logger.info("Inference done. Skipped %d. Got %d results.", skipped, len(results))
    n_correct = sum(1 for r in results if r["label"] == 0)
    logger.info(
        "BLIP accuracy: %.1f%% (%d / %d correct)",
        100 * n_correct / max(len(results), 1), n_correct, len(results),
    )
    return results


# ---------------------------------------------------------------------------
# Feature extraction from inference results
# ---------------------------------------------------------------------------

def _extract_features(
    results: list[dict],
    device: str,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract probe features for each (image, question, blip_answer) triple."""
    from medvqa_probe.models.base_vqa_model import HFVLMModel
    from medvqa_probe.utils.config import ExtractionConfig

    cfg = ExtractionConfig(
        model_name_or_path=MODEL_ID,
        layers=LAYERS,
        segments=["vision", "question", "answer"],
        pooling="mean",
        device=device,
        dtype="float16" if dtype == torch.float16 else "float32",
    )

    vqa_model = HFVLMModel()
    vqa_model.load(cfg)

    features_list = []
    labels_list = []
    skipped = 0

    for i, res in enumerate(results):
        try:
            image = Image.open(res["image_path"]).convert("RGB")
            out = vqa_model.forward(image, res["question"], res["blip_answer"])

            seg_vecs = []
            for layer_idx in LAYERS:
                acts = out.activations.get(layer_idx)
                if acts is None:
                    continue
                acts_np = acts.squeeze(0).float().cpu().numpy()
                for mask in [out.vision_mask, out.question_mask, out.answer_mask]:
                    if mask is not None:
                        seg_acts = acts_np[mask.cpu().numpy().astype(bool)]
                        seg_vecs.append(seg_acts.mean(axis=0) if len(seg_acts) > 0
                                        else np.zeros(acts_np.shape[-1]))
                    else:
                        seg_vecs.append(np.zeros(acts_np.shape[-1]))

            if not seg_vecs:
                skipped += 1
                continue

            features_list.append(np.concatenate(seg_vecs))
            labels_list.append(res["label"])

            if (i + 1) % 100 == 0:
                logger.info("Extracted features %d / %d", i + 1, len(results))

        except Exception as e:
            logger.warning("Feature extraction error on %d: %s", i, e)
            skipped += 1
            continue

    logger.info("Feature extraction done. Skipped %d. Got %d feature vectors.", skipped, len(features_list))
    return np.stack(features_list), np.array(labels_list)


# ---------------------------------------------------------------------------
# Probe evaluation
# ---------------------------------------------------------------------------

def _eval_probe(
    features: np.ndarray,
    labels: np.ndarray,
    checkpoint_path: str,
    device: str,
) -> dict:
    """Load Stage-2 probe checkpoint and evaluate on the features."""
    from medvqa_probe.models.mlp_classifier import MLPClassifier
    from medvqa_probe.utils.config import ClassifierConfig

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Training saves either a bare state_dict or a wrapped dict.
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        classifier_cfg = raw.get("classifier_config", {})
        temperature = raw.get("temperature", 1.0)
    else:
        state_dict = raw
        classifier_cfg = {}
        temperature = 1.0

    cfg = ClassifierConfig(
        input_dim=features.shape[1],
        hidden_dim=classifier_cfg.get("hidden_dim", 512),
        num_layers=classifier_cfg.get("num_layers", 3),
        dropout=classifier_cfg.get("dropout", 0.30),
        num_classes=1,
    )
    probe = MLPClassifier(cfg).to(device).eval()
    probe.load_state_dict(state_dict, strict=False)

    X = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = probe(X).squeeze(-1).cpu().numpy()
    probs = torch.sigmoid(torch.tensor(logits / temperature)).numpy()

    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")

    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    logger.info("=== BLIP INFERENCE EVAL (reward-labeled) ===")
    logger.info("  Samples: %d correct (0) + %d hallucinated (1)", n0, n1)
    logger.info("  Probe accuracy:  %.4f", acc)
    logger.info("  Probe ROC-AUC:   %.4f", auc)
    logger.info("  Baseline (majority class): %.4f", max(n0, n1) / (n0 + n1))

    return {
        "n_correct_blip": n0,
        "n_hallucinated_blip": n1,
        "probe_accuracy": float(acc),
        "probe_roc_auc": float(auc),
        "majority_baseline": max(n0, n1) / (n0 + n1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global TOKEN_F1_THRESHOLD

    parser = argparse.ArgumentParser(description="Eval probe on BLIP's real outputs")
    parser.add_argument("--checkpoint", required=True, help="Stage-2 checkpoint .pt")
    parser.add_argument("--output", default="outputs/eval/blip_inference")
    parser.add_argument("--f1_threshold", type=float, default=TOKEN_F1_THRESHOLD,
                        help="Token F1 threshold for labeling correct (default: 0.5)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Cap number of questions (default: all)")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip BLIP inference; load existing inference_results.json")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_cache_dir = output_dir / "image_cache"
    inference_cache = output_dir / "inference_results.json"

    TOKEN_F1_THRESHOLD = args.f1_threshold

    # Step 1: BLIP inference + text-based scoring.
    if args.skip_inference:
        if not inference_cache.exists():
            raise FileNotFoundError(
                f"--skip_inference set but {inference_cache} not found. "
                "Run without --skip_inference first."
            )
        logger.info("Loading cached inference results from %s", inference_cache)
        with open(inference_cache) as f:
            results = json.load(f)
        logger.info("Loaded %d inference results", len(results))
    else:
        logger.info("Loading VQA-RAD from HuggingFace (test split)...")
        from datasets import load_dataset
        ds = load_dataset("flaviagiammarino/vqa-rad")
        records = list(ds["test"])
        if args.max_questions:
            records = records[:args.max_questions]
        logger.info("Using %d VQA-RAD test-split questions", len(records))

        results = _run_blip_inference(records, image_cache_dir, args.device, dtype)

        with open(inference_cache, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved inference results to %s", inference_cache)

    # Step 2: Extract BLIP hidden-state features.
    features, labels = _extract_features(results, args.device, dtype)
    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "labels.npy", labels)

    # Step 3: Evaluate probe.
    metrics = _eval_probe(features, labels, args.checkpoint, args.device)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", output_dir / "metrics.json")


if __name__ == "__main__":
    main()
