"""Zero-shot evaluation: run BLIP on unseen VQA-RAD questions and test the probe.

Pipeline:
  1. Load VQA-RAD questions NOT used in training (HuggingFace test split).
  2. Run BLIP .generate() to get the model's actual answer for each question.
  3. Normalize and compare to ground truth → binary label
       0 = BLIP answered correctly (faithful)
       1 = BLIP answered incorrectly (hallucinated)
  4. Extract hidden states from those same (image, question, blip_answer) triples.
  5. Apply the Stage-2 probe → report AUC.

This is the most principled eval: no synthetic labels, real model errors.

Usage:
    python -m fine_tuned.test.blip_inference_eval \
        --checkpoint outputs/checkpoints/stage2_hallucination/best_model.pt \
        --images data/VQA_RAD_Image_Folder \
        --output outputs/eval/blip_inference \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score

from medvqa_probe.utils.logging import setup_logging

logger = setup_logging(name=__name__)

MODEL_ID = "Salesforce/blip-image-captioning-base"
LAYERS = [2, 6, 10]


# ---------------------------------------------------------------------------
# Answer normalization
# ---------------------------------------------------------------------------

def _normalize(answer: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    answer = answer.lower().strip()
    answer = re.sub(r"[^\w\s]", "", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


def _answers_match(pred: str, gold: str) -> bool:
    """Check if BLIP's answer matches ground truth."""
    pred_n = _normalize(pred)
    gold_n = _normalize(gold)
    if pred_n == gold_n:
        return True
    # Handle yes/no variants.
    yes_words = {"yes", "true", "correct", "right"}
    no_words  = {"no", "false", "incorrect", "wrong"}
    if gold_n in yes_words and pred_n in yes_words:
        return True
    if gold_n in no_words and pred_n in no_words:
        return True
    return False


# ---------------------------------------------------------------------------
# BLIP inference
# ---------------------------------------------------------------------------

def _run_blip_inference(
    records: list[dict],
    images_dir: Path,
    device: str,
    dtype: torch.dtype,
) -> list[dict]:
    """Run BLIP .generate() on each VQA-RAD question.

    Returns list of dicts with keys:
        question, gold_answer, blip_answer, label (0=correct, 1=wrong),
        image_path
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    logger.info("Loading BLIP model for inference...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    ).to(device).eval()

    results = []
    skipped = 0
    for i, rec in enumerate(records):
        img_name = rec.get("image_name", rec.get("image", ""))
        img_path = images_dir / img_name
        if not img_path.exists():
            img_path = images_dir / img_name.replace(".jpg", ".png")
        if not img_path.exists():
            skipped += 1
            continue

        question = str(rec.get("question", ""))
        gold = str(rec.get("answer", ""))
        if not question or not gold:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            # Prompt BLIP in QA mode.
            prompt = f"Question: {question} Answer:"
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=32)
            blip_answer = processor.decode(generated[0], skip_special_tokens=True)
            # Strip the prompt from the output if echoed.
            if "Answer:" in blip_answer:
                blip_answer = blip_answer.split("Answer:")[-1].strip()

            label = 0 if _answers_match(blip_answer, gold) else 1

            results.append({
                "image_path": str(img_path),
                "question": question,
                "gold_answer": gold,
                "blip_answer": blip_answer,
                "label": label,
            })

            if (i + 1) % 100 == 0:
                n_correct = sum(1 for r in results if r["label"] == 0)
                logger.info("Processed %d / %d  |  BLIP accuracy so far: %.1f%%",
                            i + 1, len(records), 100 * n_correct / len(results))

        except Exception as e:
            logger.warning("Error on record %d: %s", i, e)
            continue

    logger.info("Inference done. Skipped %d (no image). Got %d results.", skipped, len(results))
    n_correct = sum(1 for r in results if r["label"] == 0)
    logger.info("BLIP raw accuracy on these questions: %.1f%% (%d / %d correct)",
                100 * n_correct / max(len(results), 1), n_correct, len(results))
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
    from transformers import AutoProcessor, AutoModelForImageTextToText
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

            # Pool each segment per layer → concatenate (same as training).
            seg_vecs = []
            for layer_idx in LAYERS:
                acts = out.activations.get(layer_idx)
                if acts is None:
                    continue
                acts_np = acts.squeeze(0).float().cpu().numpy()  # (seq, hidden)
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
    from medvqa_probe.models.mlp_probe import MLPProbe
    from medvqa_probe.utils.config import ClassifierConfig

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classifier_cfg = ckpt.get("classifier_config", {})

    probe = MLPProbe(
        input_dim=features.shape[1],
        hidden_dim=classifier_cfg.get("hidden_dim", 512),
        num_layers=classifier_cfg.get("num_layers", 3),
        dropout=classifier_cfg.get("dropout", 0.30),
    ).to(device).eval()

    probe.load_state_dict(ckpt["model_state_dict"], strict=False)

    X = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = probe(X).squeeze(-1).cpu().numpy()

    # Temperature scaling if available.
    temperature = ckpt.get("temperature", 1.0)
    probs = torch.sigmoid(torch.tensor(logits / temperature)).numpy()

    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")

    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    logger.info("=== BLIP INFERENCE EVAL (real errors) ===")
    logger.info("  Samples: %d correct (0) + %d hallucinated (1)", n0, n1)
    logger.info("  Probe accuracy:  %.4f", acc)
    logger.info("  Probe ROC-AUC:   %.4f", auc)
    logger.info("  Baseline (majority class): %.4f", max(n0, n1) / (n0 + n1))

    return {
        "n_correct_blip": n0,
        "n_hallucinated_blip": n1,
        "probe_accuracy": acc,
        "probe_roc_auc": auc,
        "majority_baseline": max(n0, n1) / (n0 + n1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Eval probe on BLIP's real outputs")
    parser.add_argument("--checkpoint", required=True, help="Stage-2 checkpoint .pt")
    parser.add_argument("--images", required=True, help="VQA-RAD image folder")
    parser.add_argument("--output", default="outputs/eval/blip_inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Cap number of questions (default: all)")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    images_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load VQA-RAD questions from HuggingFace test split (clean holdout).
    logger.info("Loading VQA-RAD from HuggingFace (test split)...")
    from datasets import load_dataset
    ds = load_dataset("flaviagiammarino/vqa-rad")
    # Use the test split — never seen during training.
    records = list(ds["test"])
    if args.max_questions:
        records = records[:args.max_questions]
    logger.info("Using %d VQA-RAD test-split questions", len(records))

    # Step 1: Run BLIP inference.
    results = _run_blip_inference(records, images_dir, args.device, dtype)

    # Save inference results.
    with open(output_dir / "inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved inference results to %s", output_dir / "inference_results.json")

    # Step 2: Extract features.
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
