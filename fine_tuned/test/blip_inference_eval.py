"""Zero-shot evaluation: run BLIP on unseen VQA-RAD questions and test the probe.

Pipeline:
  1. Load VQA-RAD questions NOT used in training (HuggingFace test split).
  2. Run BLIP .generate() to get the model's actual answer for each question.
  3. LLM judge scores each (question, gold, blip_answer) triple → 0.0–1.0
       score >= 0.5  → label=0 (correct / faithful)
       score <  0.5  → label=1 (wrong / hallucinated)
  4. Extract hidden states from those same (image, question, blip_answer) triples.
  5. Apply the Stage-2 probe → report AUC.

This is the most principled eval: no synthetic labels, real model errors,
semantically-aware correctness scoring via an LLM judge.

Usage:
    python -m fine_tuned.test.blip_inference_eval \
        --checkpoint outputs/checkpoints/stage2_hallucination/best_model.pt \
        --output outputs/eval/blip_inference \
        --device cuda

    # Use a different judge model (default: Qwen/Qwen2.5-1.5B-Instruct):
    python -m fine_tuned.test.blip_inference_eval \
        --checkpoint outputs/checkpoints/stage2_hallucination/best_model.pt \
        --judge_model Qwen/Qwen2.5-7B-Instruct \
        --output outputs/eval/blip_inference \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score

from medvqa_probe.utils.logging import setup_logging

logger = setup_logging(name=__name__)

MODEL_ID = "Salesforce/blip-image-captioning-base"
DEFAULT_JUDGE = "Qwen/Qwen2.5-1.5B-Instruct"
LAYERS = [2, 6, 10]

JUDGE_PROMPT = """\
You are a medical VQA evaluator. Given a question, a gold answer, and a predicted answer, \
decide whether the predicted answer is semantically correct.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Is the predicted answer correct? Reply with a single word: yes or no."""


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def _load_judge(judge_model: str, device: str):
    """Load a small instruction-tuned LLM for answer judging."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    logger.info("Loading judge model: %s", judge_model)
    tokenizer = AutoTokenizer.from_pretrained(judge_model)
    model = AutoModelForCausalLM.from_pretrained(
        judge_model, torch_dtype=torch.float16, device_map=device
    ).eval()
    return tokenizer, model


def _judge_score(
    tokenizer,
    judge_model,
    question: str,
    gold: str,
    pred: str,
    device: str,
) -> float:
    """Return probability in [0, 1] that the prediction is correct.

    Uses the log-probability of the 'yes' token vs 'no' token at the first
    generated position — no sampling needed, just a single forward pass.
    """
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)

    # Apply chat template if available, otherwise use raw prompt.
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = judge_model(**inputs)
        # logits shape: (1, seq_len, vocab_size) — take the last position
        next_token_logits = out.logits[0, -1, :]

    # Get token IDs for "yes" and "no" (first token of each word).
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids  = tokenizer.encode("no",  add_special_tokens=False)

    yes_logit = next_token_logits[yes_ids[0]].float()
    no_logit  = next_token_logits[no_ids[0]].float()

    # Softmax over just the two options → probability of "yes".
    score = torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)[0].item()
    return score


# ---------------------------------------------------------------------------
# BLIP inference + judging
# ---------------------------------------------------------------------------

def _run_blip_inference(
    records: list[dict],
    image_cache_dir: Path,
    device: str,
    dtype: torch.dtype,
    judge_model_id: str,
) -> list[dict]:
    """Run BLIP .generate() on each VQA-RAD question and judge correctness.

    HF VQA-RAD records have an "image" key holding a PIL.Image directly.
    Images are cached to disk so downstream feature extraction can open them.

    Returns list of dicts with keys:
        question, gold_answer, blip_answer, judge_score (0–1),
        label (0=correct, 1=wrong), image_path
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    image_cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading BLIP model for inference...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    blip = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    ).to(device).eval()

    judge_tok, judge_lm = _load_judge(judge_model_id, device)

    results = []
    skipped = 0
    for i, rec in enumerate(records):
        # HF dataset stores image as PIL.Image under "image" key.
        pil_image = rec.get("image")
        if pil_image is None:
            skipped += 1
            continue

        question = str(rec.get("question", ""))
        gold = str(rec.get("answer", ""))
        if not question or not gold:
            skipped += 1
            continue

        # Cache image to disk so _extract_features can reload it.
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

            # LLM judge: P(answer is correct) in [0, 1].
            score = _judge_score(judge_tok, judge_lm, question, gold, blip_answer, device)
            label = 0 if score >= 0.5 else 1

            results.append({
                "image_path": str(img_cache_path),
                "question": question,
                "gold_answer": gold,
                "blip_answer": blip_answer,
                "judge_score": round(score, 4),
                "label": label,
            })

            if (i + 1) % 50 == 0:
                n_correct = sum(1 for r in results if r["label"] == 0)
                avg_score = sum(r["judge_score"] for r in results) / len(results)
                logger.info(
                    "Processed %d / %d  |  BLIP accuracy: %.1f%%  |  avg judge score: %.3f",
                    i + 1, len(records),
                    100 * n_correct / len(results),
                    avg_score,
                )

        except Exception as e:
            logger.warning("Error on record %d: %s", i, e)
            continue

    logger.info("Inference done. Skipped %d. Got %d results.", skipped, len(results))
    n_correct = sum(1 for r in results if r["label"] == 0)
    logger.info(
        "BLIP accuracy (judge-labeled): %.1f%% (%d / %d correct)",
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
    logger.info("=== BLIP INFERENCE EVAL (judge-labeled) ===")
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
    parser = argparse.ArgumentParser(description="Eval probe on BLIP's real outputs (LLM judge)")
    parser.add_argument("--checkpoint", required=True, help="Stage-2 checkpoint .pt")
    parser.add_argument("--output", default="outputs/eval/blip_inference")
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE,
                        help=f"HF model ID for LLM judge (default: {DEFAULT_JUDGE})")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Cap number of questions (default: all)")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_cache_dir = output_dir / "image_cache"

    logger.info("Loading VQA-RAD from HuggingFace (test split)...")
    from datasets import load_dataset
    ds = load_dataset("flaviagiammarino/vqa-rad")
    records = list(ds["test"])
    if args.max_questions:
        records = records[:args.max_questions]
    logger.info("Using %d VQA-RAD test-split questions", len(records))

    # Step 1: Run BLIP inference + LLM judge scoring.
    results = _run_blip_inference(records, image_cache_dir, args.device, dtype, args.judge_model)

    with open(output_dir / "inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved inference results to %s", output_dir / "inference_results.json")

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
