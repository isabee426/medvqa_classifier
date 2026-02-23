"""Zero-shot evaluation: run BLIP on unseen VQA-RAD questions and test the probe.

Pipeline:
  1. Load VQA-RAD questions NOT used in training (HuggingFace test split).
  2. Run BLIP .generate() to get the model's actual answer for each question.
  3. InternLM-XComposer2.5-Reward scores each answer vs gold answer:
       R = 1 - exp(-(S(pred) - S(gold)) * beta)  if S(pred) > S(gold)
       R = 0                                       otherwise
       R > 0  → label=0 (correct / faithful)
       R = 0  → label=1 (wrong / hallucinated)
  4. Extract hidden states from those same (image, question, blip_answer) triples.
  5. Apply the Stage-2 probe → report AUC.

Reward model: internlm/internlm-xcomposer2d5-7b-reward
Paper: https://arxiv.org/abs/2501.12368
Formula from: https://arxiv.org/html/2504.11468v1

Usage:
    python -m fine_tuned.test.blip_inference_eval \
        --checkpoint outputs/checkpoints/stage2_hallucination/best_model.pt \
        --output outputs/eval/blip_inference \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score

from medvqa_probe.utils.logging import setup_logging

logger = setup_logging(name=__name__)

MODEL_ID = "Salesforce/blip-image-captioning-base"
DEFAULT_JUDGE = "internlm/internlm-xcomposer2d5-7b-reward"
LAYERS = [2, 6, 10]
BETA = 1.0  # smoothing parameter from paper


# ---------------------------------------------------------------------------
# Reward model judge (InternLM-XComposer2.5-Reward)
# ---------------------------------------------------------------------------

def _load_judge(judge_model: str, device: str):
    """Load InternLM-XComposer2.5-Reward model."""
    from transformers import AutoModel, AutoTokenizer, AutoConfig

    logger.info("Loading reward model: %s", judge_model)
    tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)

    # The model's __init__ reads config.max_length which may not be set.
    config = AutoConfig.from_pretrained(judge_model, trust_remote_code=True)
    if not hasattr(config, "max_length"):
        config.max_length = 4096

    model = AutoModel.from_pretrained(
        judge_model,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=False,  # model __init__ loads CLIP internally; meta device breaks this
    ).to(device).eval()
    model.tokenizer = tokenizer
    return model


def _sanitize(text: str, max_chars: int = 512) -> str:
    """Strip special tokens and control characters, cap length."""
    import re
    # Remove anything that looks like a special token (<xxx>)
    text = re.sub(r"<[^>]{1,20}>", "", text)
    # Remove non-printable / control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    text = text.strip()
    # Fallback for empty string after sanitization
    if not text:
        text = "unknown"
    return text[:max_chars]


def _judge_score(
    judge_model,
    image_path: str,
    question: str,
    gold: str,
    pred: str,
) -> float:
    """Score BLIP's answer vs gold using the reward model.

    Applies the paper's formula:
        R = 1 - exp(-(S(pred) - S(gold)) * beta)  if S(pred) > S(gold)
        R = 0                                       otherwise

    Returns R in [0, 1]. R > 0 means pred scored better than gold.
    """
    question = _sanitize(question, max_chars=256)
    gold     = _sanitize(gold,     max_chars=256)
    pred     = _sanitize(pred,     max_chars=256)

    chat_pred = [
        {"role": "user",      "content": "<ImageHere>" + question},
        {"role": "assistant", "content": pred},
    ]
    chat_gold = [
        {"role": "user",      "content": "<ImageHere>" + question},
        {"role": "assistant", "content": gold},
    ]
    image = [image_path]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        scores = judge_model.get_scores(
            [chat_pred, chat_gold], [image, image], hd_num=4
        )

    s_pred, s_gold = scores[0], scores[1]

    diff = s_pred - s_gold
    # Paper formula for continuous reward (used for logging).
    if diff > 0:
        reward = 1.0 - math.exp(-diff * BETA)
    else:
        reward = 0.0

    return s_pred, s_gold, reward


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
    """Run BLIP .generate() on each VQA-RAD question and score with reward model.

    Returns list of dicts with keys:
        question, gold_answer, blip_answer, reward (0–1),
        label (0=correct, 1=wrong), image_path
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    image_cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading BLIP model for inference...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    blip = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    ).to(device).eval()

    judge = _load_judge(judge_model_id, device)

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

        # Cache image to disk — reward model needs a file path.
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

            # Reward model: label=0 (correct) if pred scores >= gold.
            s_pred, s_gold, reward = _judge_score(
                judge, str(img_cache_path), question, gold, blip_answer
            )
            label = 0 if s_pred >= s_gold else 1

            results.append({
                "image_path": str(img_cache_path),
                "question": question,
                "gold_answer": gold,
                "blip_answer": blip_answer,
                "score_pred": round(float(s_pred), 4),
                "score_gold": round(float(s_gold), 4),
                "reward": round(reward, 4),
                "label": label,
            })

            if (i + 1) % 50 == 0:
                n_correct = sum(1 for r in results if r["label"] == 0)
                avg_r = sum(r["reward"] for r in results) / len(results)
                logger.info(
                    "Processed %d / %d  |  BLIP accuracy: %.1f%%  |  avg reward: %.3f",
                    i + 1, len(records),
                    100 * n_correct / len(results),
                    avg_r,
                )

        except Exception as e:
            logger.warning("Error on record %d: %s", i, e)
            continue

    logger.info("Inference done. Skipped %d. Got %d results.", skipped, len(results))
    n_correct = sum(1 for r in results if r["label"] == 0)
    logger.info(
        "BLIP accuracy (reward-labeled): %.1f%% (%d / %d correct)",
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
    parser = argparse.ArgumentParser(description="Eval probe on BLIP's real outputs (reward model judge)")
    parser.add_argument("--checkpoint", required=True, help="Stage-2 checkpoint .pt")
    parser.add_argument("--output", default="outputs/eval/blip_inference")
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE,
                        help=f"Reward model HF ID (default: {DEFAULT_JUDGE})")
    parser.add_argument("--beta", type=float, default=BETA,
                        help="Reward smoothing parameter (default: 1.0)")
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

    # Step 1: BLIP inference + reward model scoring.
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
