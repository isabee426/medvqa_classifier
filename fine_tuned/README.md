# Stage-2: Hallucination Fine-Tuning

Fine-tunes the Stage-1 corruption classifier into a hallucination detector using [HALT-MedVQA](https://github.com/knowlab/halt-medvqa), a benchmark designed to test whether VLMs fabricate answers to medical questions.

## Results

| Metric | Stage-1 (corruption) | Stage-2 (hallucination) | 82% Baseline |
|--------|---------------------|------------------------|--------------|
| Accuracy | 74.4% | **99.3%** | 82% |
| ROC-AUC | 0.823 | **0.9996** | — |
| F1 | 0.748 | **0.992** | — |
| ECE | 0.065 | **0.006** | — |

## How it works

### Stage-1 → Stage-2 transfer

Stage-1 trained an MLP probe to detect synthetic corruptions (swapped answers/images) in VQA-RAD by reading BLIP's internal hidden states. Stage-2 warm-starts from those weights and fine-tunes on real hallucination labels from HALT-MedVQA.

### HALT-MedVQA dataset

HALT-MedVQA defines three hallucination scenarios. We use two (filtered to VQA-RAD radiology images only):

| Scenario | What happens | Correct answer |
|----------|-------------|----------------|
| **FAKE** | Question replaced with nonsensical medical question | "I do not know" |
| **SWAP** | Image swapped with unrelated radiology image | "I do not know" |

For each HALT record, we construct two triples:
- **Faithful (label=0)**: image + question + "I do not know" (correct abstention)
- **Hallucinated (label=1)**: image + question + wrong medical option (fabricated answer)

These are run through a frozen BLIP backbone. Hidden states from layers 2/6/10 are pooled per token segment (vision, question, answer) and concatenated into a 6912-dim feature vector — identical to Stage-1.

### Data pipeline

```
HALT-MedVQA JSONs (auto-downloaded from GitHub)
  + VQA-RAD images (synpic*.jpg from OSF)
    │
    ├── Filter to synpic* (VQA-RAD radiology only)
    ├── Create faithful + hallucinated triples
    ├── 70/30 train/test split
    │
    └── BLIP feature extraction (frozen, layers 2/6/10)
         │
         └── MLP probe (warm-start from Stage-1 v5)
              ├── lr=0.0001, weight_decay=0.05
              ├── Early stopping (patience=10)
              └── → 99.3% accuracy on hallucination detection
```

### Training details

- **Architecture**: 3-layer residual MLP, 512 hidden dim, GELU, dropout=0.30
- **Warm-start**: Stage-1 v5 checkpoint (corruption AUC=0.823)
- **Learning rate**: 0.0001 (4x lower than Stage-1 — fine-tuning, not from scratch)
- **Data**: 962 train / 414 test examples from HALT FAKE + SWAP scenarios
- **Labels**: balanced (~50/50 faithful vs hallucinated)

## Usage

### Setup

Download VQA-RAD images (needed to match HALT's image references):
```bash
# Images are auto-downloaded by the extraction script from OSF
# Or manually: https://osf.io/89kps/ → VQA_RAD Image Folder
```

### Run pipeline

```bash
# 1. Extract hallucination features
python -m fine_tuned extract_features --config fine_tuned/configs/extract_halt_medvqa.yaml

# 2. Fine-tune classifier
python -m fine_tuned train --config fine_tuned/configs/train_stage2_hallucination.yaml

# 3. Evaluate vs 82% baseline
python -m fine_tuned eval --config fine_tuned/configs/eval_stage2_hallucination.yaml
```

## File structure

```
fine_tuned/
  __init__.py, __main__.py           # CLI entry point
  data/
    halt_medvqa_loader.py            # Downloads HALT JSONs, filters synpic*, creates triples
  extract_hallucination_features.py  # Feature extraction (reuses Stage-1 BLIP + pooling)
  train_hallucination.py             # Training with warm-start from Stage-1
  eval_hallucination.py              # Eval with per-dataset breakdown + baseline comparison
  configs/
    extract_halt_medvqa.yaml         # Extraction config (layers, segments, image paths)
    train_stage2_hallucination.yaml  # Training config (lr, warm-start checkpoint)
    eval_stage2_hallucination.yaml   # Eval config
  tests/
    test_halt_loader.py              # Sanity tests
```

## References

- [HALT-MedVQA](https://github.com/knowlab/halt-medvqa) — Hallucination benchmark for Medical VQA
- [VQA-RAD](https://osf.io/89kps/) — Visual Question Answering in Radiology dataset
