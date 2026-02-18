# MedVQA Internal-State Hallucination Classifier

A config-driven pipeline for detecting hallucinations in medical VQA models by probing their internal representations. Trains a classifier on BLIP backbone hidden states extracted from [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) to distinguish clean (aligned) from corrupted (misaligned) image-question-answer triples.

Based on Razvan-style corruption detection as a proxy for hallucination, following the approach described in [Lau et al. (2018)](https://www.nature.com/articles/sdata2018251).

## Pipeline Overview

```
VQA-RAD dataset
      |
      v
[Corruption Generator] --> clean + corrupted (image, question, answer) triples
      |
      v
[BLIP Feature Extractor] --> pooled hidden states from transformer layers
      |                       (vision / question / answer tokens, per layer)
      v
[MLP Probe Classifier] --> binary prediction: clean (1) vs corrupted (0)
      |
      v
[Temperature-Scaled Evaluation] --> calibrated metrics (AUC, F1, ECE)
```

## Results

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.8227** |
| Accuracy | 74.4% |
| Precision | 73.6% |
| Recall | 76.1% |
| F1 | 74.8% |
| PR-AUC | 75.8% |
| ECE (calibration) | 0.065 |

**Ablations** (test set ROC-AUC):

| Config | AUC |
|--------|-----|
| Logistic Regression baseline | 0.697 |
| MLP 4L/512, wd=0.01 (original) | 0.817 |
| MLP 4L/512, wd=0.05, noise=0.02 (v3) | 0.8159 |
| MLP 2L/256, wd=0.10, noise=0.03 (v4) | 0.8133 |
| **MLP 3L/512, wd=0.05, noise=0.02 (v5)** | **0.8227** |

## Setup

**Local (Python venv):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

**Docker (extract → train → eval pipeline):**
```bash
docker compose run --rm extract   # extracts features
docker compose run --rm train     # trains classifier
docker compose run --rm eval      # evaluates checkpoint
```

## Usage

### 1. Extract features

Runs the BLIP model on VQA-RAD examples (clean + corrupted), hooks internal transformer layers, pools activations by token type, and saves feature vectors to disk.

```bash
python -m medvqa_probe extract_features --config configs/extract_vqarad_stage1.yaml
```

### 2. Train classifier

Trains a residual MLP probe on the extracted features (binary: clean vs corrupted).

```bash
python -m medvqa_probe train_classifier --config configs/train_stage1_corruption.yaml
```

### 3. Evaluate

Loads the best checkpoint, applies temperature scaling for calibration, and reports metrics.

```bash
python -m medvqa_probe eval_classifier --config configs/eval_stage1_corruption.yaml
```

## Architecture

### Feature Extraction
- **Base model**: Salesforce/blip-image-captioning-base (BLIP v1)
- **Hooked layers**: 2, 6, 10 (early / middle / late)
- **Token segments**: vision, question, answer (mean-pooled separately)
- **Feature vector**: concatenation of all pooled representations across layers and segments

### Corruption Strategies
- **swap_answer**: Replace answer with one from a different example
- **swap_image**: Pair question+answer with a different radiology image
- **empty_answer**: Replace answer with "", "N/A", or "[BLANK]"
- **distort_answer**: Swap anatomy/finding terms (e.g., lung->liver, left->right, benign->malignant)

### Classifier
- 3-layer residual MLP with LayerNorm, GELU activation, and dropout (0.30)
- Input feature standardization (zero mean, unit variance)
- Gaussian noise augmentation on features during training (noise_std=0.02)
- BCE loss with AdamW optimizer (weight_decay=0.05), cosine LR schedule with linear warmup
- Early stopping on val AUC (patience=15)
- Gradient clipping (max_norm=1.0)
- Post-hoc temperature scaling (Platt scaling) for calibration

## Project Structure

```
medvqa_probe/
  data/
    vqarad_loader.py      # VQA-RAD dataset loading (HuggingFace or local)
    corruptions.py         # Razvan-style corruption strategies
    features_dataset.py    # Feature storage, loading, PyTorch Dataset
  models/
    base_vqa_model.py      # BLIP backbone with activation hooks
    mlp_classifier.py      # Residual MLP probe + loss functions
  extract_features.py      # Feature extraction CLI
  train_classifier.py      # Training CLI
  eval_classifier.py       # Evaluation CLI with temperature scaling
  utils/
    config.py              # YAML/JSON config loading and schema
    logging.py             # Logging setup
    metrics.py             # Binary classification metrics (AUC, F1, ECE)
configs/
  extract_vqarad_stage1.yaml
  train_stage1_corruption.yaml
  eval_stage1_corruption.yaml
scripts/
  sklearn_baseline.py          # LR baseline for sanity-checking nonlinearity
tests/
  test_config.py
  test_data.py
  test_training.py
```

## Configuration

All pipeline stages are config-driven via YAML files. Key configurable parameters:

- Which transformer layers to hook
- Pooling strategy (mean / max / cls)
- Which token segments to include
- Corruption strategies and their probabilities
- Classifier architecture (hidden dim, num layers, dropout, activation)
- Training hyperparameters (LR, scheduler, warmup, loss function, epochs)
- Number of output classes (binary or multi-class for future hallucination labels)

## Stage 2: Hallucination Labels (Future)

The code is designed to support a second label type (`label_type="hallucination"`) using judge-model labels or answer-correctness signals. The same feature extraction and classifier infrastructure can be reused by changing the config to point at hallucination labels instead of corruption labels.
