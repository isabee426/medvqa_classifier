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
| ROC-AUC | **0.817** |
| Accuracy | 75.4% |
| Precision | 74.3% |
| Recall | 77.6% |
| F1 | 75.9% |
| PR-AUC | 74.3% |
| ECE (calibration) | 0.066 |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
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
- 4-layer residual MLP with LayerNorm, GELU activation, and dropout
- Input feature standardization (zero mean, unit variance)
- BCE loss with AdamW optimizer, cosine LR schedule with linear warmup
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
