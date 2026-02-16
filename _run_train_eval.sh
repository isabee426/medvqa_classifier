#!/bin/bash
set -e
cd /mnt/c/Users/Isabe/medvqa
echo "=== TRAINING ==="
.venv/bin/python -m medvqa_probe train_classifier --config configs/train_stage1_corruption.yaml
echo ""
echo "=== EVALUATION ==="
.venv/bin/python -m medvqa_probe eval_classifier --config configs/eval_stage1_corruption.yaml
echo ""
echo "=== DONE ==="
