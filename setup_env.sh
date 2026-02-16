#!/bin/bash
set -e
PROJ=/mnt/c/Users/Isabe/medvqa
VENV=$PROJ/.venv
$VENV/bin/pip install pytest pyyaml numpy scikit-learn
$VENV/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
echo "--- DONE ---"
