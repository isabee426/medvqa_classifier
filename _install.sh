#!/bin/bash
set -e
PIP=/mnt/c/Users/Isabe/medvqa/.venv/bin/pip
$PIP install pytest pyyaml numpy scikit-learn
$PIP install torch --index-url https://download.pytorch.org/whl/cpu
