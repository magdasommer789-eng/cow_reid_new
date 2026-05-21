#!/bin/bash
cd ~/cow_reid_new
PYTHON=~/cow_reid_new/venv/bin/python

echo "=== Evaluating C3D ==="
PYTHONWARNINGS=ignore $PYTHON -W ignore -m scripts.train --model c3d --eval_only     --checkpoint checkpoints/c3d_best.pt --config configs/config.yaml     2>&1 | tee logs/c3d_eval.log

echo "=== Evaluating X3D ==="
PYTHONWARNINGS=ignore $PYTHON -W ignore -m scripts.train --model x3d --eval_only     --checkpoint checkpoints/x3d_best.pt --config configs/config.yaml     2>&1 | tee logs/x3d_eval.log

echo "=== Eval done ==="
