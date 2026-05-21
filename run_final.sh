#!/bin/bash
set -e
cd ~/cow_reid_new
PYTHON=~/cow_reid_new/venv/bin/python

echo "=== Evaluating SWIN (best checkpoint from epoch ~9) at $(date) ==="
PYTHONWARNINGS=ignore $PYTHON -W ignore -m scripts.train --model swin --eval_only \
    --checkpoint checkpoints/swin_best.pt --config configs/config.yaml \
    2>&1 | tee logs/swin_eval.log

echo "=== Training VIVIT (8 epochs) at $(date) ==="
PYTHONWARNINGS=ignore $PYTHON -W ignore -m scripts.train --model vivit \
    --config configs/config.yaml \
    2>&1 | tee logs/vivit_train.log

echo "=== All done at $(date) ==="
