#!/bin/bash
set -e
cd ~/cow_reid_new
PYTHON=~/cow_reid_new/venv/bin/python
PYTHONWARNINGS=ignore

echo "==============================="
echo "Starting training: $(date)"
echo "==============================="

for MODEL in c3d x3d swin vivit; do
    echo ""
    echo ">>> Training $MODEL at $(date)"
    PYTHONWARNINGS=ignore $PYTHON -W ignore -m scripts.train --model $MODEL --config configs/config.yaml \
        2>&1 | tee logs/${MODEL}_train.log
    echo ">>> Done $MODEL at $(date)"
done

echo ""
echo "==============================="
echo "All models done: $(date)"
echo "==============================="
