#!/bin/bash
# Full pipeline: HPO for all models → final training → test evaluation
# Run this AFTER C3D HPO has completed (or to run everything from scratch).
# Safe to re-run: skips steps already done (checks for output files).
#
# Usage:
#   bash run_pipeline.sh            # run everything that isn't done yet
#   bash run_pipeline.sh --from x3d # start from X3D (skip C3D)

set -e
cd /home/hswts124607/cow_reid_new
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=logs/pipeline_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "========================================================"
echo "Cow Re-ID Pipeline — $(date)"
echo "========================================================"

# ── Data preparation (idempotent) ────────────────────────────────────────────
if [ ! -f data/processed/dataset_metadata.json ]; then
    echo "[STEP 1] Data preparation..."
    python -m scripts.train --prepare_data
else
    echo "[STEP 1] Data already prepared — skipping."
fi

# ── HPO ──────────────────────────────────────────────────────────────────────
for MODEL in c3d x3d swin vivit; do
    HPARAMS="results/${MODEL}_best_hparams.json"
    if [ -f "$HPARAMS" ]; then
        echo "[HPO] $MODEL already optimised — skipping."
    else
        echo "[HPO] Running $MODEL HPO (30 trials)..."
        python -m scripts.train --hpo --model $MODEL
        echo "[HPO] $MODEL done. Best params: $(cat $HPARAMS)"
    fi
done

# ── Final training + test evaluation ─────────────────────────────────────────
echo "[FINAL] Training all models on train+val and evaluating on test..."
python -m scripts.train --final --all

echo "========================================================"
echo "Pipeline complete — $(date)"
echo "Results in: results/comparison_table.md"
echo "========================================================"
