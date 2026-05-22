#!/bin/bash
# Run HPO for X3D, Swin, ViViT in PARALLEL on GPUs 1-2-3.
# Waits for all three, then runs final training (all 4 models, one per GPU).
# Skips any model whose best_hparams.json already exists.
set -e
cd /home/hswts124607/cow_reid_new
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

echo "=========================================="
echo "Parallel HPO on GPUs 1-2-3 — $(date)"
echo "=========================================="

# ── Launch HPO jobs in parallel ──────────────────────────────────────────────
declare -A PIDS

for IDX_MODEL in "1 x3d" "2 swin" "3 vivit"; do
    GPU=$(echo $IDX_MODEL | cut -d' ' -f1)
    MODEL=$(echo $IDX_MODEL | cut -d' ' -f2)
    HPARAMS="results/${MODEL}_best_hparams.json"

    if [ -f "$HPARAMS" ]; then
        echo "[$MODEL] Already optimised — skipping."
    else
        LOG="logs/hpo_${MODEL}_$(date +%Y%m%d_%H%M%S).log"
        echo "[$MODEL] Launching on GPU $GPU → $LOG"
        CUDA_VISIBLE_DEVICES=$GPU \
            python -m scripts.train --hpo --model $MODEL \
            > "$LOG" 2>&1 &
        PIDS[$MODEL]=$!
    fi
done

# ── Wait for all HPO jobs ─────────────────────────────────────────────────────
echo ""
echo "Waiting for parallel HPO jobs to finish..."
ALL_OK=true
for MODEL in "${!PIDS[@]}"; do
    PID=${PIDS[$MODEL]}
    echo -n "  [$MODEL pid=$PID] "
    if wait "$PID"; then
        echo "done ✓"
    else
        echo "FAILED ✗  (check logs/hpo_${MODEL}_*.log)"
        ALL_OK=false
    fi
done

# ── Final training (parallel, one model per GPU) ─────────────────────────────
echo ""
echo "All HPO done. Running final training in parallel on all 4 GPUs..."

declare -A FINAL_PIDS

for IDX_MODEL in "0 c3d" "1 x3d" "2 swin" "3 vivit"; do
    GPU=$(echo $IDX_MODEL | cut -d' ' -f1)
    MODEL=$(echo $IDX_MODEL | cut -d' ' -f2)
    LOG="logs/final_${MODEL}_$(date +%Y%m%d_%H%M%S).log"
    echo "  [final $MODEL] GPU $GPU → $LOG"
    CUDA_VISIBLE_DEVICES=$GPU \
        python -m scripts.train_final --model $MODEL \
        > "$LOG" 2>&1 &
    FINAL_PIDS[$MODEL]=$!
done

echo "Waiting for final training..."
for MODEL in "${!FINAL_PIDS[@]}"; do
    PID=${FINAL_PIDS[$MODEL]}
    echo -n "  [final $MODEL pid=$PID] "
    if wait "$PID"; then
        echo "done ✓"
    else
        echo "FAILED ✗  (check logs/final_${MODEL}_*.log)"
    fi
done

# ── Build comparison table ────────────────────────────────────────────────────
echo ""
echo "Building comparison table..."
python -c "
import json
from pathlib import Path
from scripts.evaluate import build_results_table
results_dir = Path('results')
all_results = []
for model in ['c3d', 'x3d', 'swin', 'vivit']:
    p = results_dir / f'{model}_results.json'
    if p.exists():
        d = json.load(open(p))
        all_results.append(d['summary'])
if all_results:
    build_results_table(all_results, str(results_dir))
else:
    print('No results found.')
"

echo "=========================================="
echo "DONE — $(date)"
echo "Results: results/comparison_table.md"
echo "=========================================="
