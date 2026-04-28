#!/bin/bash
#SBATCH --job-name=nemotron_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Walltime is 1h (not 30m) because Stage 5 may download ~60GB of BF16
# weights on a cold HF cache. Drop back to 00:30:00 once the cache at
# $SCRATCH/huggingface is warm.

set -euo pipefail

echo "=== Stage 0: Slurm allocation check ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

PROJECT_DIR=$HOME/Nemotron_Challenge
CONTAINER=$SCRATCH/containers/nemotron_vllm.sif

export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub

echo ""
echo "=== Stage 1: Apptainer GPU passthrough ==="
apptainer exec --nv $CONTAINER nvidia-smi

echo ""
echo "=== Stages 2-5: Python smoke test ==="
apptainer exec --nv \
  --bind $PROJECT_DIR:/workspace \
  --bind $SCRATCH:$SCRATCH \
  $CONTAINER \
  bash -lc "
    cd /workspace
    python3 scripts/check_env.py
    python3 scripts/smoke_test.py
  "

echo ""
echo "=== Stage 6: 5-prompt mini baseline ==="
apptainer exec --nv \
  --bind $PROJECT_DIR:/workspace \
  --bind $SCRATCH:$SCRATCH \
  $CONTAINER \
  bash -lc "
    cd /workspace
    python3 scripts/baseline_generate.py \
      --config configs/smoke_h200.yaml \
      --output outputs/smoke_predictions_${SLURM_JOB_ID}.jsonl
  "

echo ""
echo "Smoke test finished at: $(date)"
