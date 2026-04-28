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
SMOKE_OUTPUT=outputs/smoke_predictions_${SLURM_JOB_ID}.jsonl

export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Apptainer auto-binds $HOME, which means ~/.local/lib/python3.12/site-packages
# (a stale user pip install) shadows the container's packages. That host-side
# vllm 0.19.1 has a flashinfer version mismatch and isn't what the model card
# recommends anyway. Force Python to ignore user-site so we use the container's
# vllm 0.12.0, which is exactly what the Nemotron 3 Nano model card requires
# (`vllm>=0.12.0`) and ships with internally-consistent flashinfer versions.
export PYTHONNOUSERSITE=1

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
      --output $SMOKE_OUTPUT
  "

echo ""
echo "=== Stage 6 JSONL verification ==="
cd "$PROJECT_DIR"
wc -l "$SMOKE_OUTPUT"
head -n 1 "$SMOKE_OUTPUT"
tail -n 1 "$SMOKE_OUTPUT"

echo ""
echo "Smoke test finished at: $(date)"
