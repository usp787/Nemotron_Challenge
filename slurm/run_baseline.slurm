#!/bin/bash
#SBATCH --job-name=nemotron_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"

nvidia-smi

PROJECT_DIR=$HOME/Nemotron_Challenge
CONTAINER=$SCRATCH/containers/nemotron_vllm.sif

export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# See slurm/smoke_test.slurm for the rationale: ignore ~/.local user-site so
# Python uses the container's vllm 0.12.0 and matched flashinfer versions.
export PYTHONNOUSERSITE=1

apptainer exec --nv \
  --bind $PROJECT_DIR:/workspace \
  --bind $SCRATCH:$SCRATCH \
  $CONTAINER \
  bash -lc "
    cd /workspace

    python3 scripts/check_env.py

    python3 scripts/baseline_generate.py \
      --config configs/baseline_h200.yaml \
      --output outputs/predictions_${SLURM_JOB_ID}.jsonl
  "

echo "Finished at: $(date)"
