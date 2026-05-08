#!/bin/bash
#SBATCH --job-name=lora_package
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CPU-only submission packaging.
#
# Splits Stage 6 of slurm/eval.slurm into a standalone job so the zip can be
# built without queueing for an H200 just to do filesystem work. Use this
# when you trust the trained adapter (e.g. you already validated a previous
# run with the full eval.slurm) and only need the upload artifact.
#
# Submit standalone (uses outputs/lora_adapter):
#   sbatch slurm/package.slurm
#
# Or against a specific archived adapter:
#   LORA_ADAPTER_DIR=$HOME/Nemotron_Challenge/outputs/lora_adapter_<TRAIN_JOBID> \
#     sbatch slurm/package.slurm
#
# Output: outputs/submission_<this_job_id>.zip
#
# No GPU needed -- the script just reads the adapter directory and writes a
# zip. Login-node `apptainer exec` of the same command tends to hang because
# the container's import path drags in vLLM/transformers heavyweight modules
# that exceed login-node memory caps; running here gets a clean 24 GB
# allocation.

set -euo pipefail

PROJECT_DIR=$HOME/Nemotron_Challenge
TRAIN_CONTAINER=$SCRATCH/containers/nemotron_vllm.sif
ADAPTER_DIR=${LORA_ADAPTER_DIR:-$PROJECT_DIR/outputs/lora_adapter}
SUBMISSION_ZIP=outputs/submission_${SLURM_JOB_ID}.zip

echo "=== Stage 0: Slurm allocation check ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Adapter to package: $ADAPTER_DIR"

cd "$PROJECT_DIR"
mkdir -p outputs logs

echo ""
echo "=== Stage 1: Prerequisites ==="
test -f "$TRAIN_CONTAINER" || {
    echo "FAIL: container missing at $TRAIN_CONTAINER"
    exit 1
}
test -d "$ADAPTER_DIR" || {
    echo "FAIL: adapter missing at $ADAPTER_DIR"
    echo "Run train.slurm first, or set LORA_ADAPTER_DIR=/abs/path/to/adapter."
    exit 1
}
test -f "$ADAPTER_DIR/adapter_config.json" || {
    echo "FAIL: adapter_config.json missing -- adapter is incomplete."
    exit 1
}
test -f "$ADAPTER_DIR/adapter_model.safetensors" || {
    echo "FAIL: adapter_model.safetensors missing -- adapter is incomplete."
    exit 1
}
ls -lh "$ADAPTER_DIR"

echo ""
echo "=== Stage 2: Package submission ==="
# No --nv: this is CPU-only filesystem work, no GPU passthrough needed.
apptainer exec \
  --bind "$PROJECT_DIR:/workspace" \
  "$TRAIN_CONTAINER" \
  bash -lc "
    cd /workspace
    python3 scripts/package_submission.py \
      --adapter $ADAPTER_DIR \
      --output  $SUBMISSION_ZIP
  "

echo ""
echo "=== Submission artifact ==="
ls -lh "$SUBMISSION_ZIP"
echo ""
echo "Zip manifest:"
unzip -l "$SUBMISSION_ZIP" | head -20

echo ""
echo "Package job $SLURM_JOB_ID finished at: $(date)"
