#!/bin/bash
#SBATCH --job-name=lora_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# LoRA-adapted eval at Kaggle params on the local stratified holdout.
#
# Baseline eval and submission packaging are intentionally not here:
#   - base-model numbers come from prior Kaggle leaderboard submissions
#   - submission zip is built by slurm/package.slurm (CPU-only, separate job)
#
# Submit standalone:
#   sbatch slurm/eval.slurm
# Or against a specific archived adapter:
#   LORA_ADAPTER_DIR=$HOME/Nemotron_Challenge/outputs/lora_adapter_<JOB_ID> \
#     sbatch slurm/eval.slurm

set -euo pipefail

PROJECT_DIR=$HOME/Nemotron_Challenge
EVAL_CONTAINER=${VLLM_LORA_EVAL_CONTAINER:-$SCRATCH/containers/nemotron_vllm_v0200_cu129.sif}
ADAPTER_DIR=${LORA_ADAPTER_DIR:-$PROJECT_DIR/outputs/lora_adapter}

LORA_OUTPUT=outputs/lora_eval_${SLURM_JOB_ID}.jsonl

export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONNOUSERSITE=1

echo "=== Stage 0: Slurm allocation check ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Adapter under eval: $ADAPTER_DIR"
nvidia-smi

cd "$PROJECT_DIR"
mkdir -p outputs logs

echo ""
echo "=== Stage 1: Prerequisites ==="
test -f "$EVAL_CONTAINER" || {
    echo "FAIL: LoRA-capable eval container missing at $EVAL_CONTAINER"
    echo "Run: sbatch slurm/pull_vllm_lora_eval.slurm"
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
test -f data/stratified_holdout.jsonl || {
    echo "FAIL: data/stratified_holdout.jsonl missing -- run scripts/build_stratified_holdout.py on the local box first (after scripts/categorize.py)."
    exit 1
}
ls -lh "$ADAPTER_DIR"

echo ""
echo "=== Stage 2: Container GPU passthrough ==="
apptainer exec --nv "$EVAL_CONTAINER" nvidia-smi >/dev/null && echo "LoRA eval container: GPU OK"

# eval_kaggle.yaml hardcodes outputs/lora_adapter as the LoRA path. If the
# caller pointed LORA_ADAPTER_DIR at an archive, repoint outputs/lora_adapter
# at it via a symlink so the eval YAML resolves correctly without edits.
if [ "$ADAPTER_DIR" != "$PROJECT_DIR/outputs/lora_adapter" ]; then
    rm -rf $PROJECT_DIR/outputs/lora_adapter
    ln -s "$ADAPTER_DIR" $PROJECT_DIR/outputs/lora_adapter
    echo "Repointed outputs/lora_adapter -> $ADAPTER_DIR"
fi

echo ""
echo "=== Stage 3: LoRA-adapted eval ==="
apptainer exec --nv \
  --bind $PROJECT_DIR:/workspace \
  --bind $SCRATCH:$SCRATCH \
  "$EVAL_CONTAINER" \
  bash -lc "
    cd /workspace
    python3 scripts/baseline_generate.py \
      --config configs/eval_kaggle.yaml \
      --output $LORA_OUTPUT
  "

echo ""
echo "=== Stage 4: Score ==="
apptainer exec \
  --bind $PROJECT_DIR:/workspace \
  "$EVAL_CONTAINER" \
  bash -lc "
    cd /workspace
    python3 scripts/evaluate.py --predictions $LORA_OUTPUT --score --by-category
  "

echo ""
echo "=== Eval artifacts ==="
ls -lh $LORA_OUTPUT

echo ""
echo "Eval job $SLURM_JOB_ID finished at: $(date)"
