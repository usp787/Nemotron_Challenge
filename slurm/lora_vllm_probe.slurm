#!/bin/bash
#SBATCH --job-name=lora_vllm_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# One-prompt LoRA-enabled vLLM probe.
#
# Default GPU target is H200 to match the full verification path. Explorer's
# A100 pool may include 40 GB cards, which is too ambiguous for this LoRA load
# probe.

set -euo pipefail

PROJECT_DIR=$HOME/Nemotron_Challenge
EVAL_CONTAINER=${VLLM_LORA_EVAL_CONTAINER:-$SCRATCH/containers/nemotron_vllm_v0200_cu129.sif}
ADAPTER_DIR=${LORA_ADAPTER_DIR:-$PROJECT_DIR/outputs/lora_adapter}
PROBE_OUTPUT=outputs/lora_probe_${SLURM_JOB_ID}.jsonl

export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONNOUSERSITE=1

echo "=== Stage 0: Slurm allocation check ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

cd "$PROJECT_DIR"
mkdir -p outputs logs

echo ""
echo "=== Stage 1: Prerequisites ==="
test -f "$EVAL_CONTAINER" || {
    echo "FAIL: newer vLLM eval container missing at $EVAL_CONTAINER"
    echo "Run: sbatch slurm/pull_vllm_lora_eval.slurm"
    exit 1
}
test -d "$ADAPTER_DIR" || {
    echo "FAIL: LoRA adapter missing at $ADAPTER_DIR"
    echo "Run a LoRA training stage first, or set LORA_ADAPTER_DIR=/path/to/adapter."
    exit 1
}
test -f data/lora_probe_prompt.jsonl || {
    echo "FAIL: data/lora_probe_prompt.jsonl missing."
    exit 1
}
ls -lh "$EVAL_CONTAINER"
ls -lh "$ADAPTER_DIR"

echo ""
echo "=== Stage 2: Container GPU passthrough ==="
apptainer exec --nv "$EVAL_CONTAINER" nvidia-smi >/dev/null
echo "new vLLM eval container: GPU OK"

echo ""
echo "=== Stage 3: One-prompt LoRA generation ==="
apptainer exec --nv \
  --bind "$PROJECT_DIR:/workspace" \
  --bind "$SCRATCH:$SCRATCH" \
  "$EVAL_CONTAINER" \
  bash -lc "
    cd /workspace
    python3 -c 'import torch, vllm; print(\"torch\", torch.__version__, \"cuda?\", torch.cuda.is_available()); print(\"vllm\", getattr(vllm, \"__version__\", \"unknown\"))'
    python3 scripts/baseline_generate.py \
      --config configs/eval_lora_probe.yaml \
      --output $PROBE_OUTPUT
  "

echo ""
echo "=== Probe JSONL verification ==="
wc -l "$PROBE_OUTPUT"
cat "$PROBE_OUTPUT"

echo ""
echo "Finished at: $(date)"
