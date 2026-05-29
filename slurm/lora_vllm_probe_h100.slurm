#!/bin/bash
#SBATCH --job-name=lora_vllm_probe_h100
#SBATCH --partition=sharing
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# One-prompt LoRA-enabled vLLM probe — H100 variant for the `sharing`
# partition (1-hour walltime cap).
#
# Used when the gpu partition's H200 nodes are queued up and we want a
# quick "does vLLM accept this adapter" answer before committing to a
# 15+ hour 3-chunk training run.
#
# Key differences vs slurm/lora_vllm_probe.slurm:
#   - partition=sharing (not gpu)
#   - gres=gpu:h100:1 (not gpu:h200:1)
#   - eval_lora_probe_h100.yaml uses gpu_memory_utilization=0.92 so the
#     ~65 GB of base model + adapter + buffers fits H100's 80 GB
#
# OPERATIONAL NOTE: the smoke / training adapter we want to verify
# typically lives at outputs/lora_adapter_smoke or
# outputs/lora_adapter_thk_chunk*. The probe config points at
# outputs/lora_adapter (the symlink the final training chunk promotes).
# Either symlink the adapter you want to probe before submitting:
#
#   ln -sfn $HOME/Nemotron_Challenge/outputs/lora_adapter_smoke \
#           $HOME/Nemotron_Challenge/outputs/lora_adapter
#
# or override via LORA_ADAPTER_DIR env var when submitting:
#
#   sbatch --export=ALL,LORA_ADAPTER_DIR=$HOME/Nemotron_Challenge/outputs/lora_adapter_smoke \
#          slurm/lora_vllm_probe_h100.slurm

set -euo pipefail

PROJECT_DIR=$HOME/Nemotron_Challenge
EVAL_CONTAINER=${VLLM_LORA_EVAL_CONTAINER:-$SCRATCH/containers/nemotron_vllm_v0200_cu129.sif}
ADAPTER_DIR=${LORA_ADAPTER_DIR:-$PROJECT_DIR/outputs/lora_adapter}
PROBE_OUTPUT=outputs/lora_probe_h100_${SLURM_JOB_ID}.jsonl

export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONNOUSERSITE=1

echo "=== Stage 0: Slurm allocation check ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

# Sanity: confirm an 80 GB H100, not a 40 GB variant. The model needs
# 60.6 GB just to load — 40 GB would OOM before generation even starts.
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU VRAM: ${VRAM_MIB} MiB"
if [ "${VRAM_MIB}" -lt 70000 ]; then
    echo "FAIL: allocated GPU has < 70 GB VRAM. The 60.6 GB base model will OOM."
    echo "      H100 80 GB is required. Resubmit and request again — the sharing"
    echo "      partition may have a 40 GB variant on a different node."
    exit 1
fi

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
test -f configs/eval_lora_probe_h100.yaml || {
    echo "FAIL: configs/eval_lora_probe_h100.yaml missing."
    exit 1
}
ls -lh "$EVAL_CONTAINER"
ls -lhL "$ADAPTER_DIR"

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
      --config configs/eval_lora_probe_h100.yaml \
      --output $PROBE_OUTPUT
  "

echo ""
echo "=== Probe JSONL verification ==="
wc -l "$PROBE_OUTPUT"
cat "$PROBE_OUTPUT"

echo ""
echo "Finished at: $(date)"
