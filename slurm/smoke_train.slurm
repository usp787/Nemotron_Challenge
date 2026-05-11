#!/bin/bash
#SBATCH --job-name=smoke_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 30-minute smoke shake-down for the THK-matching recipe.
#
# What this catches BEFORE we commit to a 14h production run:
#   1. Does the rank=32 + grad_accum=64 LoRA actually fit on a single H200?
#   2. Does AutoTokenizer load the Nemotron tokenizer from our HF cache?
#   3. How many of the 17,391 records overflow the 8192-token cap?
#      (silent overflow truncation is the iteration-3 flat-loss bug.)
#   4. Does the chat-template <think>/</think> boundary line up so that
#      train_lora.py's loss mask supervises the assistant turn?
#   5. Do 20 actual optimizer steps complete with loss decreasing?
#   6. Does the saved adapter have the right shape (adapter_config.json +
#      adapter_model.safetensors)?
#
# What it does NOT do: vLLM-load the adapter, evaluate on a holdout,
# package a submission. Those happen in slurm/lora_verification.slurm /
# slurm/eval.slurm against the production adapter.
#
# Prerequisites (must already be on the cluster):
#   - data/sft_combined.jsonl (the 17,391-record combined corpus from Phase 5)
#   - $SCRATCH/containers/nemotron_vllm.sif
#   - $SCRATCH/lora_pip with peft + accelerate + mamba_ssm + causal_conv1d
#
# Submit:
#   sbatch slurm/smoke_train.slurm
#   tail -f logs/smoke_train_<job_id>.out

set -euo pipefail

echo "=== Stage 0: Slurm allocation check ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

PROJECT_DIR=$HOME/Nemotron_Challenge
TRAIN_CONTAINER=$SCRATCH/containers/nemotron_vllm.sif
LORA_PIP_DIR=$SCRATCH/lora_pip
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/smoke_train.yaml}

DIAG_JSON=outputs/smoke_diag_${SLURM_JOB_ID}.json
SMOKE_ADAPTER_DIR=outputs/lora_adapter_smoke
SMOKE_ADAPTER_ARCHIVE=outputs/lora_adapter_smoke_${SLURM_JOB_ID}

export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

mkdir -p $PROJECT_DIR/outputs $PROJECT_DIR/logs

echo ""
echo "=== Stage 1: Prerequisites check ==="
test -f $PROJECT_DIR/data/sft_combined.jsonl || {
    echo "FAIL: data/sft_combined.jsonl missing — run Phase 5 (categorize.py + build_sft_traces.py + build_augmenter_traces.py + combine_traces.py) on the local box first, then git push & git pull."
    exit 1
}
test -f $TRAIN_CONTAINER || {
    echo "FAIL: training vLLM container missing at $TRAIN_CONTAINER"
    exit 1
}
test -d $LORA_PIP_DIR/peft || {
    echo "FAIL: peft not installed at $LORA_PIP_DIR — see docs/lora_strategy.md."
    exit 1
}
wc -l $PROJECT_DIR/data/sft_combined.jsonl

echo ""
echo "=== Stage 2: Container GPU passthrough ==="
apptainer exec --nv $TRAIN_CONTAINER nvidia-smi >/dev/null && echo "training container: GPU OK"

echo ""
echo "=== Stage 3: Pre-train diagnostics (tokenizer, length histogram, boundary) ==="
apptainer exec --nv \
  --bind $PROJECT_DIR:/workspace \
  --bind $SCRATCH:$SCRATCH \
  $TRAIN_CONTAINER \
  bash -lc "
    cd /workspace
    export PYTHONPATH=$LORA_PIP_DIR:\$PYTHONPATH
    python3 scripts/smoke_train_diagnostics.py \
      --data data/sft_combined.jsonl \
      --max-seq-len 8192 \
      --max-overflow-frac 0.10 \
      --sample 20 \
      --out-json $DIAG_JSON
  "

echo ""
echo "Diagnostic JSON written to $DIAG_JSON"
# Show the first ~60 lines of the JSON without using a fragile cat|json.tool|head
# pipe. Job 6704908 died here (exit 32) because head closed the pipe early,
# upstream SIGPIPE propagated, set -e killed the script *before* Stage 4 ran.
if [ -f "$PROJECT_DIR/$DIAG_JSON" ]; then
    head -n 60 "$PROJECT_DIR/$DIAG_JSON" || true
fi

echo ""
echo "=== Stage 4: 20-step LoRA train (rank=32, grad_accum=64) ==="
# Clean any prior smoke adapter so a missing-file at end means a real failure.
rm -rf $PROJECT_DIR/$SMOKE_ADAPTER_DIR

apptainer exec --nv \
  --bind $PROJECT_DIR:/workspace \
  --bind $SCRATCH:$SCRATCH \
  $TRAIN_CONTAINER \
  bash -lc "
    cd /workspace
    export PYTHONPATH=$LORA_PIP_DIR:\$PYTHONPATH
    python3 -c 'import torch; print(\"torch\", torch.__version__, \"cuda?\", torch.cuda.is_available())'
    python3 -c 'import transformers, peft, accelerate; print(\"transformers\", transformers.__version__, \"peft\", peft.__version__, \"accelerate\", accelerate.__version__)'
    python3 -c 'import mamba_ssm, causal_conv1d; print(\"mamba_ssm\", mamba_ssm.__version__, \"causal_conv1d\", causal_conv1d.__version__)'
    python3 scripts/train_lora.py --config $TRAIN_CONFIG
  "

echo ""
echo "=== Stage 5: Adapter shape verification ==="
test -d $PROJECT_DIR/$SMOKE_ADAPTER_DIR || {
    echo "FAIL: smoke adapter dir missing at $SMOKE_ADAPTER_DIR — training never wrote weights."
    exit 1
}
ls -lh $PROJECT_DIR/$SMOKE_ADAPTER_DIR
test -f $PROJECT_DIR/$SMOKE_ADAPTER_DIR/adapter_config.json || {
    echo "FAIL: adapter_config.json missing."
    exit 1
}
echo "adapter_config.json:"
cat $PROJECT_DIR/$SMOKE_ADAPTER_DIR/adapter_config.json | python3 -m json.tool

# Archive the smoke adapter for inspection without polluting the main
# outputs/lora_adapter symlink that lora_verification.slurm uses.
cp -r $PROJECT_DIR/$SMOKE_ADAPTER_DIR $PROJECT_DIR/$SMOKE_ADAPTER_ARCHIVE
echo ""
echo "Smoke adapter archived at $SMOKE_ADAPTER_ARCHIVE"

echo ""
echo "=== Smoke train PASSED — ready to run the full job ==="
echo "Diagnostic summary:   $DIAG_JSON"
echo "Smoke adapter (archived): $SMOKE_ADAPTER_ARCHIVE"
echo "Smoke adapter (live):     $SMOKE_ADAPTER_DIR"
echo ""
echo "Finished at: $(date)"
