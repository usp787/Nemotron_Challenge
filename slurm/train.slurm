#!/bin/bash
#SBATCH --job-name=lora_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# LoRA training job (paired with slurm/eval.slurm).
#
# First full-data attempt: 3K samples x 1 epoch, attention + Mamba + MoE
# (target_modules=all-linear) at rank 16, max_seq_len 4096. Effective batch 16.
# At ~30s/step on H200 with grad checkpointing this is ~1.5h of pure step time
# plus model load/save, comfortably fitting the 4h walltime.
#
# Submit eval.slurm with --dependency=afterok:$JOBID to score the result.
#
# Prerequisites (one-time, see docs/lora_strategy.md sections 3-4):
#   - data/lora_traces.jsonl produced by prepare_reasoning_traces.py
#   - $SCRATCH/containers/nemotron_vllm.sif present (training container)
#   - $SCRATCH/lora_pip with peft + accelerate + mamba_ssm + causal_conv1d

set -euo pipefail

echo "=== Stage 0: Slurm allocation check ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

PROJECT_DIR=$HOME/Nemotron_Challenge
TRAIN_CONTAINER=$SCRATCH/containers/nemotron_vllm.sif
LORA_PIP_DIR=$SCRATCH/lora_pip
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/lora_verification.yaml}

# train_lora.py writes to outputs/lora_adapter (per the YAML). After training
# we mv the real directory into a per-job archive and replace
# outputs/lora_adapter with a symlink to it, so eval.slurm's default path
# always resolves to the latest run while older runs remain pinnable via
# LORA_ADAPTER_DIR=outputs/lora_adapter_<job_id>.
ADAPTER_ARCHIVE=outputs/lora_adapter_${SLURM_JOB_ID}

export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Same rationale as smoke_test.slurm and lora_verification.slurm: ignore
# ~/.local user-site so we use the container's vllm 0.12.0 + matched torch.
export PYTHONNOUSERSITE=1
# Iteration 2 ran 3h with no visible step logs because Python's default block
# buffering held all stdout in a buffer until the process exited. Turning off
# buffering makes train_lora.py's [step N/187] lines show up live in tail -f
# so a non-learning run can be early-aborted instead of waiting walltime out.
export PYTHONUNBUFFERED=1

mkdir -p $PROJECT_DIR/outputs $PROJECT_DIR/logs

echo ""
echo "=== Stage 1: Prerequisites check ==="
test -f $PROJECT_DIR/data/sft_traces.jsonl || {
    echo "FAIL: data/sft_traces.jsonl missing -- run scripts/categorize.py + scripts/build_sft_traces.py on the local box first."
    exit 1
}
test -f $TRAIN_CONTAINER || {
    echo "FAIL: training vLLM container missing at $TRAIN_CONTAINER"
    exit 1
}
test -d $LORA_PIP_DIR/peft || {
    echo "FAIL: peft missing at $LORA_PIP_DIR -- see docs/lora_strategy.md section 3."
    exit 1
}
# mamba-ssm + causal-conv1d are required by Nemotron-H's trust_remote_code
# path. Fail fast here, before the model load attempts an unconditional import.
test -d $LORA_PIP_DIR/mamba_ssm || {
    echo "FAIL: mamba_ssm missing -- rebuild per docs/lora_strategy.md section 3."
    exit 1
}
test -d $LORA_PIP_DIR/causal_conv1d || {
    echo "FAIL: causal_conv1d missing -- rebuild per docs/lora_strategy.md section 3."
    exit 1
}
wc -l $PROJECT_DIR/data/sft_traces.jsonl
echo "Training config: $TRAIN_CONFIG"

echo ""
echo "=== Stage 2: Container GPU passthrough ==="
apptainer exec --nv $TRAIN_CONTAINER nvidia-smi >/dev/null && echo "training vLLM container: GPU OK"

echo ""
echo "=== Stage 3: Train LoRA adapter ==="
# train_lora.py saves to outputs/lora_adapter via the YAML. If that path is
# a stale symlink from a prior run, save_pretrained() would follow it and
# overwrite the previous run's archived adapter. Clear it so the trainer
# writes a fresh real directory.
rm -rf $PROJECT_DIR/outputs/lora_adapter

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
echo "=== Stage 4: Archive adapter to per-job dir ==="
# Trainer just wrote a real directory at outputs/lora_adapter. Move it to the
# per-job archive (mv avoids 1.7 GB of duplication and never preserves a
# stale symlink the way `cp -r` would), then replace outputs/lora_adapter
# with a symlink to the archive so eval.slurm's default path resolves.
if [ -L $PROJECT_DIR/outputs/lora_adapter ] || [ ! -d $PROJECT_DIR/outputs/lora_adapter ]; then
    echo "FAIL: outputs/lora_adapter is a symlink or missing -- trainer did not save a real directory."
    exit 1
fi
rm -rf $PROJECT_DIR/$ADAPTER_ARCHIVE
mv $PROJECT_DIR/outputs/lora_adapter $PROJECT_DIR/$ADAPTER_ARCHIVE
ln -s $PROJECT_DIR/$ADAPTER_ARCHIVE $PROJECT_DIR/outputs/lora_adapter
ls -lh $PROJECT_DIR/$ADAPTER_ARCHIVE
ls -la $PROJECT_DIR/outputs/lora_adapter

echo ""
echo "=== Stage 5: Adapter shape verification ==="
test -f $PROJECT_DIR/$ADAPTER_ARCHIVE/adapter_config.json || {
    echo "FAIL: adapter_config.json missing -- eval would reject this adapter."
    exit 1
}
echo "adapter_config.json:"
cat $PROJECT_DIR/$ADAPTER_ARCHIVE/adapter_config.json | python3 -m json.tool | head -n 40

echo ""
echo "Train job $SLURM_JOB_ID finished at: $(date)"
echo ""
echo "Next step:"
echo "  sbatch --dependency=afterok:$SLURM_JOB_ID slurm/eval.slurm"
echo "or, to evaluate this specific archive:"
echo "  LORA_ADAPTER_DIR=$PROJECT_DIR/$ADAPTER_ARCHIVE sbatch slurm/eval.slurm"
