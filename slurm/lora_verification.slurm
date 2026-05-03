#!/bin/bash
#SBATCH --job-name=lora_verification
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# LoRA verification pipeline.
#
# Goal: prove the full path runs end-to-end on a single H200:
#   data prep (already done on login node) -> train LoRA in the known-good
#   vLLM v0.12 container -> verify adapter shape -> evaluate at Kaggle eval
#   params in the newer LoRA eval vLLM container -> score -> package zip.
#
# This is a *verification* run, not a tuning run: tiny dataset, one epoch.
# Pass criterion is "every stage produces the expected artifact". Quality
# of the score is secondary on this pass.
#
# Prerequisites (do these once, BEFORE submitting this script):
#   1. data/lora_traces.jsonl exists
#        python3 scripts/prepare_reasoning_traces.py --num 200    # login node
#   2. data/aime25.jsonl exists (verification eval set)
#        python3 scripts/prepare_aime25.py                        # login node
#   3. Training vLLM container exists at $SCRATCH/containers/nemotron_vllm.sif
#        Already in place from the smoke-test bring-up.
#   4. LoRA eval vLLM container exists at
#        $SCRATCH/containers/nemotron_vllm_v0200_cu129.sif
#        sbatch slurm/pull_vllm_lora_eval.slurm
#        sbatch slurm/lora_vllm_probe.slurm
#   5. peft (and accelerate) installed under $SCRATCH/lora_pip
#        See docs/lora_strategy.md, "Setup" — one-time pip install
#        --target into $SCRATCH so we don't need a second container.

set -euo pipefail

echo "=== Stage 0: Slurm allocation check ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

PROJECT_DIR=$HOME/Nemotron_Challenge
TRAIN_CONTAINER=$SCRATCH/containers/nemotron_vllm.sif
EVAL_CONTAINER=${VLLM_LORA_EVAL_CONTAINER:-$SCRATCH/containers/nemotron_vllm_v0200_cu129.sif}
LORA_PIP_DIR=$SCRATCH/lora_pip

ADAPTER_DIR=outputs/lora_adapter_${SLURM_JOB_ID}
EVAL_OUTPUT=outputs/lora_eval_${SLURM_JOB_ID}.jsonl
SUBMISSION_ZIP=outputs/submission_${SLURM_JOB_ID}.zip

export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Same rationale as smoke_test.slurm: ignore ~/.local user-site so the
# container uses its own bundled Python packages, except for the explicit
# scratch-resident peft/accelerate install we add via PYTHONPATH below.
export PYTHONNOUSERSITE=1

# Adapter path is referenced by configs/eval_kaggle.yaml; the eval config
# expects outputs/lora_adapter, so we materialise the per-job dir there
# via a stable symlink at the end of training.
STABLE_ADAPTER_LINK=$PROJECT_DIR/outputs/lora_adapter

mkdir -p $PROJECT_DIR/outputs $PROJECT_DIR/logs

echo ""
echo "=== Stage 1: Prerequisites check ==="
test -f $PROJECT_DIR/data/lora_traces.jsonl || {
    echo "FAIL: data/lora_traces.jsonl missing — run prepare_reasoning_traces.py on the login node first."
    exit 1
}
test -f $PROJECT_DIR/data/aime25.jsonl || {
    echo "FAIL: data/aime25.jsonl missing — run prepare_aime25.py on the login node first."
    exit 1
}
test -f $TRAIN_CONTAINER || {
    echo "FAIL: training vLLM container missing at $TRAIN_CONTAINER"
    exit 1
}
test -f $EVAL_CONTAINER || {
    echo "FAIL: newer LoRA eval vLLM container missing at $EVAL_CONTAINER"
    echo "Run: sbatch slurm/pull_vllm_lora_eval.slurm"
    echo "Then run: sbatch slurm/lora_vllm_probe.slurm"
    exit 1
}
test -d $LORA_PIP_DIR/peft || {
    echo "FAIL: peft not installed at $LORA_PIP_DIR — see docs/lora_strategy.md, 'Setup'"
    exit 1
}
wc -l $PROJECT_DIR/data/lora_traces.jsonl $PROJECT_DIR/data/aime25.jsonl

# mamba-ssm + causal-conv1d must be present in $LORA_PIP_DIR and ABI-matched
# to the container's torch. Nemotron-H's trust_remote_code modeling code does
# an unconditional mamba_ssm import — there is no PyTorch fallback. See
# docs/lora_strategy.md §3 "Mamba-2 SSM extension" for the rebuild recipe.
test -d $LORA_PIP_DIR/mamba_ssm || {
    echo "FAIL: mamba_ssm missing from $LORA_PIP_DIR — rebuild per docs/lora_strategy.md §3."
    exit 1
}
test -d $LORA_PIP_DIR/causal_conv1d || {
    echo "FAIL: causal_conv1d missing from $LORA_PIP_DIR — rebuild per docs/lora_strategy.md §3."
    exit 1
}

echo ""
echo "=== Stage 2: Container GPU passthrough ==="
apptainer exec --nv $TRAIN_CONTAINER nvidia-smi >/dev/null && echo "training vLLM container: GPU OK"
apptainer exec --nv $EVAL_CONTAINER nvidia-smi >/dev/null && echo "LoRA eval vLLM container: GPU OK"

echo ""
echo "=== Stage 3: Train LoRA adapter (vLLM container + scratch peft) ==="
# Inject the scratch-resident peft/accelerate via PYTHONPATH so the vLLM
# container picks them up without modifying the image. The container's
# own torch + transformers are reused.
apptainer exec --nv \
  --bind $PROJECT_DIR:/workspace \
  --bind $SCRATCH:$SCRATCH \
  $TRAIN_CONTAINER \
  bash -lc "
    cd /workspace
    export PYTHONPATH=$LORA_PIP_DIR:\$PYTHONPATH
    python3 -c 'import torch; print(\"torch\", torch.__version__, \"cuda?\", torch.cuda.is_available())'
    python3 -c 'import transformers, peft, accelerate; print(\"transformers\", transformers.__version__, \"peft\", peft.__version__, \"accelerate\", accelerate.__version__)'
    # mamba-ssm and causal-conv1d are REQUIRED by Nemotron-H's trust_remote_code
    # path. Import them here so an ABI mismatch fails fast (before the model
    # download), instead of crashing 60s into model load.
    python3 -c 'import mamba_ssm, causal_conv1d; print(\"mamba_ssm\", mamba_ssm.__version__, \"causal_conv1d\", causal_conv1d.__version__)'
    python3 scripts/train_lora.py --config configs/lora_verification.yaml
  "

# Promote outputs/lora_adapter to a per-job directory so reruns don't
# clobber each other and the submission package keeps a record.
if [ -d $PROJECT_DIR/outputs/lora_adapter ]; then
    rm -rf $PROJECT_DIR/$ADAPTER_DIR
    cp -r $PROJECT_DIR/outputs/lora_adapter $PROJECT_DIR/$ADAPTER_DIR
fi

echo ""
echo "=== Stage 4: Adapter shape verification ==="
ls -lh $PROJECT_DIR/$ADAPTER_DIR
test -f $PROJECT_DIR/$ADAPTER_DIR/adapter_config.json || {
    echo "FAIL: adapter_config.json missing — Kaggle would reject this submission."
    exit 1
}
echo "adapter_config.json:"
cat $PROJECT_DIR/$ADAPTER_DIR/adapter_config.json | python3 -m json.tool | head -n 40

echo ""
echo "=== Stage 5: Evaluate at Kaggle eval params (newer LoRA eval vLLM container) ==="
# configs/eval_kaggle.yaml points at outputs/lora_adapter; ensure that
# path resolves to the per-job directory we just produced.
apptainer exec --nv \
  --bind $PROJECT_DIR:/workspace \
  --bind $SCRATCH:$SCRATCH \
  $EVAL_CONTAINER \
  bash -lc "
    cd /workspace
    python3 scripts/baseline_generate.py \
      --config configs/eval_kaggle.yaml \
      --output $EVAL_OUTPUT
  "

echo ""
echo "=== Stage 6: Score ==="
apptainer exec --nv \
  --bind $PROJECT_DIR:/workspace \
  $EVAL_CONTAINER \
  bash -lc "
    cd /workspace
    python3 scripts/evaluate.py --predictions $EVAL_OUTPUT --score
  "

echo ""
echo "=== Stage 7: Package submission ==="
apptainer exec \
  --bind $PROJECT_DIR:/workspace \
  $TRAIN_CONTAINER \
  bash -lc "
    cd /workspace
    python3 scripts/package_submission.py \
      --adapter $ADAPTER_DIR \
      --output  $SUBMISSION_ZIP
  "

echo ""
echo "=== LoRA verification artifacts ==="
ls -lh $PROJECT_DIR/$ADAPTER_DIR
ls -lh $PROJECT_DIR/$EVAL_OUTPUT $PROJECT_DIR/$SUBMISSION_ZIP

echo ""
echo "Finished at: $(date)"
