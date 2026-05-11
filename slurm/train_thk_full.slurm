#!/bin/bash
#SBATCH --job-name=train_thk_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
# TODO (after smoke Stage 4):
#   - Default 08:00:00 = option B walltime.
#   - Measure sec/optimizer-step from logs/smoke_train_<jobid>.out Stage 4
#     (20 steps; take the median of steps 5..19, discarding warmup).
#   - Required time ~= ceil(max_steps_in_config * sec_per_step * 1.4 / 3600)
#     where 1.4 is a SLURM safety factor for warmup, checkpoint write, and
#     transient slowdowns. Round UP to whole hours.
#   - If measurement gives a value >> 8h, switch back to the user on
#     options A vs C from the Phase 6 walltime menu before raising --time.
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ───────────────────────────────────────────────────────────────────────
# DEPRECATED — DO NOT SUBMIT.
#
# This script was designed for path alpha (single uninterrupted run on the
# gpu partition). Smoke job 6710260 measured ~230 sec/optimizer-step on a
# single H200, so a faithful THK run (245 steps) requires ~15.7h. The gpu
# partition's hard TIMELIMIT is 8h, so a single-job approach is infeasible.
#
# Use slurm/train_thk_chunk.slurm instead. That script splits the run into
# 3 chunks of ~82 steps each, persisting optimizer/scheduler/sampler state
# between chunks. See docs/path_beta_runbook.md for the operational sequence.
#
# This file is retained for reference. Submitting it WILL time out before
# the run completes.
# ───────────────────────────────────────────────────────────────────────
#
# Phase 6 production LoRA training run.
#
# Runs the THK 04-10-04-33 recipe (configs/train_thk_full.yaml) over the
# full 17,391-record corpus (data/sft_combined.jsonl). Target: LB ~0.85,
# matching THK's submitted adapter behavior.
#
# This script does TRAINING ONLY. Evaluation, scoring, and submission
# packaging happen in a separate job (slurm/lora_verification.slurm or
# a future slurm/eval_thk.slurm) once we've inspected the loss curve.
#
# Stages:
#   0. Slurm allocation check (host, GPU)
#   1. Prerequisites check (corpus, container, peft, mamba)
#   2. Container GPU passthrough
#   3. Pre-train diagnostics (overflow rate, boundary alignment, sample
#      length histogram). Cheap insurance — re-validating that the config
#      that passed smoke is still consistent with the corpus on disk.
#   4. Full LoRA train (THK recipe, max_steps from config)
#   5. Adapter shape verification (adapter_config.json + safetensors)
#   6. Promote outputs/lora_adapter_thk to a stable symlink at
#      outputs/lora_adapter for downstream eval configs to pick up.
#
# Prerequisites (must already be on the cluster):
#   - data/sft_combined.jsonl (Phase 5)
#   - $SCRATCH/containers/nemotron_vllm.sif
#   - $SCRATCH/lora_pip with peft + accelerate + mamba_ssm + causal_conv1d
#   - configs/smoke_train.yaml smoke run PASSED (i.e. recipe was validated)
#
# Submit:
#   sbatch slurm/train_thk_full.slurm
#   tail -f logs/train_thk_full_<job_id>.out

set -euo pipefail

echo "=== Stage 0: Slurm allocation check ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

PROJECT_DIR=$HOME/Nemotron_Challenge
TRAIN_CONTAINER=$SCRATCH/containers/nemotron_vllm.sif
LORA_PIP_DIR=$SCRATCH/lora_pip
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_thk_full.yaml}

DIAG_JSON=outputs/train_thk_diag_${SLURM_JOB_ID}.json
ADAPTER_DIR=outputs/lora_adapter_thk
ADAPTER_ARCHIVE=outputs/lora_adapter_thk_${SLURM_JOB_ID}
STABLE_ADAPTER_LINK=$PROJECT_DIR/outputs/lora_adapter

export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

mkdir -p $PROJECT_DIR/outputs $PROJECT_DIR/logs

echo ""
echo "=== Stage 1: Prerequisites check ==="
test -f $PROJECT_DIR/data/sft_combined.jsonl || {
    echo "FAIL: data/sft_combined.jsonl missing — re-run Phase 5 (categorize.py + build_sft_traces.py + build_augmenter_traces.py + combine_traces.py) on the local box and git push, then git pull on the cluster."
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
test -d $LORA_PIP_DIR/mamba_ssm || {
    echo "FAIL: mamba_ssm missing from $LORA_PIP_DIR — rebuild per docs/lora_strategy.md §3."
    exit 1
}
test -d $LORA_PIP_DIR/causal_conv1d || {
    echo "FAIL: causal_conv1d missing from $LORA_PIP_DIR — rebuild per docs/lora_strategy.md §3."
    exit 1
}
test -f $PROJECT_DIR/$TRAIN_CONFIG || {
    echo "FAIL: training config missing at $TRAIN_CONFIG"
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
# Same SIGPIPE-safe head pattern as smoke_train.slurm; do NOT pipe into
# json.tool | head — early pipe close trips set -e.
if [ -f "$PROJECT_DIR/$DIAG_JSON" ]; then
    head -n 60 "$PROJECT_DIR/$DIAG_JSON" || true
fi

echo ""
echo "=== Stage 4: Full LoRA train (THK recipe) ==="
# Clean any prior thk adapter so a missing-file at end means a real failure.
# Smoke adapter (outputs/lora_adapter_smoke) is intentionally not touched.
rm -rf $PROJECT_DIR/$ADAPTER_DIR

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
test -d $PROJECT_DIR/$ADAPTER_DIR || {
    echo "FAIL: thk adapter dir missing at $ADAPTER_DIR — training never wrote weights."
    exit 1
}
ls -lh $PROJECT_DIR/$ADAPTER_DIR
test -f $PROJECT_DIR/$ADAPTER_DIR/adapter_config.json || {
    echo "FAIL: adapter_config.json missing."
    exit 1
}
test -f $PROJECT_DIR/$ADAPTER_DIR/adapter_model.safetensors || {
    echo "FAIL: adapter_model.safetensors missing."
    exit 1
}
echo "adapter_config.json:"
cat $PROJECT_DIR/$ADAPTER_DIR/adapter_config.json | python3 -m json.tool

# Archive a copy keyed by job id so reruns are not destructive.
cp -r $PROJECT_DIR/$ADAPTER_DIR $PROJECT_DIR/$ADAPTER_ARCHIVE
echo ""
echo "thk adapter archived at $ADAPTER_ARCHIVE"

echo ""
echo "=== Stage 6: Promote to stable symlink (outputs/lora_adapter) ==="
# configs/eval_kaggle.yaml expects outputs/lora_adapter. Replace any
# pre-existing link/dir there with a fresh symlink to this run's adapter,
# so the next eval slurm picks it up without config edits.
if [ -e $STABLE_ADAPTER_LINK ] || [ -L $STABLE_ADAPTER_LINK ]; then
    rm -rf $STABLE_ADAPTER_LINK
fi
ln -s $PROJECT_DIR/$ADAPTER_DIR $STABLE_ADAPTER_LINK
echo "Stable symlink: $STABLE_ADAPTER_LINK -> $PROJECT_DIR/$ADAPTER_DIR"
ls -lh $STABLE_ADAPTER_LINK

echo ""
echo "=== Phase 6 production train COMPLETE ==="
echo "Diagnostic summary:        $DIAG_JSON"
echo "thk adapter (live):        $ADAPTER_DIR"
echo "thk adapter (archived):    $ADAPTER_ARCHIVE"
echo "Stable symlink:            outputs/lora_adapter -> $ADAPTER_DIR"
echo ""
echo "Next: inspect loss curve in logs/train_thk_full_${SLURM_JOB_ID}.out,"
echo "      then run slurm/lora_verification.slurm (or the dedicated"
echo "      stratified-holdout eval job) before any Kaggle submission."
echo ""
echo "Finished at: $(date)"
