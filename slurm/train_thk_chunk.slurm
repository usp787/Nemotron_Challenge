#!/bin/bash
#SBATCH --job-name=train_thk_chunk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=07:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Path-beta production LoRA training — ONE CHUNK of three.
#
# Smoke job 6710260 measured ~230 sec/optimizer-step on a single H200.
# A faithful THK run (245 steps) would take ~15.7h — over the gpu
# partition's 8h walltime cap. This script implements path beta by
# splitting the run into 3 chunks of ~82 steps each, with the
# intermediate state (AdamW momentum, LR schedule position, sampler
# position in the deterministic shuffle) persisted between chunks via
# train_lora.py's training_state.pt mechanism.
#
# Each chunk has 7h of SLURM walltime (= ~5.5h expected + ~1.5h safety
# margin). The diagnostic + setup overhead is paid once per chunk.
#
# OPERATIONAL SEQUENCE (do NOT chain via --dependency):
#   sbatch --export=ALL,THK_CHUNK_IDX=1 slurm/train_thk_chunk.slurm
#   <wait for COMPLETED, inspect log + outputs/lora_adapter_thk_chunk1/>
#   sbatch --export=ALL,THK_CHUNK_IDX=2 slurm/train_thk_chunk.slurm
#   <wait, inspect>
#   sbatch --export=ALL,THK_CHUNK_IDX=3 slurm/train_thk_chunk.slurm
#   <wait, inspect — final chunk also promotes outputs/lora_adapter symlink>
#
# Full runbook at docs/path_beta_runbook.md.

set -euo pipefail

# Validate the chunk index. THK_CHUNK_IDX must be 1, 2, or 3.
THK_CHUNK_IDX=${THK_CHUNK_IDX:?THK_CHUNK_IDX must be set (1/2/3). Submit with: sbatch --export=ALL,THK_CHUNK_IDX=<n> slurm/train_thk_chunk.slurm}
case "$THK_CHUNK_IDX" in
  1|2|3) ;;
  *) echo "FAIL: THK_CHUNK_IDX must be 1, 2, or 3 (got: $THK_CHUNK_IDX)"; exit 1 ;;
esac

# Per-chunk cumulative max_steps target. Each value is the absolute
# global_step the chunk should reach (NOT a delta). Chunk 2 effectively
# performs steps 83..164 because chunk 1 already advanced global_step to 82.
case "$THK_CHUNK_IDX" in
  1) THIS_MAX_STEPS=82 ;;
  2) THIS_MAX_STEPS=164 ;;
  3) THIS_MAX_STEPS=246 ;;
esac

echo "=== Stage 0: Slurm allocation check ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Chunk: $THK_CHUNK_IDX / 3 (cumulative max_steps target: $THIS_MAX_STEPS)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

PROJECT_DIR=$HOME/Nemotron_Challenge
TRAIN_CONTAINER=$SCRATCH/containers/nemotron_vllm.sif
LORA_PIP_DIR=$SCRATCH/lora_pip
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_thk_full.yaml}

# Output dir for THIS chunk (adapter + training_state.pt)
THIS_OUTPUT_DIR=outputs/lora_adapter_thk_chunk${THK_CHUNK_IDX}
THIS_ARCHIVE_DIR=outputs/lora_adapter_thk_chunk${THK_CHUNK_IDX}_${SLURM_JOB_ID}

# Path-beta resume: chunks 2 and 3 read the previous chunk's training state
RESUME_FROM=""
if [ "$THK_CHUNK_IDX" -gt 1 ]; then
    PREV_CHUNK=$((THK_CHUNK_IDX - 1))
    RESUME_FROM=outputs/lora_adapter_thk_chunk${PREV_CHUNK}
    if [ ! -d "$PROJECT_DIR/$RESUME_FROM" ]; then
        echo "FAIL: chunk $THK_CHUNK_IDX needs prior chunk dir at $RESUME_FROM"
        echo "      but it does not exist. Re-run chunk $PREV_CHUNK first."
        exit 1
    fi
    if [ ! -f "$PROJECT_DIR/$RESUME_FROM/training_state.pt" ]; then
        echo "FAIL: previous chunk's training_state.pt missing at $RESUME_FROM/training_state.pt"
        echo "      The previous chunk likely failed mid-training."
        echo "      Re-run chunk $PREV_CHUNK before submitting this one."
        exit 1
    fi
    if [ ! -f "$PROJECT_DIR/$RESUME_FROM/adapter_config.json" ]; then
        echo "FAIL: previous chunk's adapter_config.json missing at $RESUME_FROM"
        exit 1
    fi
fi

DIAG_JSON=outputs/train_thk_chunk${THK_CHUNK_IDX}_diag_${SLURM_JOB_ID}.json
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
    echo "FAIL: data/sft_combined.jsonl missing — Phase 5 has not been pushed/pulled to the cluster."
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
# Re-run on every chunk: cheap (~2-3 min) and catches any corpus drift.
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
if [ -f "$PROJECT_DIR/$DIAG_JSON" ]; then
    head -n 60 "$PROJECT_DIR/$DIAG_JSON" || true
fi

echo ""
echo "=== Stage 4: LoRA train chunk $THK_CHUNK_IDX / 3 ==="
echo "  cumulative max_steps target: $THIS_MAX_STEPS"
echo "  output dir: $THIS_OUTPUT_DIR"
if [ -n "$RESUME_FROM" ]; then
    echo "  resuming from: $RESUME_FROM"
else
    echo "  starting fresh (no resume)"
fi

# Build the per-chunk CLI flags
RESUME_FLAG=""
if [ -n "$RESUME_FROM" ]; then
    RESUME_FLAG="--resume-from-checkpoint $RESUME_FROM"
fi

# Clear any stale dir for THIS chunk so a missing file at the end means a
# real failure, not leftover output from a previous attempt.
rm -rf $PROJECT_DIR/$THIS_OUTPUT_DIR

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
    python3 scripts/train_lora.py \
      --config $TRAIN_CONFIG \
      --max-steps $THIS_MAX_STEPS \
      --output-dir $THIS_OUTPUT_DIR \
      $RESUME_FLAG
  "

echo ""
echo "=== Stage 5: Adapter + training-state shape verification ==="
test -d $PROJECT_DIR/$THIS_OUTPUT_DIR || {
    echo "FAIL: chunk output dir missing at $THIS_OUTPUT_DIR — training did not save."
    exit 1
}
ls -lh $PROJECT_DIR/$THIS_OUTPUT_DIR
test -f $PROJECT_DIR/$THIS_OUTPUT_DIR/adapter_config.json || {
    echo "FAIL: adapter_config.json missing."
    exit 1
}
test -f $PROJECT_DIR/$THIS_OUTPUT_DIR/adapter_model.safetensors || {
    echo "FAIL: adapter_model.safetensors missing."
    exit 1
}
test -f $PROJECT_DIR/$THIS_OUTPUT_DIR/training_state.pt || {
    echo "FAIL: training_state.pt missing — the next chunk will not be able to resume."
    exit 1
}
echo "adapter_config.json:"
cat $PROJECT_DIR/$THIS_OUTPUT_DIR/adapter_config.json | python3 -m json.tool

# Archive a copy keyed by job_id so reruns don't clobber the canonical
# chunk dir that the next chunk's resume reads from.
cp -r $PROJECT_DIR/$THIS_OUTPUT_DIR $PROJECT_DIR/$THIS_ARCHIVE_DIR
echo ""
echo "Chunk $THK_CHUNK_IDX adapter archived at $THIS_ARCHIVE_DIR"

# Stage 6 (final chunk only): promote outputs/lora_adapter symlink so
# configs/eval_kaggle.yaml picks up the trained adapter without edits.
if [ "$THK_CHUNK_IDX" = "3" ]; then
    echo ""
    echo "=== Stage 6: Final chunk — promoting outputs/lora_adapter symlink ==="
    if [ -e $STABLE_ADAPTER_LINK ] || [ -L $STABLE_ADAPTER_LINK ]; then
        rm -rf $STABLE_ADAPTER_LINK
    fi
    ln -s $PROJECT_DIR/$THIS_OUTPUT_DIR $STABLE_ADAPTER_LINK
    echo "Stable symlink: $STABLE_ADAPTER_LINK -> $PROJECT_DIR/$THIS_OUTPUT_DIR"
    ls -lh $STABLE_ADAPTER_LINK
fi

echo ""
echo "=== Chunk $THK_CHUNK_IDX COMPLETE ==="
echo "Diagnostic summary:       $DIAG_JSON"
echo "Chunk adapter (live):     $THIS_OUTPUT_DIR"
echo "Chunk adapter (archived): $THIS_ARCHIVE_DIR"
if [ "$THK_CHUNK_IDX" != "3" ]; then
    NEXT_CHUNK=$((THK_CHUNK_IDX + 1))
    echo ""
    echo "Next: submit chunk $NEXT_CHUNK with"
    echo "  sbatch --export=ALL,THK_CHUNK_IDX=$NEXT_CHUNK slurm/train_thk_chunk.slurm"
    echo "after inspecting logs/train_thk_chunk_${SLURM_JOB_ID}.out."
else
    echo ""
    echo "All 3 chunks done. Adapter is at outputs/lora_adapter -> $THIS_OUTPUT_DIR"
    echo "Next: inspect loss curve end-to-end, then run your eval pipeline."
fi
echo ""
echo "Finished at: $(date)"
