# LoRA Strategy

This document explains the design choices behind the `LoRA_verification`
pipeline. Code lives in:

- [configs/lora_verification.yaml](../configs/lora_verification.yaml) — training config
- [configs/eval_kaggle.yaml](../configs/eval_kaggle.yaml) — eval config mirroring the Kaggle leaderboard
- [scripts/prepare_reasoning_traces.py](../scripts/prepare_reasoning_traces.py) — data prep
- [scripts/train_lora.py](../scripts/train_lora.py) — LoRA SFT trainer
- [scripts/baseline_generate.py](../scripts/baseline_generate.py) — vLLM inference, now LoRA-capable
- [scripts/package_submission.py](../scripts/package_submission.py) — submission zip
- [slurm/lora_verification.slurm](../slurm/lora_verification.slurm) — orchestration

## 1. The Kaggle constraint shapes everything

The leaderboard runs the user's adapter on top of `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` via vLLM with these fixed parameters:

| param | value |
|---|---|
| `max_lora_rank` | 32 |
| `max_tokens` | 7680 |
| `max_model_len` | 8192 |
| `temperature` | 0.0 (greedy) |
| `top_p` | 1.0 |
| `max_num_seqs` | 64 |
| `gpu_memory_utilization` | 0.85 |

Final answers are extracted from `\boxed{...}` (with fallbacks) and graded by exact-match or relative numerical tolerance.

This is **the inverse** of what the model card was tuned for. The card recommends `temperature=1.0` and budgets reasoning traces up to ~30K tokens. The April 28 baseline run in the README already showed 17/30 AIME25 traces truncating mid-final-answer at 24K tokens — at 7680 tokens with greedy decoding, naive thinking-mode generation will repeat or run past the cap on hard problems.

The LoRA's job is therefore to **shift the response distribution toward concise, decisive reasoning that ends in a balanced `\boxed{}` within 7680 greedy-decoded tokens**, without giving up too much accuracy on problems the base already handles well.

## 2. Strategy choices

### Stack: HF transformers + peft, not pure NeMo CLI

NeMo Automodel is the official NVIDIA path for LoRA on Nemotron-3 Nano, but Automodel itself is a wrapper over HF transformers + peft + accelerate. We call those libraries directly because:

- The YAML schema for NeMo Automodel changes between releases; calling HF APIs directly is more durable.
- Nemotron-3 Nano is a hybrid Mamba-2/MoE custom architecture loaded via `trust_remote_code=True`. PEFT's `target_modules="all-linear"` discovers LoRA target layers programmatically, which avoids hard-coding `q_proj`/`k_proj`/`v_proj`/`o_proj` and silently missing MoE expert / Mamba projection layers.
- HF PEFT writes `adapter_config.json` + `adapter_model.safetensors` in the exact shape Kaggle expects.

Both NeMo Automodel and our direct path produce equivalent adapter artifacts. If NeMo lands first-class Nemotron-3 Nano LoRA recipes ([NeMo issue #14856](https://github.com/NVIDIA-NeMo/NeMo/issues/14856)) we can swap in.

### Data: NVIDIA OpenMathReasoning (CoT split), aggressively length-filtered

`nvidia/OpenMathReasoning` is NVIDIA's own curated math reasoning corpus. Picking it for fine-tuning a Nemotron model minimises distribution shift / catastrophic forgetting risk.

Filter rules in `prepare_reasoning_traces.py`:

- `inference_mode == "cot"` — keep chain-of-thought, skip tool-integrated reasoning (Kaggle has no tools).
- `pass_rate_1 == 1.0` — keep only traces the teacher model solved correctly (clean labels).
- `len(generated_solution) <= 14000` chars (~ 4-4.5K tokens) — strict cap so traces fit comfortably under the 7680-token eval budget with headroom.
- Final answer must contain `\\boxed{`.

This biases the model toward shorter, correct traces. The baseline trace length on AIME25 is ~67K chars / ~17K tokens; filtered training data sits around a third of that. That distribution shift is intentional.

### Format alignment: assistant-only loss, chat-template rendered

Each training sample is a two-turn chat:

```json
{"messages": [
  {"role": "user", "content": "Solve the following math problem... \\boxed{}\n\n<problem>"},
  {"role": "assistant", "content": "<full reasoning trace ending with \\boxed{N}>"}
]}
```

The user message uses the same NeMo-Skills generic boxed-answer template our AIME25 baseline already uses, so train-time and eval-time prompts share the same structural shape.

`train_lora.py` renders the messages via `tokenizer.apply_chat_template`, then masks loss on the user-prompt tokens (`label = -100`) so gradients only flow on the assistant turn. This matches how the model is used at inference — it never has to predict its own input.

Important consequence: **whatever `enable_thinking` default Kaggle's eval uses, we want to be agnostic to it.** Since the assistant trace already contains the full reasoning text, the model learns to produce that format regardless of whether the chat template inserts a `<think>` prefix at inference time. We can later A/B both inference-time settings against the same adapter.

### Hyperparameters: deliberately small for verification

| param | value | reason |
|---|---|---|
| `r` (LoRA rank) | 16 | half the Kaggle cap of 32, leaves room to grow |
| `alpha` | 32 | conventional `alpha = 2 × rank` |
| `target_modules` | `"all-linear"` | robust to the hybrid arch |
| `dropout` | 0.0 | small dataset, no need |
| `max_seq_len` | 4096 | half the eval context — eases activation memory on a single H200 during backward |
| `epochs` | 1 | verification, not tuning |
| `batch × grad_accum` | 1 × 16 | effective batch 16, fits H200 with checkpointing |
| `lr` | 2e-4 | typical LoRA sweet spot |
| `gradient_checkpointing` | true | required to fit 30B BF16 + activations |

These are sized for ~200 samples × 1 epoch ≈ 12 optimisation steps. The job exists to verify the artifact, not to win the leaderboard.

## 3. Setup (one-time)

We use the existing vLLM container for both training and eval. The only thing missing for training is `peft` (and `accelerate`); we install those into a `$SCRATCH`-resident path and `PYTHONPATH`-inject them. This sidesteps a multi-hour container pull and saves ~30 GB of `$SCRATCH` versus pulling a separate NeMo image.

### vLLM container — already in place

`$SCRATCH/containers/nemotron_vllm.sif` (`vllm/vllm-openai:v0.12.0`) was pulled during the smoke-test bring-up. No changes needed.

### peft + accelerate into `$SCRATCH/lora_pip`

Run this once, from a CPU allocation on the `short` partition. `salloc` on Explorer does NOT auto-shell into the compute node — you have to `srun --pty bash` after it:

```bash
salloc --partition=short --cpus-per-task=4 --mem=16G --time=00:30:00
srun --jobid=$SLURM_JOB_ID --pty bash    # actually enter the compute node

mkdir -p $SCRATCH/lora_pip
apptainer exec \
  --bind $SCRATCH:$SCRATCH \
  $SCRATCH/containers/nemotron_vllm.sif \
  bash -lc "pip install --target=$SCRATCH/lora_pip peft accelerate"
```

`pip install --target=...` writes packages to a directory you control rather than into the container's site-packages or `~/.local`. This bypasses `PYTHONNOUSERSITE=1` cleanly because we activate the path explicitly via `PYTHONPATH` instead of relying on user-site discovery.

### Verify the install

Still inside the compute-node shell:

```bash
apptainer exec \
  --bind $SCRATCH:$SCRATCH \
  $SCRATCH/containers/nemotron_vllm.sif \
  bash -lc "
    export PYTHONPATH=$SCRATCH/lora_pip:\$PYTHONPATH
    python3 -c 'import torch, transformers, peft, accelerate;
print(\"torch\", torch.__version__);
print(\"transformers\", transformers.__version__);
print(\"peft\", peft.__version__);
print(\"accelerate\", accelerate.__version__)'
  "
```

If all four versions print, you're done. The slurm verification script will inject the same `PYTHONPATH` automatically — no further setup needed.

### Why this is safe versus the container's own packages

`PYTHONPATH` is searched *between* the script directory and the container's default `site-packages`, so the scratch versions of `peft` and `accelerate` win, but `torch` and `transformers` (which we did NOT install to scratch) come from the container as before. There is no version mix between scratch and container that we depend on at runtime — `torch` and `transformers` are the ones that matter for CUDA compatibility, and we leave both untouched.

## 4. How to run the LoRA verification

On the login node:

```bash
cd $HOME/Nemotron_Challenge
git pull

python3 scripts/prepare_reasoning_traces.py --num 200      # produces data/lora_traces.jsonl
python3 scripts/prepare_aime25.py                          # if not already present
sbatch slurm/lora_verification.slurm
```

The slurm script runs Stages 0-7. Pass criteria:

| Stage | Check |
|---|---|
| 1. Prerequisites | both JSONLs present, both containers present |
| 2. Container GPU | `nvidia-smi` works inside both |
| 3. Train | `train_lora.py` exits 0; adapter dir created |
| 4. Adapter shape | `adapter_config.json` exists, content looks sane |
| 5. Eval | `outputs/lora_eval_<JOB_ID>.jsonl` written, no run-wide error |
| 6. Score | `evaluate.py --score` prints an accuracy number |
| 7. Package | `submission_<JOB_ID>.zip` exists, contains `adapter_config.json` + weights |

The accuracy number from Stage 6 is **not** a quality benchmark on this pass — it tells us "the pipeline produced a usable adapter that vLLM can load and that scores against AIME25 without errors." Quality work begins after Stage 7 returns success.

## 5. What this verification does NOT prove

- That the chosen training data is the right distribution for the actual Kaggle test set. The competition page says "novel benchmark developed by NVIDIA Research." We are using AIME25 as the local proxy.
- That `enable_thinking=True` vs `False` at inference is the right call. The eval YAML currently keeps `enable_thinking=True` (matches `baseline_generate.py`'s default); a follow-up should A/B both modes against the same adapter.
- That rank 16 / 200 samples / 1 epoch is the right hyperparameter shape. It is a starting point.
- That HF PEFT correctly hooks all the right layers in Nemotron-3's hybrid arch. `print_trainable_parameters()` output during training will tell us the trainable-parameter count; a value implausibly small (e.g. < 0.05% of total) would indicate target-module discovery missed the bulk of the model.

These are explicit follow-ups, not gaps in the verification.

## 6. After verification passes

1. Re-run `prepare_reasoning_traces.py --num 5000` (or larger) for a real training run.
2. Bump `r` to 24 or 32 (the Kaggle cap), increase epochs.
3. A/B `enable_thinking=True` vs `False` at eval time using the same adapter.
4. Add a held-out dev split (e.g. MATH-500 or GPQA-Diamond) so improvement signals are not dominated by AIME25's 30-problem variance.
5. Consider on-policy distillation from a stronger teacher (DeepSeek-R1, QwQ-32B) for the next iteration — research shows ~7-10× faster convergence than RL with comparable quality at this rank range.

## 7. Future direction: split into two sequential cluster jobs

The current `slurm/lora_verification.slurm` is monolithic — train + eval + score + package in one allocation. That's right for the verification milestone but starts to creak once we move to real training. Future shape:

- **Job A — `train.slurm`** allocates one H200, runs the NeMo container, trains, saves the adapter to a job-scoped directory (or a `outputs/lora_adapter` symlink). Walltime ~1-2h.
- **Job B — `eval.slurm`** allocates one H200, runs the vLLM container, loads the adapter Job A produced, runs `baseline_generate.py --config configs/eval_kaggle.yaml`, scores, packages. Walltime ~1h.

Job B can be submitted with `sbatch --dependency=afterok:<job_a_id> slurm/eval.slurm` so it queues immediately and starts the moment training finishes successfully.

**Why we want this once the verification pass is green:**

- Tighter retry surface — eval crashes don't re-cost an hour of training.
- Job B can be re-run repeatedly against the same adapter to A/B `enable_thinking=True` vs `False`, swap eval datasets, or sweep eval-time decoding params, with no retraining cost.
- Shorter walltime requests get backfilled faster on Slurm; the bring-up log already noted 2h gaps are far more common than 4h ones.

**Natural follow-up: training checkpoints.** A real training run on 5K+ samples at higher rank exceeds any single Slurm walltime. Adding "save full state (adapter + AdamW optimizer state + LR scheduler step) every N optim steps to `$SCRATCH/checkpoints/...`, resume from latest on startup" to `train_lora.py` makes it possible to chain multiple `train.slurm` jobs across days. Eval can also be made resume-friendly cheaply (skip prompt IDs already present in the output JSONL).

These three changes — split, training checkpoints, eval resume — are the right next infrastructure pass after the verification milestone passes.
