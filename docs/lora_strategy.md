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
- Nemotron-3 Nano is a hybrid Mamba-2/MoE custom architecture loaded via `trust_remote_code=True`. PEFT's `target_modules` is set to the attention-only list `[q_proj, k_proj, v_proj, o_proj]` — see §3 "vLLM MoE LoRA constraint" for why "all-linear" can't be used here.
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
| `target_modules` | `[q_proj, k_proj, v_proj, o_proj]` | attention-only; expert/Mamba layers excluded — see §3 "vLLM MoE LoRA constraint" |
| `dropout` | 0.0 | small dataset, no need |
| `max_seq_len` | 4096 | half the eval context — eases activation memory on a single H200 during backward |
| `epochs` | 1 | verification, not tuning |
| `batch × grad_accum` | 1 × 16 | effective batch 16, fits H200 with checkpointing |
| `lr` | 2e-4 | typical LoRA sweet spot |
| `gradient_checkpointing` | true | required to fit 30B BF16 + activations |

These are sized for ~200 samples × 1 epoch ≈ 12 optimisation steps. The job exists to verify the artifact, not to win the leaderboard.

## 3. Setup (one-time)

We use the existing vLLM container for both training and eval. Required scratch-resident packages: `peft`, `accelerate`, `mamba-ssm`, `causal-conv1d`. We install all four into a `$SCRATCH`-resident path and `PYTHONPATH`-inject them. This sidesteps a multi-hour container pull and saves ~30 GB of `$SCRATCH` versus pulling a separate NeMo image.

Do not switch to a NeMo container just because `mamba-ssm` fails to import. A failure like:

```text
selective_scan_cuda...so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...
```

means the scratch-installed `mamba-ssm` extension was compiled against a different PyTorch C++ ABI than the vLLM container's torch. The fix is to rebuild it inside the container — see "Mamba-2 SSM extension" below. **There is no PyTorch fallback for this model.** NVIDIA's `modeling_nemotron_h.py` (loaded via `trust_remote_code=True`) does an unconditional `from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn` at module load and re-raises `ImportError` if it fails. Earlier guidance in this section claimed a "pure-PyTorch Mamba path" was usable for verification; that is true for transformers' built-in `MambaModel`, but NOT for the Nemotron-H custom code path.

### vLLM MoE LoRA constraint

vLLM 0.12.0's built-in `NemotronHForCausalLM` (the in-engine model class, not the `trust_remote_code` modeling file) does not implement `get_expert_mapping`. When vLLM tries to load a LoRA adapter that touches MoE expert weights, `process_packed_modules_mapping` raises:

```text
AttributeError: To support LoRA for MoE model, 'get_expert_mapping' must be implemented
```

Job 6516064 hit this. Training (Stage 3) succeeded with `target_modules="all-linear"` and produced an adapter where 11,868 of 12,008 LoRA tensors (≈ 98.8%) targeted `mixer.experts.<idx>.{down_proj,up_proj}`. vLLM rejected the load before the eval could run.

**This is a Kaggle constraint, not just a local-eval one.** Kaggle scores via vLLM (same engine, same model). Until upstream vLLM ships `get_expert_mapping` for Nemotron-H, any adapter we submit must avoid expert layers.

Practical implication for `target_modules`:

- **Verification** (current): attention-only, explicit list `[q_proj, k_proj, v_proj, o_proj]`. ~48 LoRA tensors. Definitely loads.
- **Possible follow-up:** `target_modules: all-linear` + `exclude_modules: "experts"` to add Mamba `mixer.in_proj`/`mixer.out_proj` LoRA on top of attention. Adapter would grow to ~140 tensors. Untested whether vLLM's NemotronHForCausalLM supports LoRA on Mamba mixer modules — try after verification turns green.
- **Don't:** target Mamba/MoE via "all-linear" without excluding experts. Experts dominate the layer count and trigger the failure above.

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

### Mamba-2 SSM extension (required for verification too)

`mamba-ssm` and `causal-conv1d` must be present in `$SCRATCH/lora_pip` and built against the vLLM container's torch. They are not optional — the Nemotron-H custom modeling code imports them unconditionally (see §3 intro). The slurm verification script now hard-fails in Stage 1 if either directory is missing.

The container's bundled torch is `2.9.0+cu129` (verified 2026-05-02 with `--nv` + `PYTHONNOUSERSITE=1`). Earlier "torch 2.10.0+cu128" claims in this repo's notes referred to `~/.local`'s user-site torch, which `PYTHONNOUSERSITE=1` deliberately excludes — that ambiguity was the root cause of the original ABI crash, because the previous build picked up the user-site torch while the slurm runtime used the container torch.

Build (or rebuild after an ABI mismatch) inside the same vLLM container so the kernels link against that container's torch. **`PYTHONNOUSERSITE=1` must be set during the build**, otherwise `~/.local` shadows the container's torch and you'll rebuild against the wrong ABI:

```bash
apptainer exec --bind $SCRATCH:$SCRATCH $SCRATCH/containers/nemotron_vllm.sif \
  bash -lc "
    export PYTHONNOUSERSITE=1
    rm -rf $SCRATCH/lora_pip/mamba_ssm* $SCRATCH/lora_pip/causal_conv1d* \
           $SCRATCH/lora_pip/selective_scan_cuda*.so
    python3 -c 'import torch; print(\"build will link against torch\", torch.__version__)'
    MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    pip install --target=$SCRATCH/lora_pip --no-build-isolation --no-deps --no-cache-dir \
      causal-conv1d mamba-ssm
  "
```

Notes:

- `PYTHONNOUSERSITE=1` matches the slurm runtime, so the build's torch and the runtime's torch are the same module. This is the one piece the previous (failed) build was missing.
- `--no-build-isolation` keeps the build tied to that torch instead of compiling against a transient build-environment torch — this is what prevents the `_ZN3c104cuda...` ABI crash at import time.
- `--no-deps` prevents pip from upgrading the container's torch.
- Versions are intentionally unpinned because torch 2.9 is recent (Oct 2025) and the matching mamba-ssm release moves quickly. If a specific pin is desired, pick whatever pip resolves on first install and record it in the README.
- No `--nv` flag is needed for the build — it invokes `nvcc` for compilation but does not launch CUDA kernels, so it does not need a GPU. Run on a CPU compute node with ≥ 32 GB RAM (login-node `/tmp` is too small and triggers OOM during the link step).
- Build time is 15–30 min on 8 CPUs.

After the rebuild, verify the import. **`PYTHONNOUSERSITE=1` must be set here too** for the same reason, otherwise the verify shell loads `~/.local`'s torch and reports a misleading version:

```bash
apptainer exec --bind $SCRATCH:$SCRATCH $SCRATCH/containers/nemotron_vllm.sif \
  bash -lc "
    export PYTHONPATH=$SCRATCH/lora_pip:\$PYTHONPATH
    export PYTHONNOUSERSITE=1
    python3 -c 'import torch, mamba_ssm, causal_conv1d; print(torch.__version__, mamba_ssm.__version__, causal_conv1d.__version__)'
  "
```

The first column must equal the torch version printed inside the build shell. If it doesn't, the build environment was different from the runtime environment — most often because `PYTHONNOUSERSITE=1` was missing on one side, or because a torch dir was added/removed from `$SCRATCH/lora_pip` between build and verify. Mismatch ⇒ rebuild.

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
| 1. Prerequisites | both JSONLs present, vLLM container present |
| 2. Container GPU | `nvidia-smi` works inside the vLLM container |
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

- **Job A — `train.slurm`** allocates one H200, runs the vLLM container with the scratch HF/PEFT deps, trains, saves the adapter to a job-scoped directory (or a `outputs/lora_adapter` symlink). Walltime ~1-2h.
- **Job B — `eval.slurm`** allocates one H200, runs the vLLM container, loads the adapter Job A produced, runs `baseline_generate.py --config configs/eval_kaggle.yaml`, scores, packages. Walltime ~1h.

Job B can be submitted with `sbatch --dependency=afterok:<job_a_id> slurm/eval.slurm` so it queues immediately and starts the moment training finishes successfully.

**Why we want this once the verification pass is green:**

- Tighter retry surface — eval crashes don't re-cost an hour of training.
- Job B can be re-run repeatedly against the same adapter to A/B `enable_thinking=True` vs `False`, swap eval datasets, or sweep eval-time decoding params, with no retraining cost.
- Shorter walltime requests get backfilled faster on Slurm; the bring-up log already noted 2h gaps are far more common than 4h ones.

**Natural follow-up: training checkpoints.** A real training run on 5K+ samples at higher rank exceeds any single Slurm walltime. Adding "save full state (adapter + AdamW optimizer state + LR scheduler step) every N optim steps to `$SCRATCH/checkpoints/...`, resume from latest on startup" to `train_lora.py` makes it possible to chain multiple `train.slurm` jobs across days. Eval can also be made resume-friendly cheaply (skip prompt IDs already present in the output JSONL).

These three changes — split, training checkpoints, eval resume — are the right next infrastructure pass after the verification milestone passes.

sacct -j 6484958 --format=JobID,State,Elapsed,ExitCode
Job ID(2026/5/1):6459431
tail -n 50 logs/lora_verification_6516064.out

tail -n 50 logs/lora_verification_6516064.err

Job ID new(2026/5/1):6484958
