# NVIDIA Nemotron Model Reasoning Challenge

This repository contains an HPC-oriented baseline workflow for the **NVIDIA Nemotron Model Reasoning Challenge** on Kaggle. The current target model is:

```text
nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

The repo is designed for a university HPC environment where:

- Jupyter Notebook is **not** required for the main run.
- Remote SSH may be unstable.
- Open OnDemand (OOD) is the main access portal.
- Docker is unavailable or unsupported.
- Apptainer/Singularity is available.
- Jobs are launched through Slurm on an H200 GPU node.

---

## 1. Project Overview

### 1.1 Challenge Context

The NVIDIA Nemotron Model Reasoning Challenge is a Kaggle competition centered on improving or evaluating model reasoning behavior. The competition description frames reasoning benchmarks as a way to measure progress in language model reasoning ability, with the practical goal of producing better reasoning outputs under the competition's evaluation setup.

This repository focuses on building a **reproducible baseline run** before attempting optimization strategies such as prompt engineering, reasoning-mode control, tool use, self-consistency, fine-tuning, LoRA, or RL-style methods.

> Note: The Kaggle competition page may require browser login and may not expose all description/evaluation text through simple command-line or scraper access. Always verify the latest official rules, dataset files, submission format, and evaluation metric directly on the Kaggle competition page before final submission.

Competition page:

```text
https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
```

---

### 1.2 Target Model

`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` is a hybrid Mamba-2/MoE BF16 model (30B total / ~3.5B active params) trained for both reasoning and non-reasoning tasks; runs on a single H200 in this repo. Model card: <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16>

---

## 2. Repository Goal

This repository is **not** initially focused on post-training.

The first goal is:

```text
Build a stable, reproducible, script-based baseline run for Nemotron 3 Nano on HPC.
```

The baseline should answer:

1. Can the H200 GPU be allocated successfully?
2. Can the Apptainer container access the GPU?
3. Can vLLM / Transformers import correctly inside the container?
4. Can the model be downloaded or loaded from cache?
5. Can a small set of prompts run end-to-end?
6. Can outputs be saved in JSONL format?
7. Can latency, errors, and basic generation metadata be recorded?
8. Can the same run be repeated by another team member?

---

## 3. Workflow Summary

The current recommended workflow is:

```text
Local laptop
  └── edit code with VS Code
  └── commit and push to GitHub
        |
        v
Open OnDemand portal
  └── open HPC terminal
  └── git pull latest code
  └── submit Slurm job
        |
        v
HPC compute node
  └── Slurm allocates H200 GPU
  └── Apptainer provides runtime environment
  └── Python/vLLM runs baseline
        |
        v
HPC storage
  └── logs/
  └── outputs/
  └── model cache under $SCRATCH
```

---

## 4. Repository Structure

The on-disk layout is `configs/`, `data/`, `scripts/`, `slurm/`, `docs/`, `logs/`, `outputs/`. See [.gitignore](.gitignore) for what is excluded from git.

---

## 5. Storage Policy

Keep code and small config files in GitHub.

Do **not** commit:

- Model weights
- Hugging Face cache
- Apptainer `.sif` files
- Large datasets
- Large JSONL outputs
- Long Slurm logs
- API keys or tokens

Recommended split:

```text
$HOME/Nemotron_Challenge/          # code repo
$SCRATCH/containers/               # Apptainer images
$SCRATCH/huggingface/              # HF cache
$SCRATCH/nemotron_outputs/         # large outputs
$SCRATCH/datasets/                 # downloaded datasets
```

---

## 6. Setup on HPC

### 6.1 Clone the Repository

From the OOD terminal:

```bash
cd $HOME
git clone git@github.com:usp787/Nemotron_Challenge.git
cd Nemotron_Challenge
mkdir -p logs outputs
```

If SSH authentication is not configured, use HTTPS or set up an HPC SSH key for GitHub.

---

### 6.2 Set Hugging Face Cache

Large model files should go to scratch storage:

```bash
export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub
```

Optional: add these exports to the Slurm scripts rather than `.bashrc` so the run is self-contained.

---

### 6.3 Pull the vLLM Container

Expected container path (not committed to GitHub):

```text
$SCRATCH/containers/nemotron_vllm.sif
```

We use `vllm/vllm-openai:v0.12.0`, pulled directly from Docker Hub via Apptainer:

```bash
mkdir -p $SCRATCH/containers
cd $SCRATCH/containers

apptainer pull nemotron_vllm.sif docker://vllm/vllm-openai:v0.12.0
```

> Pull from a CPU allocation on the `short` partition rather than the login node — the login node's `/tmp` is too small for `mksquashfs`. See [docs/hpc_setup_notes.md](docs/hpc_setup_notes.md) for recorded provenance.

Verify GPU passthrough and Python/vLLM imports:

```bash
apptainer exec --nv $SCRATCH/containers/nemotron_vllm.sif nvidia-smi
apptainer exec --nv $SCRATCH/containers/nemotron_vllm.sif bash -lc "
python3 --version
python3 -c 'import vllm; print(\"vLLM import OK\")'
"
```

The Slurm scripts reference the container through one variable:

```bash
CONTAINER=$SCRATCH/containers/nemotron_vllm.sif
```

---

## 7. Smoke Test

Driver scripts: [scripts/check_env.py](scripts/check_env.py), [scripts/smoke_test.py](scripts/smoke_test.py), [slurm/smoke_test.slurm](slurm/smoke_test.slurm).

Submit and inspect:

```bash
sbatch slurm/smoke_test.slurm
squeue -u $USER
cat logs/nemotron_smoke_<JOB_ID>.out
cat logs/nemotron_smoke_<JOB_ID>.err
```

### 7.1 Stages

The smoke test verifies the full path from Slurm allocation through container runtime to model-level generation, so failures are easy to localize. Stages 0–1 run host-side in `slurm/smoke_test.slurm`; stages 2–5 run inside the container via `scripts/smoke_test.py`; stage 6 reuses `scripts/baseline_generate.py` with `configs/smoke_h200.yaml`.

| Stage | Checks |
|---|---|
| 0. Slurm allocation | hostname, `$SLURM_JOB_ID`, `$CUDA_VISIBLE_DEVICES`, `nvidia-smi` |
| 1. Apptainer GPU passthrough | `apptainer exec --nv $CONTAINER nvidia-smi` |
| 2. Python package check | `import torch / transformers / vllm` inside the container |
| 3. Hugging Face metadata access | `huggingface_hub.model_info(...)` for the Nemotron model id |
| 4. Tokenizer + chat template | `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` + `apply_chat_template` |
| 5. Single-prompt model load and generation | small `max_model_len`, `max_tokens=64`, `temperature=0.0`, `gpu_memory_utilization=0.80` |
| 6. Five-prompt mini baseline | `baseline_generate.py` + `configs/smoke_h200.yaml` against `data/sample_prompts_5.jsonl` |

### 7.2 Pass/fail criteria

A smoke test is considered passed only if:

```text
1. Slurm allocates the H200 GPU.
2. Apptainer sees the GPU with --nv.
3. Python imports torch, transformers, and vLLM.
4. PyTorch reports CUDA available.
5. HF metadata access works.
6. Tokenizer/chat template loads.
7. One short generation succeeds.
8. Five-prompt JSONL baseline succeeds.
```

A partial pass should be documented in [docs/hpc_setup_notes.md](docs/hpc_setup_notes.md) using the partial-pass log template there.

If compute nodes cannot access the internet, pre-download the model to `$SCRATCH/huggingface` from an allowed node first.

---

## 8. Baseline Run

Driver: [scripts/baseline_generate.py](scripts/baseline_generate.py) with [configs/baseline_h200.yaml](configs/baseline_h200.yaml), submitted via [slurm/run_baseline.slurm](slurm/run_baseline.slurm). Each line of the input JSONL (e.g. [data/sample_prompts_5.jsonl](data/sample_prompts_5.jsonl)) is one independent prompt. The script renders each prompt through the Nemotron chat template, runs inference, and writes one JSON object per prompt to `outputs/predictions_<JOB_ID>.jsonl`, recording `response`, `latency_sec`, model id, backend, and any per-prompt `error` so the run is debuggable even when some prompts fail.

Example output line:

```json
{"id": "sample_001", "prompt": "...", "response": "...", "latency_sec": 12.84, "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "backend": "vllm", "error": null}
```

---

## 9. Baseline Configuration

See [configs/baseline_h200.yaml](configs/baseline_h200.yaml). Headline knobs:

```yaml
model:
  name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  dtype: bfloat16
  max_model_len: 65536
  max_tokens: 60000
  temperature: 1.0
  top_p: 1.0
  trust_remote_code: true

runtime:
  backend: vllm
  gpu_memory_utilization: 0.90
  tensor_parallel_size: 1
```

The current `max_model_len`/`max_tokens` are sized for AIME-style reasoning traces — see the AIME25 entry in §17 for the budget rationale. Drop them back to ~32K / ~24K for shorter prompts to free KV-cache headroom.

---

## 10. Submit the Baseline

The baseline Slurm script lives at [slurm/run_baseline.slurm](slurm/run_baseline.slurm). Submit with:

```bash
sbatch slurm/run_baseline.slurm
```

Predictions land in `outputs/predictions_<JOB_ID>.jsonl`.

---

## 11. Baseline Interpretation

After a baseline run, inspect:

```text
logs/
outputs/
```

Minimum baseline report:

| Metric | Meaning |
|---|---|
| Total prompts | Number of attempted samples |
| Successful generations | Number of prompts with valid model output |
| Failure count | Number of prompts with errors |
| Average latency | Mean generation time per prompt |
| Median latency | More robust latency estimate |
| Output length | Approximate response length |
| Common failures | OOM, timeout, model load error, parsing issue |

Suggested first milestone:

```text
Run 5 prompts successfully.
Then run 20 prompts.
Then run 100 prompts.
Only after that, move toward full competition evaluation.
```

---

## 12. Development Workflow

### Local Laptop

```bash
git clone git@github.com:<YOUR_USERNAME>/Nemotron_Challenge.git
cd Nemotron_Challenge
code .
```

Edit locally, then:

```bash
git add .
git commit -m "Add baseline run scripts"
git push
```

### HPC through OOD

```bash
cd $HOME/Nemotron_Challenge
git pull
sbatch slurm/smoke_test.slurm
```

Then:

```bash
sbatch slurm/run_baseline.slurm
```

---

## 13. Common Problems

### Out of Memory

Possible causes: context length too large, `gpu_memory_utilization` too high, or vLLM KV cache allocation too aggressive. Try:

```yaml
model:
  max_model_len: 8192
  max_tokens: 512

runtime:
  gpu_memory_utilization: 0.80
```

### Hugging Face download is slow or fails

Possible causes: cache path not persistent, missing HF token, model terms not accepted, login node has internet but compute node does not (or vice versa). On Explorer, `huggingface-cli` is deprecated — use `hf auth login`:

```bash
export HF_HOME=$SCRATCH/huggingface
hf auth login
```

If compute nodes have no internet, download/cache the model from an allowed environment first.

---

## 14. LoRA Verification

Once the baseline is stable, the next milestone is the **LoRA verification** run — proof that the full submit-to-Kaggle path (data prep → train adapter → load via vLLM → score → package zip) works end-to-end on a single H200. This is a verification milestone, not a tuning run.

Kaggle's eval params are fixed and inverted from the model card defaults: rank ≤ 32, `max_tokens=7680`, `max_model_len=8192`, `temperature=0.0`. Training distribution is shaped to produce concise traces that fit greedy-decoded under that 8K cap. Rationale, hyperparameter choices, and known follow-ups in [docs/lora_strategy.md](docs/lora_strategy.md).

Pipeline files:

| Stage | File |
|---|---|
| Data prep (login node) | [scripts/prepare_reasoning_traces.py](scripts/prepare_reasoning_traces.py) |
| Training config | [configs/lora_verification.yaml](configs/lora_verification.yaml) |
| Trainer | [scripts/train_lora.py](scripts/train_lora.py) |
| Eval at Kaggle params | [configs/eval_kaggle.yaml](configs/eval_kaggle.yaml) + [scripts/baseline_generate.py](scripts/baseline_generate.py) (now LoRA-aware) |
| Submission packager | [scripts/package_submission.py](scripts/package_submission.py) |
| Slurm orchestration | [slurm/lora_verification.slurm](slurm/lora_verification.slurm) |

Prerequisites and exact command sequence in [docs/lora_strategy.md §3-4](docs/lora_strategy.md). One-time setup needed: pull the NeMo container to `$SCRATCH/containers/nemo.sif`.

---

## 15. Future Optimization Directions

After LoRA verification passes, possible next steps include:

1. Real LoRA training run (5K+ traces, more epochs, rank 24-32)
2. `enable_thinking=True` vs `False` A/B at eval time against the same adapter
3. Held-out dev set beyond AIME25 (e.g. MATH-500, GPQA-Diamond) to bound sample variance
4. On-policy distillation from a stronger teacher (R1 / QwQ-32B)
5. Prompt format tuning
6. Self-consistency sampling
7. Tool-use evaluation for math/coding prompts
8. Output parser and verifier
9. NeMo Evaluator Launcher integration
10. Error taxonomy and targeted fixes

---

## 16. Verified References

- Kaggle competition page: `https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge`
- Hugging Face model card: `https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- NVIDIA NeMo Evaluator example for Nemotron 3 Nano: `https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator-launcher/examples/nemotron/nano-v3-reproducibility.md`
- vLLM project documentation: `https://docs.vllm.ai/`

---

## 17. Bring-up Log

### 2026-04-27 — Initial cluster bring-up on Northeastern Explorer

**End-of-day status: HPC/infra setup is correct, H200 application path is fluent, smoke test pipeline (Stages 0–5) passes. Stage 6 / output JSONL persistence is the open item for next session.**

**Completed today:**

- **HPC / infra setup is basically correct.** `$SCRATCH=/scratch/$USER` exported in `~/.bashrc`, GitHub SSH auth from cluster working, scratch dirs created (`containers`, `huggingface`, `apptainer/{tmp,cache}`). Storage confirmed healthy — scratch is 2.0 PB total with 627 TB free; user usage 0 B going in.
- **H200 application path is overall fluent.** Probe job 6371194 allocated a single H200 (143,771 MiB VRAM, driver 570.86.15, CUDA 12.8) on partition `gpu` via `--gres=gpu:h200:1`. The earlier "H200 partition quota issue" was a misdiagnosis: `h200` is not a partition name on Explorer — it's a `--gres` value on the `gpu`/`gpu-short`/`gpu-interactive` partitions. H200 nodes: d4052–d4055 (8 GPUs each, 512 GB system RAM). A100 backup: d1026, d1028, d1029 (80 GB VRAM each).
- **Container pulled** to `$SCRATCH/containers/nemotron_vllm.sif` (7.7 GB, `vllm/vllm-openai:v0.12.0`) from a CPU allocation on `short` partition (login-node `/tmp` is too small for `mksquashfs`). Total wall time ~37 min.
- **Slurm scripts updated** through four iterations to land at a working H200 invocation:
    1. `--partition=h200` → `--partition=gpu --gres=gpu:h200:1`.
    2. `python` → `python3` (the vLLM container has no `python` symlink).
    3. `PYTHONNOUSERSITE=1` (Apptainer auto-binds `$HOME`, so a stale `~/.local/lib/python3.12/site-packages/` was shadowing the container's `vllm`/`flashinfer`).
    4. `trust_remote_code=True` added to `LLM(...)` calls and config YAMLs (Nemotron-H is a custom architecture; `NemotronHForCausalLM` is not in transformers core).
- HF auth: `hf auth login` (the deprecated `huggingface-cli` is gone on Explorer's HF version); token saved at `~/.cache/huggingface/token`. Stage 3 of the smoke test confirms the token is picked up inside the container.
- **Smoke test pipeline itself passes.** Job 6371616 ran on H200 node d4053 and Stages 0–5 all reported success. Stage 5 metrics:
    - Model loaded into 58.9 GiB of H200 VRAM.
    - Weight download from HF: 101.7 s (cold cache).
    - Weight load: 144.5 s.
    - KV cache: 47.9 GiB free, 1.67M-token capacity, 651× max concurrency.
    - Single-prompt generation latency: 20.15 s. Real text generated.
    - Final line of `smoke_test.py`: `All stages PASSED`.
- **Open question resolved by today's run:** compute nodes on Explorer *do* have outbound internet for HF downloads — Stage 5 successfully fetched ~60 GB to `$SCRATCH/huggingface` from a compute node.

**Open as of end of day — future jobs:**

1. **Stage 6 / 5-prompt baseline did not produce output.** Confirmed by `ls -lh outputs/`: the directory does not exist on the cluster. The slurm script's Stage 6 block (which invokes `scripts/baseline_generate.py` with `configs/smoke_h200.yaml`) either silently skipped or was cut off after `smoke_test.py` returned. Need to inspect `logs/nemotron_smoke_6371616.err` (which we never read cleanly — the original `cat` call was prefixed by a stray `$`) and the tail of `logs/nemotron_smoke_6371616.out` past `All stages PASSED` to determine: did the second `apptainer exec` block run at all? Did `baseline_generate.py` start and crash before `mkdir -p outputs/`? Hypothesis to check first: walltime / OOM is unlikely (the smoke job had a 1 h budget and Stage 5 finished within ~7 min), so suspect either (a) `set -euo pipefail` tripped on something subtle between blocks, or (b) `baseline_generate.py` errored before its `out_path.parent.mkdir(...)` ran (e.g. yaml parse error, or `trust_remote_code` not being plumbed through correctly when read from the YAML rather than hardcoded).
2. **Once Stage 6 produces the JSONL**, paste back its contents and we'll record the 5 prompt/response pairs and per-prompt latencies in this log. That's the "smoke test green" milestone before kicking off the full baseline.
3. **Run the full baseline** after Stage 6 is verified: `sbatch slurm/run_baseline.slurm`. Walltime is 4 h; predictions land in `outputs/predictions_<JOB_ID>.jsonl`.
4. **Clean up `~/.local`** so it can't shadow the container again on a future bring-up. Rename rather than delete:
   ```bash
   mv ~/.local/lib/python3.12/site-packages ~/.local/lib/python3.12/site-packages.bak
   ```
   Keep the `.bak` for ~1 week; remove once nothing on the cluster is observed to depend on it.

### 2026-04-28 — Stage 6 JSONL follow-up

Likely root cause: `data/sample_prompts_5.jsonl` was covered by the repo-wide `*.jsonl` ignore rule, so a fresh cluster clone did not receive the smoke prompts. `scripts/baseline_generate.py` read the input file before creating `outputs/`, so a missing input fixture produced the observed symptom: no `outputs/` directory and no smoke JSONL.

Fix: `data/sample_prompts_5.jsonl` is now an explicit tracked exception in `.gitignore`, and `baseline_generate.py` creates the output directory before validating the input path. If the input is ever missing again, the Stage 6 log should show a direct `FileNotFoundError` naming the missing JSONL.

### 2026-04-28 — AIME25 baseline pipeline verified end-to-end

**End-of-day status: AIME25 baseline pipeline runs cleanly on H200 — 30/30 prompts complete with no errors, accuracy scoring works, and the eval pipeline produces a number. Today's headline single-sample score is 13/30 = 43.3% pass@1, but is *not* a meaningful performance result: 17/30 responses ran past the 24K-token cap and were cut off mid-final-answer. Of the 13 responses that did fit in budget, 13 were correct (100%), confirming the model and pipeline are both healthy. Today's milestone is pipeline verification, not benchmark reproduction. Token budget bumped for next run; next session moves to LoRA.**

#### Run summary (job 6382411, H200 node d4055)

- Walltime ~45 min (model load 69 s, vLLM warmup 47 s, inference 30 prompts at avg 84 s / median 104 s).
- 30/30 successes, 0 failures.
- Mean response length 66.8K chars ≈ 16.7K tokens (~68% of the 24,576-token cap).
- All 30 responses contained at least one `\boxed{`; 17 had an unbalanced trailing `\boxed{` (truncation while writing the final answer).
- Score breakdown: 13 correct, 0 wrong (boxed integer mismatch), 17 missing final boxed answer, 0 non-integer boxed.

#### Decisions made today

- **Benchmark target:** AIME25 (no tools). 30 problems, integer-answer grading, headline reasoning benchmark, fits trivially in a single H200 walltime.
- **Dataset source:** `MathArena/aime_2025` on HF — most-cited public mirror. NVIDIA's report does not disclose which AIME25 mirror they used; this is a faithful but not bit-identical substitute.
- **Prompt template:** NeMo-Skills public `generic/math.yaml` boxed-answer convention — `"Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}."` with no system prompt. NVIDIA's exact template (`math-oai.yaml`) ships only inside their proprietary eval-factory container; the NeMo-Skills version is the closest published equivalent.
- **Decoding:** `temperature=1.0`, `top_p=1.0` (HF model card recommendation for reasoning; NVIDIA's eval YAML uses 0.99999 ≈ 1.0). `enable_thinking=True` (chat-template default).
- **Sampling strategy:** single-sample, 30 generations. NVIDIA's published 89.1 is avg@64 (1,920 generations), which is out of scope for verification work.

#### Technical issue / solution log

| Issue | Symptom | Fix |
|---|---|---|
| Long walltime stuck in queue | 4 h request sat in `(Priority)` for ~1 h | Cut `slurm/run_baseline.slurm` to `--time=02:00:00`. Slurm's backfill scheduler needs a contiguous gap ≥ requested walltime; 2 h gaps are far more common than 4 h gaps on the H200 nodes. Rerun started promptly. |
| `datasets` lib not in vLLM container | `ModuleNotFoundError: No module named 'datasets'` when running `prepare_aime25.py` inside `apptainer exec` | Rewrote `scripts/prepare_aime25.py` to use Python stdlib only (`urllib.request` + `json`) via the HF datasets-server REST API. Runs anywhere with internet — no container, no extra packages, no conflict with the slurm scripts' `PYTHONNOUSERSITE=1`. |
| Bash `\` line-continuation copy-paste | Single multi-line command silently splits into two: `apptainer exec` runs without a command (arg-count error) AND `python3 ...` runs separately on the host (its own error) — confusingly interleaved output | Paste as a single line. The `\` is fragile when followed by a stray space or non-breaking character. |
| Chat template silently skipped | Initial `baseline_generate.py` fed the raw user prompt to `llm.generate()`, bypassing the chat template and disabling thinking mode without warning | Explicitly call `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)` and feed the rendered text to `llm.generate()`. Save the rendered text in the output JSONL as `formatted_prompt` for reproducibility. |
| PEP 604 annotations on login-node Python | `TypeError: unsupported operand type(s) for \|: 'type' and 'NoneType'` when running `evaluate.py` on the login node — container has Python 3.12 but login node is older | `from __future__ import annotations` at the top of any host-side script. Defers all annotations to lazy strings, works on Python ≥3.7. Apply this to any future host-side script with type hints. |
| Token budget too tight for AIME reasoning traces | 17/30 responses truncated mid-final-boxed-answer, even though the model had clearly worked out the answer earlier in the trace. Math-heavy responses pack ~3 chars/token, so 24K tokens hit the cap at ~70K characters | Bumped `configs/baseline_h200.yaml`: `max_model_len 32768 → 65536`, `max_tokens 24576 → 60000`. Previous run had 62 GiB KV cache free, 2.16M-token KV pool, 288× max concurrency at 32K context — doubling context is well within the H200's budget for single-prompt inference. |
| `evaluate.py` reported 17 "no boxed" but `grep -c '\\boxed{'` found 30 | The extractor takes the *last* `\boxed{` and balances braces forward; if the trailing `\boxed{` is unclosed (truncation), it returns None even though earlier balanced `\boxed{...}` exist. This is intentionally honest — using an intermediate working step as the "final answer" would inflate scores misleadingly | No code change; rely on token-budget bump to fix the underlying truncation. The extractor's behavior is the right one. |

#### Note on today's headline number

13/30 (43.3%) is **not** a baseline result that should be quoted, compared, or improved against. It is a verification artifact — proof the pipeline produces consumable output end-to-end. A meaningful baseline number would require: (a) the bumped token budget, (b) ideally avg@N rather than single-shot, (c) several reruns to bound sample variance. The plan is to recompute the baseline number alongside the LoRA evaluation (same config, same conditions) so the baseline-vs-LoRA comparison is apples-to-apples.

#### Next session

Move to LoRA training strategy. Defer the baseline rerun until LoRA evaluation needs a comparison number — at that point, run both the baseline-mode and LoRA-mode evaluations under identical settings.

Questions:(a) training data preparation — which dataset, what format, how to ingest it — or (b) the trainer scaffolding — which library (PEFT? Unsloth? NeMo?), checkpointing strategy, eval-during-training. 
