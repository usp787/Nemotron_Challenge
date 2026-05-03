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
| New vLLM eval image pull | [slurm/pull_vllm_lora_eval.slurm](slurm/pull_vllm_lora_eval.slurm) |
| One-prompt LoRA load probe | [configs/eval_lora_probe.yaml](configs/eval_lora_probe.yaml) + [slurm/lora_vllm_probe.slurm](slurm/lora_vllm_probe.slurm) |
| Submission packager | [scripts/package_submission.py](scripts/package_submission.py) |
| Slurm orchestration | [slurm/lora_verification.slurm](slurm/lora_verification.slurm) |

Prerequisites and exact command sequence in [docs/lora_strategy.md §3-4](docs/lora_strategy.md). The current verification path uses two vLLM containers: the known-good v0.12.0 image for HF/PEFT training, and a newer CUDA-12.9 vLLM image for LoRA-enabled eval because v0.12.0 cannot initialize Nemotron-H MoE LoRA. Pull and probe the newer eval image before running the full verification:

```bash
sbatch slurm/pull_vllm_lora_eval.slurm
sbatch slurm/lora_vllm_probe.slurm
sbatch slurm/lora_verification.slurm
```

The pull job does not need a GPU. The one-prompt probe and full LoRA verification both request H200 so failures are not confounded by ambiguous A100 memory size on Explorer.

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

### 2026-04-29 — LoRA strategy locked in, verification job submitted

**End-of-day status:** LoRA strategy is fully designed, all pipeline scripts/configs/slurm are in the repo, training data is prepped, runtime environment on the cluster is verified. Job 6412058 (`lora_verification`) is queued on the gpu partition (PD, Priority). Today is the design + plumbing milestone — running the verification is just-submitted, not finished yet.

#### Decisions made today

- **Kaggle eval contract drives everything.** The leaderboard runs the adapter via vLLM with fixed params: `max_lora_rank=32`, `max_tokens=7680`, `max_model_len=8192`, `temperature=0.0` (greedy), `top_p=1.0`, scoring on `\boxed{}` extraction. This is the **inverse** of the model card's recommended setup (temp=1.0, ~30K-token reasoning budgets). Every downstream choice — training-data length filter, chat-template format, hyperparameter sizing — is a consequence of these constraints. See [docs/lora_strategy.md](docs/lora_strategy.md) §1.
- **Training stack: HF transformers + peft + accelerate, not pure NeMo CLI.** NeMo Automodel is a wrapper over the same libraries; calling them directly is more durable across NeMo releases and avoids YAML schema drift. PEFT writes `adapter_config.json` + `adapter_model.safetensors` in exactly the shape Kaggle expects.
- **Container reuse over a NeMo container pull.** Initial plan pulled `nvcr.io/nvidia/nemo:25.07` (~30 GB, 4+ hours). Pivoted to reusing `$SCRATCH/containers/nemotron_vllm.sif` and adding a scratch-resident `peft + accelerate` install via `PYTHONPATH` injection. Setup time dropped from hours to minutes; saved ~30 GB of `$SCRATCH`.
- **Data source: `nvidia/OpenMathReasoning` (CoT split).** NVIDIA's own curated math reasoning corpus minimises distribution shift during fine-tuning. Filtered to `problem_type=="has_answer_extracted"`, `\boxed{` present in solution, length ≤ 14000 chars (~ 4-4.5K tokens) — strict enough that traces fit comfortably under the 7680-token greedy eval cap.
- **Hyperparameters: deliberately small for verification.** rank=16 (half the Kaggle cap of 32), `target_modules="all-linear"` (robust to Nemotron's hybrid Mamba-2/MoE arch), `max_seq_len=4096`, 200 samples × 1 epoch, `gradient_checkpointing=True`. Full rationale in [docs/lora_strategy.md](docs/lora_strategy.md) §2.
- **Memory budget on H200 is comfortable.** Estimated ~75-90 GiB during training (60 GiB base BF16 weights + ~10-20 GiB activations with checkpointing + LoRA/optimizer/workspace), leaving ~50-65 GiB headroom on the 140 GiB H200. Headroom is reserved for future rank/batch increases rather than a `seq_len` bump.
- **Two-job split (train.slurm + eval.slurm) deferred.** For verification, a monolithic `slurm/lora_verification.slurm` is simpler. The split, plus training checkpoints + eval resume, is documented as a future-direction in [docs/lora_strategy.md](docs/lora_strategy.md) §7 — natural follow-up once a real (multi-day) training run is needed.

#### Pipeline files added today

| Stage | File |
|---|---|
| Data prep (login node) | [scripts/prepare_reasoning_traces.py](scripts/prepare_reasoning_traces.py) |
| Training config | [configs/lora_verification.yaml](configs/lora_verification.yaml) |
| Trainer | [scripts/train_lora.py](scripts/train_lora.py) |
| Eval at Kaggle params | [configs/eval_kaggle.yaml](configs/eval_kaggle.yaml) + LoRA-aware [scripts/baseline_generate.py](scripts/baseline_generate.py) |
| Submission packager | [scripts/package_submission.py](scripts/package_submission.py) |
| Slurm orchestration | [slurm/lora_verification.slurm](slurm/lora_verification.slurm) |
| Strategy doc | [docs/lora_strategy.md](docs/lora_strategy.md) |

#### Technical issue / solution log

| Issue | Symptom | Fix |
|---|---|---|
| `prepare_reasoning_traces.py` rejected all rows | 4300 rows scanned with `kept=0/200`; eventually a 429 from rate-limited paging | Wrong column names assumed. Real schema (verified from HF dataset card): `pass_rate_72b_tir` is a string with `"n/a"` sentinel, not `pass_rate_1` numeric. Rewrote `keep()` to use the real columns, added a first-page `[diag]` dump of row keys, added a reject tally so any future schema drift surfaces in seconds rather than after 4000 silent rejections. Also added 429 backoff that reads `Retry-After`, exponential fallback, and a 0.5 s inter-page sleep. |
| Python 3.9 f-string compatibility | `prepare_reasoning_traces.py` would have raised `SyntaxError` on the login node — Python 3.9 forbids backslashes inside f-string expressions, only allowed since 3.12 | Bound the boolean to a variable before formatting, so the f-string expression contains no backslash. |
| First NeMo container pull `salloc` exceeded its time limit doing nothing | Pasted `salloc` + commands together; on Explorer `salloc` allocates resources but does NOT auto-shell into the compute node, so subsequent `mkdir`/`apptainer pull` ran on the login node while the compute allocation sat idle. NeMo container size also exceeds what fits in 2 h on `short`. | Switched the entire training stack to reuse the existing vLLM container (Option B above). Documented `srun --jobid=$SLURM_JOB_ID --pty bash` after `salloc` as the explicit step needed on Explorer. |
| `pip install --target=...` OOM-killed the login node | Attempted to download a fresh torch + CUDA wheel set (~2.5 GB) into the target dir even though the container already has compatible torch/transformers. Killed at "4/56 [triton]" by the login node's memory cgroup. | Added `--no-deps` to the `pip install --target` command. Only `peft` (~680 KB) and `accelerate` (~380 KB) get written to scratch; `torch`/`transformers`/etc. are picked up from the container at import time via the standard module-search order. Updated [docs/lora_strategy.md](docs/lora_strategy.md) §3 with the rationale. |
| Bash multi-line copy-paste broke verification command | Blank lines between continuations made bash treat `apptainer exec` as a complete command with no arguments; subsequent lines ran on the host and triggered an unrelated host-side `~/.local` vLLM traceback | Paste verification commands as a single line. Same trap previously documented in the AIME25 entry — recurring when copy-pasting from rendered docs that insert blank lines between code-block lines. |

#### Verified runtime environment

Inside `nemotron_vllm.sif` with `PYTHONPATH=$SCRATCH/lora_pip:$PYTHONPATH`:

- torch 2.10.0+cu128 (container)
- transformers 5.6.2 (container)
- peft 0.19.1 (scratch)
- accelerate 1.13.0 (scratch)

#### Open as of end of day

- **Job 6412058 in queue (PD, Priority).** Awaiting a free H200 on the gpu partition. Walltime budget is 4 h. Pass criteria are the seven verification stages enumerated in [docs/lora_strategy.md](docs/lora_strategy.md) §4. Once it runs, capture the Stage 3 `print_trainable_parameters()` output (sanity-checks PEFT's `all-linear` discovery on the hybrid arch), Stage 5 latencies, and Stage 6 score — quality of the score is secondary, the artifact existing is the milestone.
- **Spot-checks not yet done.** Need to eyeball a sample from `data/lora_traces.jsonl` (e.g. `head -1 data/lora_traces.jsonl | python3 -m json.tool`) before the job lands a GPU, just to confirm no obvious format mistakes survived the pipeline.
- **Data filter `--min-pass-rate` left at 0.0.** Most R1 traces have `pass_rate_72b_tir == "n/a"` (unevaluated by the 72B-TIR system); raising the threshold would drop the pool aggressively. Revisit after the first real training run if quality looks limited by data noise rather than data scale.

### 2026-04-30 — LoRA verification blocked by mamba-ssm; Option A install green, resubmission queued

**End-of-day status:** Two retry attempts of the LoRA verification (jobs 6412058 and 6445308) failed at Stage 3 — both crashed before any GPU compute, on the HF transformers model load. Root cause: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`'s remote `modeling_nemotron_h.py` imports `mamba_ssm` unconditionally, and the vLLM container ships no `mamba-ssm` package. Option B (drop `trust_remote_code`) failed instantly because the container's transformers is older than yesterday's note recorded. Option A (install `mamba-ssm` + `causal-conv1d` into `$SCRATCH/lora_pip`) succeeded after a source build inside the container. Resubmission of `slurm/lora_verification.slurm` is queued for next session.

#### Decisions made today

- **Diagnosed the mamba-ssm dependency.** vLLM-side inference works because vLLM ships its own internal Mamba implementation (the AIME25 baseline run on this same container is the existence proof). The HF transformers loading path is different: `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` fetches NVIDIA's `modeling_nemotron_h.py` and that file does `from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn` at module-import time. So this is purely a *training-time* problem; the eval pipeline (Stages 5–6) was never going to be affected.
- **Option B (dodge mamba-ssm by using transformers' built-in `NemotronHForCausalLM`) is dead.** Set `trust_remote_code: false` in [configs/lora_verification.yaml](configs/lora_verification.yaml) and resubmitted (job 6445308) — failed instantly with `ValueError: ... contains custom code which must be executed`. Diagnosis: the container's transformers version is **4.57.3, not 5.6.2** as [docs/lora_strategy.md](docs/lora_strategy.md) §3 recorded yesterday. 4.57.3 has no built-in `NemotronHForCausalLM`, so the auto-class loader has nowhere to dispatch without remote code. Reverted the YAML to `trust_remote_code: true` with a pointer to the matching scratch install.
- **Option A (install `mamba-ssm` + `causal-conv1d` into `$SCRATCH/lora_pip`) succeeded.** Same scratch-pip pattern as `peft`/`accelerate`: `pip install --target=$SCRATCH/lora_pip --no-deps --no-build-isolation mamba-ssm causal-conv1d`. PyPI does not ship cp312/cu128/torch2.10-compatible prebuilt wheels for either package, so pip fell through to a source build using the vLLM container's bundled `nvcc`. Build took ~3 min on a CPU compute node; resulting wheels are large (mamba-ssm 2.3.1: 533 MB, causal-conv1d 1.6.1: 287 MB) but cached at `~/.cache/pip/wheels/...` for reinstalls. `--no-deps` kept torch/transformers coming from the container; `--no-build-isolation` told pip to reuse the container's torch headers for the compile rather than fetching a fresh isolated build environment.
- **Verified imports.** From a CPU compute-node shell inside the container, `python3 -c "import torch, mamba_ssm, causal_conv1d; print(torch.__version__)"` printed `2.10.0+cu128` and exited 0. mamba-ssm and causal-conv1d both load cleanly against the container's torch.

#### Technical issue / solution log

| Issue | Symptom | Fix |
|---|---|---|
| `mamba-ssm` missing from vLLM container | Job 6412058 stage 3 crashed at `train_lora.py:168` with `ImportError: mamba-ssm is required by the Mamba model but cannot be imported`, raised from inside `modeling_nemotron_h.py` line 66 | Installed `mamba-ssm` and `causal-conv1d` into `$SCRATCH/lora_pip` via `pip install --target=... --no-deps --no-build-isolation`. Source build because no prebuilt wheel matches the container's torch/cu version, but the build succeeded inside the container without extra setup. |
| Option B fast-failed | Job 6445308 with `trust_remote_code: false` immediately raised `ValueError: ... contains custom code which must be executed. Please pass the argument 'trust_remote_code=True'` | transformers 4.57.3 has no built-in NemotronH and the model config carries `auto_map`, so the auto-loader refuses to dispatch without remote code. Option B is permanently off the table for this stack; reverted [configs/lora_verification.yaml](configs/lora_verification.yaml). |
| Multi-line copy-paste trap struck again | Pasted block of `salloc` + `srun --pty bash` + `apptainer exec ... \` did not actually enter the compute node and the install never ran. `salloc` allocated then returned to the login node; queued lines ran on the login node where the `\` continuation broke at a blank line, producing `Error for command "exec": requires at least 2 arg(s)`. The verify command failed the same way. | Recovery: `srun --jobid=<id> --pty bash` as a standalone command (confirm the prompt changes to the compute-node hostname **before** running anything else), then run install/verify each as a single physical line. Third time this trap has bitten — also documented in the AIME25 and 04-29 entries. Future doc edits should give one-line copy-paste targets, not multi-line fragments with `\` continuations. |
| torch version mismatch between failed jobs and today's verify | Failed Stage 3 logs reported `torch 2.9.0+cu129`; today's import verify on c0587 prints `torch 2.10.0+cu128` from the same `.sif` with the same `PYTHONPATH=$SCRATCH/lora_pip` | Unresolved — flagged as a watch-out for resubmission. Most likely cause is a leftover `torch/` directory inside `$SCRATCH/lora_pip` from the day-1 OOM-killed install (which ran *without* `--no-deps`); H200 nodes might be picking it up while the CPU node we used today doesn't. `ls $SCRATCH/lora_pip/torch 2>/dev/null` will confirm. If Stage 3 still prints 2.9 on resubmission, the leftover should be removed before mamba-ssm — built today against torch 2.10 — gets used in anger and runs into an ABI mismatch. |

#### Verified runtime environment (refresh of yesterday's note)

Inside `nemotron_vllm.sif` with `PYTHONPATH=$SCRATCH/lora_pip:$PYTHONPATH`, observed today on a CPU compute node:

- torch 2.10.0+cu128 (container)
- transformers 4.57.3 (container) — corrects yesterday's note that recorded 5.6.2
- peft 0.19.1 (scratch)
- accelerate 1.13.0 (scratch)
- mamba-ssm 2.3.1 (scratch, source-built today against the container's torch)
- causal-conv1d 1.6.1 (scratch, source-built today against the container's torch)

#### Open as of end of day

- **Resubmit `sbatch slurm/lora_verification.slurm`.** Pass criteria unchanged from yesterday (the seven verification stages in [docs/lora_strategy.md](docs/lora_strategy.md) §4). Stage 3 is now expected to clear the model-load step and start emitting `Loading checkpoint shards: ...%`. The `print_trainable_parameters()` line shortly after is the next signal worth capturing — implausibly small (<0.05% of total) would mean PEFT's `all-linear` discovery missed the bulk of the hybrid arch and we need to revisit `target_modules`.
- **Strategy doc has stale version numbers.** [docs/lora_strategy.md](docs/lora_strategy.md) §3 "Verified runtime environment" lists `torch 2.10.0+cu128 / transformers 5.6.2`. Actual is `torch 2.10.0+cu128 / transformers 4.57.3`. Worth a small follow-up edit; non-blocking. Should also note today's mamba-ssm + causal-conv1d additions to the scratch install.
- **Investigate the torch 2.9 vs 2.10 discrepancy if it recurs on H200.** First check: `ls $SCRATCH/lora_pip/torch 2>/dev/null` from a login-node shell. If a `torch/` directory exists, that's almost certainly the leftover from the day-1 OOM install; remove it (`rm -rf $SCRATCH/lora_pip/torch $SCRATCH/lora_pip/torch-*.dist-info $SCRATCH/lora_pip/nvidia*`) so PYTHONPATH falls through to the container's torch 2.10, which is what mamba-ssm was source-built against today.
The latest job ID is 6459431

### 2026-05-02 — Mamba-ssm ABI mismatch resolved; new blocker is vLLM MoE LoRA limitation; attention-only retraining queued

**End-of-day status:** Three days of mamba-ssm failures finally root-caused — the previous source build linked against `~/.local`'s torch (2.10) while the slurm runtime forced the container's torch (2.9) via `PYTHONNOUSERSITE=1`, producing the `_ZN3c104cuda...` ABI symbol crash that earlier sessions misdiagnosed as "missing fallback" and tried to fix by deleting mamba-ssm. Today's rebuild with `PYTHONNOUSERSITE=1` set inside the build environment links cleanly against torch 2.9, and the resulting adapter trained end-to-end (job 6516064 reached Stage 3 model load + LoRA fit + adapter save without errors). New blocker discovered immediately after: **vLLM 0.12.0's `NemotronHForCausalLM` does not implement `get_expert_mapping`**, so any LoRA adapter that targets MoE expert layers — which the `target_modules: all-linear` config produced (11,868 of 12,008 LoRA tensors lived under `mixer.experts.<idx>.{down_proj,up_proj}`) — fails to load with `AttributeError: To support LoRA for MoE model, 'get_expert_mapping' must be implemented`. Pivoted `target_modules` to an explicit attention-only list `[q_proj, k_proj, v_proj, o_proj]`; resubmission is queued.

#### Decisions made today

- **Root cause of the multi-day mamba-ssm blocker.** The container ships `torch 2.9.0+cu129`, not `2.10.0+cu128` as recorded on 04-29. The 04-29 verification was run *without* `PYTHONNOUSERSITE=1`, so apptainer's default home bind-mount let `~/.local/lib/python3.12/site-packages/torch` (2.10.0+cu128, source unknown but predates this project) shadow the container's bundled torch. The mamba-ssm source build also ran without `PYTHONNOUSERSITE=1`, so it linked against `~/.local`'s torch 2.10. The slurm script then sets `PYTHONNOUSERSITE=1` (deliberately, per the comment at [slurm/lora_verification.slurm:53-54](slurm/lora_verification.slurm#L53-L54)), which switches runtime to container torch 2.9. Result: ABI mismatch every time the job ran. The eb3f915 commit's "delete mamba-ssm so transformers falls back" was a misdiagnosis — Nemotron-H's `trust_remote_code` modeling code does an unconditional `from mamba_ssm... import rmsnorm_fn` at module load with no fallback path.
- **Fix: rebuild with `PYTHONNOUSERSITE=1` set during the build.** Same `pip install --target=$SCRATCH/lora_pip --no-build-isolation --no-deps` invocation as 04-29, but executed inside `apptainer exec ... bash -lc "export PYTHONNOUSERSITE=1; ... pip install ..."` on a CPU compute node (32 GB RAM, 8 CPUs, ~15 min wall time for the mamba-ssm wheel). Build log confirms `linking against torch 2.9.0+cu129` on the first line; verify shell with the same `PYTHONNOUSERSITE=1` reports `2.9.0+cu129 2.3.1 1.6.1` — torch/mamba_ssm/causal_conv1d. ABI match is the signal that this round will hold.
- **Reverted the eb3f915 "delete mamba-ssm" hack.** [slurm/lora_verification.slurm](slurm/lora_verification.slurm) lines 85-96 used to `rm -rf` the scratch mamba-ssm install at the start of every job. Replaced with `test -d` prerequisite checks that fail Stage 1 fast if either `mamba_ssm/` or `causal_conv1d/` is missing. Stage 3's pre-training health-check now does a hard `import mamba_ssm, causal_conv1d` instead of the previous "absent ⇒ slow path" stub — ABI mismatches surface in seconds instead of 60s into model load.
- **Job 6516064 (resubmission) cleared every prior failure point and exposed a new one.** Stages 0–4 green: GPU allocation, container GPU passthrough, all four library imports including the rebuilt mamba-ssm, model load, training (12 optim steps in ~6 min), adapter save. Stage 5 (vLLM eval) crashed inside `process_packed_modules_mapping` with `AttributeError: To support LoRA for MoE model, 'get_expert_mapping' must be implemented`. The error is structural: vLLM's MoE-LoRA codepath dispatches to a model-class-specific `get_expert_mapping` to know which expert weights map to which LoRA matrices, and `NemotronHForCausalLM` (vLLM's in-engine class, not the trust_remote_code one) does not provide one.
- **The MoE LoRA constraint is a Kaggle constraint, not just a local one.** Kaggle scores via vLLM (same engine, same model class). Until upstream vLLM ships `get_expert_mapping` for Nemotron-H, any submission containing LoRA on expert weights fails to load. Patching vLLM locally would not fix the leaderboard.
- **Adapter inspection confirmed expert-dominated topology.** The trained `outputs/lora_adapter/adapter_model.safetensors` from job 6516064 has 12,008 LoRA tensors. Expert-named (substring `experts`): 11,868 (≈ 98.8%). Mamba-named (`in_proj`/`out_proj`/`x_proj`/`dt_proj`): 46. Attention q/k/v/o: 48. Naming pattern is clean: `base_model.model.backbone.layers.<L>.mixer.experts.<E>.{down_proj,up_proj}.lora_{A,B}.weight`. So an `exclude_modules: "experts"` substring filter would also work, but introduces unknown risk around whether vLLM supports LoRA on Mamba mixer modules — deferred as a follow-up.
- **Switched `target_modules` to attention-only.** [configs/lora_verification.yaml](configs/lora_verification.yaml) now uses an explicit list `[q_proj, k_proj, v_proj, o_proj]`. Substring matching captures attention layers across all attention blocks and ignores experts, Mamba mixers, routers, and lm_head. Trade-off: LoRA capacity is concentrated on attention and won't touch the Mamba/MoE pathways. For the verification milestone the bar is "adapter is produced, vLLM loads it, AIME25 scores without erroring" — not quality. The exclude_modules variant remains the right next experiment once verification is green.
- **[docs/lora_strategy.md](docs/lora_strategy.md) and [scripts/train_lora.py](scripts/train_lora.py) updated.** Strategy doc §2 stack bullet, §2 hyperparameters table, and a new §3 subsection "vLLM MoE LoRA constraint" explain the new target_modules choice and the failure mode that drove it. Train script's docstring rationale matches.

#### Technical issue / solution log

| Issue | Symptom | Fix |
|---|---|---|
| `~/.local` user-site torch shadowed container torch in 04-29 verify | The 04-29 verify command printed `torch 2.10.0+cu128`, but the slurm job (with `PYTHONNOUSERSITE=1`) printed `torch 2.9.0+cu129`. Same container `.sif`, two different torches, depending on whether `PYTHONNOUSERSITE=1` was set. The 04-29 mamba-ssm wheel linked against `~/.local`'s 2.10; the slurm runtime used the container's 2.9 → ABI symbol crash. | Rebuilt mamba-ssm + causal-conv1d with `PYTHONNOUSERSITE=1` exported inside the apptainer build shell. Verified with the same flag set. Both build-time and runtime now see `torch 2.9.0+cu129` from `/usr/local/lib/python3.12/dist-packages/`, and the adapter loads without ABI errors at training time. `~/.local` left alone — it is harmless once `PYTHONNOUSERSITE=1` is honored everywhere, which the slurm script already does. |
| eb3f915 "delete mamba-ssm" commit was based on a wrong premise | The commit's rationale was "transformers will fall back to a pure-PyTorch Mamba path" if mamba-ssm is removed. That is true for transformers' built-in `MambaModel`, but Nemotron-H is loaded via `trust_remote_code` and NVIDIA's `modeling_nemotron_h.py` does an unconditional `from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn` at module load with no fallback branch — it just re-raises `ImportError`. Job 6484958 (2026-05-01) failed at exactly that import, taking the misdiagnosis-fix-misdiagnosis loop one step further. | Reverted the rm-block in the slurm script and replaced it with hard `test -d` prerequisite checks. Strategy doc §3 now says explicitly "There is no PyTorch fallback for this model." |
| vLLM 0.12.0 `NemotronHForCausalLM` lacks `get_expert_mapping` | Job 6516064 Stage 5 crashed inside `vllm/lora/utils.py:301`: `AttributeError: To support LoRA for MoE model, 'get_expert_mapping' must be implemented`. The crash happens in `process_packed_modules_mapping` during LoRA manager construction, before any expert kernel runs — vLLM cannot even *plan* LoRA application without the mapping. | Switched `target_modules` from `all-linear` (which produced LoRAs on 11,868 expert tensors) to an explicit attention-only list `[q_proj, k_proj, v_proj, o_proj]`. Adapter shrinks from 12,008 LoRA tensors to ~48 — only attention. Patching vLLM locally is not viable because Kaggle uses the same vLLM. |
| Multi-line copy-paste trap, again | Pasted the verify command with `\` line continuations + blank lines between them. Result: `apptainer exec` ran with no command (arg-count error), `bash -lc ...` ran on the host without torch in scope, and `python3 -c '...'` ran on the host as a separate process — three confusingly-interleaved errors for what should have been one command. | Same fix as the AIME25, 04-29, and 04-30 entries: paste verification commands as a single physical line. Already three sessions in a row; the doc edits in this branch removed multi-line `\`-continuation snippets where possible to reduce future occurrences. |
| `srun --pty bash` after `salloc` did not actually shell into the compute node | After `salloc` granted the allocation, the chained `srun --jobid=$SLURM_JOB_ID --pty bash` returned to the login node prompt instead of `c0612`. Subsequent commands ran on the login node, which has too little `/tmp` for the mamba-ssm build. | `srun --jobid=<id> --pty bash` as a standalone command (with the literal job id from `salloc` output, since `$SLURM_JOB_ID` is not always exported back to the parent shell). Confirm hostname change with `hostname` before running anything else. |

#### Verified runtime environment (revised)

Inside `nemotron_vllm.sif` with `PYTHONPATH=$SCRATCH/lora_pip:$PYTHONPATH` and `PYTHONNOUSERSITE=1`, observed today on c0612 and c0587:

- torch 2.9.0+cu129 (container — corrects the 04-29 record of `2.10.0+cu128`, which was reading from `~/.local`)
- transformers 4.57.3 (container)
- peft 0.19.1 (scratch)
- accelerate 1.13.0 (scratch)
- mamba-ssm 2.3.1 (scratch, rebuilt today against the container's torch 2.9)
- causal-conv1d 1.6.1 (scratch, rebuilt today against the container's torch 2.9)

Without `PYTHONNOUSERSITE=1`, `~/.local`'s torch 2.10.0+cu128 is loaded instead. This must be set on both build and verify shells, not just slurm runtime.

#### Open as of end of day

- **Resubmitted `slurm/lora_verification.slurm` is queued (job ID: 6518135).** Pass criteria: Stage 5 reaches `Loading weights took ...s` and continues into vLLM warmup + AIME25 generation, instead of crashing in `process_packed_modules_mapping`. Adapter sanity check on the new run: `print_trainable_parameters()` should show `trainable% ≈ 0.05–0.2%` (much smaller than the all-linear run's number, since we dropped 98.8% of the LoRA tensors). Anything implausibly small (<0.01%) means the substring match missed the attention layers.
- **Quality of the verification score is still secondary.** Goal is end-to-end pipeline green, not leaderboard quality. AIME25 accuracy on a 200-sample × 1-epoch attention-only LoRA is expected to be near or below the no-LoRA baseline; that is fine for this pass.
- **Follow-up: try `target_modules: all-linear` + `exclude_modules: "experts"` once verification is green.** Would broaden LoRA coverage to Mamba `mixer.in_proj`/`mixer.out_proj` (~46 tensors on top of the attention 48). Untested whether vLLM's `NemotronHForCausalLM` supports LoRA on Mamba mixer modules — last run never got past the MoE check to find out. Cheap experiment after the milestone is unblocked.
- **Long-term: watch for upstream vLLM `get_expert_mapping` support on Nemotron-H.** If/when it ships, MoE-expert LoRA becomes viable and the strategy can revisit `all-linear` for richer adapters. Track via vLLM release notes; not urgent.
