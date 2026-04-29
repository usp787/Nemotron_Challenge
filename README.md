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

### 1.2 Target Model: NVIDIA Nemotron 3 Nano 30B-A3B BF16

Model card:

```text
https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

According to NVIDIA's Hugging Face model card, **Nemotron-3-Nano-30B-A3B-BF16** is a large language model trained from scratch by NVIDIA and designed as a unified model for both reasoning and non-reasoning tasks.

Important model properties:

| Property | Description |
|---|---|
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Provider | NVIDIA |
| Architecture | Hybrid Mixture-of-Experts architecture with Mamba-2/MoE layers and attention layers |
| Total parameters | 30B |
| Active parameters | Approximately 3.5B active parameters per token |
| Precision | BF16 checkpoint |
| Main use case | Reasoning and non-reasoning chat/instruction tasks |
| Reasoning behavior | Can generate reasoning traces before final answers, depending on chat template / reasoning configuration |
| Hardware target in this repo | H200 GPU through HPC Slurm job |

NVIDIA's evaluator examples also show Nemotron 3 Nano being evaluated on reasoning, coding, tool-use, math, science, instruction-following, multilingual, and long-context benchmarks such as AIME 2025, GPQA, LiveCodeBench, MMLU-Pro, BFCL, IFBench, HLE, WMT24++, and RULER.

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

### Why not Jupyter Notebook?

Jupyter Notebook is useful for result inspection, but it is not ideal for the main baseline run because:

- Notebook kernels often use a different Python environment than the GPU job.
- Long model-loading jobs are harder to reproduce from cells.
- Slurm logs are easier to debug than notebook-side crashes.
- vLLM server or batch inference is better managed as a script.
- H200 allocations should be used with explicit resource requests.

Recommended usage:

| Tool | Role |
|---|---|
| VS Code local | Code editing |
| GitHub | Version control and sync |
| OOD portal | HPC access |
| Slurm | GPU job scheduling |
| Apptainer | Reproducible runtime |
| Python scripts | Baseline execution |
| JupyterLab | Optional result analysis only |

---

## 4. Key Concepts

### 4.1 Slurm

Slurm is the HPC job scheduler. It controls access to shared resources such as GPU nodes, CPU cores, RAM, and runtime.

Instead of directly running a large model on a login node, we submit a job:

```bash
sbatch slurm/run_baseline.slurm
```

Slurm then decides when and where the job runs.

Common commands:

```bash
# Submit a job
sbatch slurm/run_baseline.slurm

# Check user jobs
squeue -u $USER

# Cancel a job
scancel <JOB_ID>
```

---

### 4.2 Apptainer

Apptainer is a container runtime commonly used on HPC systems. It is similar in purpose to Docker, but it is designed for shared clusters where Docker may be blocked.

Apptainer provides the Python/CUDA/vLLM/Transformers environment. Slurm provides the GPU allocation.

Typical pattern:

```bash
apptainer exec --nv container.sif python3 script.py
```

The `--nv` flag exposes NVIDIA GPUs to the container.

---

## 5. Suggested Repository Structure

```text
Nemotron_Challenge/
├── README.md
├── .gitignore
├── configs/
│   ├── smoke_h200.yaml
│   └── baseline_h200.yaml
├── data/
│   ├── sample_prompts_5.jsonl
│   └── README.md
├── scripts/
│   ├── check_env.py
│   ├── smoke_test.py
│   ├── baseline_generate.py
│   └── evaluate.py
├── slurm/
│   ├── smoke_test.slurm
│   └── run_baseline.slurm
├── docs/
│   └── hpc_setup_notes.md
├── logs/
└── outputs/
```

Recommended `.gitignore`:

```gitignore
# Runtime outputs
logs/
outputs/
*.out
*.err
*.jsonl
!data/sample_prompts_5.jsonl

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# Caches
.cache/
.huggingface/

# Large files
models/
checkpoints/
*.sif
*.simg

# Local environment
.env
.venv/
```

---

## 6. Storage Policy

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

## 7. Setup on HPC

### 7.1 Clone the Repository

From the OOD terminal:

```bash
cd $HOME
git clone git@github.com:usp787/Nemotron_Challenge.git
cd Nemotron_Challenge
mkdir -p logs outputs
```

If SSH authentication is not configured, use HTTPS or set up an HPC SSH key for GitHub.

---

### 7.2 Set Hugging Face Cache

Large model files should go to scratch storage:

```bash
export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub
```

Optional: add these exports to the Slurm scripts rather than `.bashrc` so the run is self-contained.

---

### 7.3 Container Provenance and Location

Expected container path:

```text
$SCRATCH/containers/nemotron_vllm.sif
```

This file is not committed to GitHub.

#### Why the container is still necessary even after config is set

The YAML config only controls **runtime choices**:

```text
model name
dtype
max_model_len
temperature
max_tokens
input/output paths
vLLM runtime options
```

The config does **not** provide the executable software environment:

```text
Python version
CUDA runtime libraries
PyTorch build
vLLM version
Transformers version
FlashAttention / custom kernels
system shared libraries
OpenAI-compatible serving dependencies
```

Therefore, the pipeline needs both:

```text
Apptainer container = reproducible software environment
YAML config         = reproducible experiment settings
Slurm script        = reproducible HPC resource request
```

A useful mental model:

```text
Slurm asks:       Where and with what resources should this job run?
Apptainer asks:   What software environment should this job use?
Config asks:      What model and generation settings should this experiment use?
```

#### Recommended container provenance levels

Use the most trustworthy available container source first.

| Priority | Container source | When to use | Notes |
|---|---|---|---|
| 1 | HPC-admin-provided vLLM / PyTorch / NeMo Apptainer image | Best first choice if available | Most likely to match the cluster driver, CUDA stack, filesystem policy, and H200 partition |
| 2 | Official vLLM Docker image converted to Apptainer `.sif` | Best self-managed baseline choice | Good for OpenAI-compatible server baseline and batch inference |
| 3 | NVIDIA NGC / NeMo Framework container converted to Apptainer | Good if using NeMo tooling or NVIDIA evaluator workflow | Potentially heavier, but closer to NVIDIA's ecosystem |
| 4 | Custom Apptainer definition file | Use only after baseline works | More control, but more maintenance/debugging |
| 5 | Manually patched Jupyter/conda environment | Avoid for main run | Harder to reproduce and easier to break |

The baseline recommendation for this repo is:

```text
Start with an existing vLLM-compatible container.
Do not build a custom container until a simple baseline already works.
```

#### Option A: Use an HPC-provided image

Ask the HPC team:

```text
Is there a maintained Apptainer image for CUDA + PyTorch + vLLM on the H200 partition?
Is there a maintained NVIDIA NeMo or NGC image?
Which image is recommended for H200 inference jobs?
```

If they provide a path like:

```text
/shared/containers/vllm/vllm-openai-cu12.sif
```

then set the Slurm script variable:

```bash
CONTAINER=/shared/containers/vllm/vllm-openai-cu12.sif
```

This is usually the cleanest option because the image is more likely to match the cluster driver and CUDA environment.

#### Option B: Pull an official vLLM Docker image into Apptainer

If the cluster allows pulling from Docker Hub:

```bash
mkdir -p $SCRATCH/containers
cd $SCRATCH/containers

apptainer pull nemotron_vllm.sif docker://vllm/vllm-openai:<PINNED_TAG>
```

Then test GPU visibility:

```bash
apptainer exec --nv $SCRATCH/containers/nemotron_vllm.sif nvidia-smi
```

Then test Python/vLLM:

```bash
apptainer exec --nv $SCRATCH/containers/nemotron_vllm.sif bash -lc "
python --version
python -c 'import vllm; print(\"vLLM import OK\")'
"
```

For reproducibility, avoid using `latest` permanently. After the first successful test, pin a specific image tag and record it in `docs/hpc_setup_notes.md`.

Example:

```text
Container source: docker://vllm/vllm-openai:<PINNED_TAG>
Converted SIF: $SCRATCH/containers/nemotron_vllm.sif
HPC node type: H200
Date tested: YYYY-MM-DD
Smoke test status: passed / failed
```

#### Option C: Use NVIDIA NGC / NeMo image

This is a good choice if the workflow later depends on NeMo Evaluator, NeMo Skills, or NVIDIA-specific evaluation tooling.

Ask the HPC team whether NGC images are already mirrored or available. If not, check whether pulling from NGC is allowed from the cluster.

Record the source clearly:

```text
Container source: NVIDIA NGC / NeMo Framework
Original image URI: <IMAGE_URI>
Converted SIF path: $SCRATCH/containers/nemotron_vllm.sif
Reason for use: NeMo/NVIDIA evaluator compatibility
```

#### Option D: Custom container definition file

Use this only after the basic baseline works. A custom definition file is useful when:

```text
official vLLM image lacks a required package
cluster-provided image has incompatible package versions
we need to pin exact versions for final reproducibility
we need to add project-specific evaluation dependencies
```

Example definition-file direction:

```text
Bootstrap: docker
From: vllm/vllm-openai:<PINNED_TAG>

%post
    pip install --no-cache-dir datasets pandas pyyaml tqdm

%environment
    export HF_HOME=/scratch/huggingface
```

Do not start here unless necessary. Custom containers increase debugging cost.

#### How the current pipeline references the container

The pipeline references the container through one variable in the Slurm script:

```bash
CONTAINER=$SCRATCH/containers/nemotron_vllm.sif
```

Then the Slurm script runs commands inside it:

```bash
apptainer exec --nv \
  --bind $PROJECT_DIR:/workspace \
  --bind $SCRATCH:$SCRATCH \
  $CONTAINER \
  bash -lc "
    cd /workspace
    python3 scripts/check_env.py
    python3 scripts/baseline_generate.py --config configs/baseline_h200.yaml
  "
```

In other words, the container is not imported by Python and not listed inside the YAML config. It is referenced by the **Slurm execution layer**.

Recommended separation:

```text
slurm/run_baseline.slurm       contains container path and HPC resource request
configs/baseline_h200.yaml     contains model/generation/input/output settings
scripts/baseline_generate.py   contains inference logic
```

#### Minimal container provenance checklist

Before trusting a container, record (a fillable template lives in `docs/hpc_setup_notes.md`):

```text
Container source URI:
Container tag or digest:
SIF path:
Build/pull date:
HPC node type:
NVIDIA driver version:
CUDA version visible in container:
Python version:
PyTorch version:
vLLM version:
Transformers version:
Smoke test result:
Known issues:
```

Example command to collect part of this information:

```bash
apptainer exec --nv $CONTAINER bash -lc "
echo '=== Python ==='
python --version

echo '=== NVIDIA ==='
nvidia-smi

echo '=== Packages ==='
python - <<'PY'
import importlib.metadata as md

for pkg in ['torch', 'vllm', 'transformers', 'accelerate', 'huggingface_hub']:
    try:
        print(pkg, md.version(pkg))
    except md.PackageNotFoundError:
        print(pkg, 'NOT INSTALLED')
PY
"
```

Before using a container for model execution, test:

```bash
apptainer exec --nv $SCRATCH/containers/nemotron_vllm.sif nvidia-smi
```

---

## 8. Smoke Test

Before running the full baseline, run a small smoke test.

### 8.1 `scripts/check_env.py`

Example:

```python
import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
else:
    raise RuntimeError("CUDA is not available. Check Slurm GPU allocation and Apptainer --nv.")
```

### 8.2 `slurm/smoke_test.slurm`

Example:

```bash
#!/bin/bash
#SBATCH --job-name=nemotron_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

PROJECT_DIR=$HOME/Nemotron_Challenge
CONTAINER=$SCRATCH/containers/nemotron_vllm.sif

export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub

apptainer exec --nv \
  --bind $PROJECT_DIR:/workspace \
  --bind $SCRATCH:$SCRATCH \
  $CONTAINER \
  bash -lc "
    cd /workspace
    python3 scripts/check_env.py
  "
```

Submit:

```bash
sbatch slurm/smoke_test.slurm
```

Check status:

```bash
squeue -u $USER
```

Read logs:

```bash
cat logs/nemotron_smoke_<JOB_ID>.out
cat logs/nemotron_smoke_<JOB_ID>.err
```

---

### 8.3 Recommended Smoke Test Scope

The smoke test should not be only `nvidia-smi`. It should verify the full path from Slurm allocation to container runtime to Python import to model-level generation.

Use a staged smoke test so failures are easy to locate. Stages 0 and 1 run in `slurm/smoke_test.slurm` (host side); stages 2–5 run inside `scripts/smoke_test.py`; stage 6 reuses `scripts/baseline_generate.py` with `configs/smoke_h200.yaml`.

#### Stage 0: Slurm allocation check

Goal:

```text
Confirm the job is running on a compute node with the requested GPU.
```

Checks:

```bash
hostname
echo $SLURM_JOB_ID
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
```

Success means:

```text
Slurm gave the job a GPU allocation.
The job is not accidentally running on the login node.
The NVIDIA driver can see the H200.
```

Failure usually means:

```text
wrong partition
missing --gres=gpu:1
OOD shell used directly instead of sbatch/salloc
GPU node unavailable
```

#### Stage 1: Apptainer GPU passthrough check

Goal:

```text
Confirm the container can see the same GPU through Apptainer.
```

Command:

```bash
apptainer exec --nv $CONTAINER nvidia-smi
```

Success means:

```text
Apptainer --nv is working.
The host driver is visible inside the container.
```

Failure usually means:

```text
forgot --nv
container runtime issue
host driver/library path problem
container incompatible with cluster GPU setup
```

#### Stage 2: Python package check

Goal:

```text
Confirm the container has the expected Python inference stack.
```

Inside `scripts/smoke_test.py`:

```python
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu name:', torch.cuda.get_device_name(0))
    print('bf16 supported:', torch.cuda.is_bf16_supported())

import transformers
print('transformers:', transformers.__version__)

import vllm
print('vllm:', vllm.__version__)
```

Success means Python, PyTorch, Transformers, and vLLM are usable inside the container. Failure usually means wrong container, missing package, or CUDA/PyTorch mismatch.

#### Stage 3: Hugging Face metadata access

Goal:

```text
Confirm the job can reach Hugging Face metadata or use existing authentication/cache.
```

Recommended environment variables (set in the Slurm script):

```bash
export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub
```

Minimal check (no weight download):

```python
from huggingface_hub import model_info
info = model_info('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')
print('model id:', info.modelId)
print('sha:', info.sha)
```

Failure usually means missing HF token, no internet on the compute node, model gating not accepted, or the cache path was not bind-mounted into the container.

If compute nodes cannot access the internet, pre-download the model to `$SCRATCH/huggingface` from an allowed node first.

#### Stage 4: Tokenizer + chat template check

Goal:

```text
Confirm the model tokenizer can be loaded before loading full weights.
```

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    trust_remote_code=True,
)
messages = [{'role': 'user', 'content': 'Answer in one sentence: what is 2 + 2?'}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text[:500])
```

Failure usually means missing `trust_remote_code`, inaccessible model metadata, or an incompatible Transformers version.

#### Stage 5: Single-prompt model load and generation

Goal:

```text
Confirm the model can generate one short response.
```

Use small runtime settings first:

```text
max_model_len: 8192
max_tokens: 64
temperature: 0.0
gpu_memory_utilization: 0.80
```

Expected behavior: the first run may take a long time because model weights are downloaded and loaded; the second run should be much faster if `$SCRATCH/huggingface` is persistent.

Failure usually means GPU OOM, vLLM unsupported version, model architecture support issue, CUDA kernel issue, or a cache/download problem.

#### Stage 6: Five-prompt mini baseline

Goal:

```text
Confirm the baseline loop, error handling, and JSONL output format.
```

Driver: `scripts/baseline_generate.py` with `configs/smoke_h200.yaml`.

Input: `data/sample_prompts_5.jsonl`
Expected output: `outputs/smoke_predictions_<JOB_ID>.jsonl`

Checks:

```bash
wc -l outputs/smoke_predictions_<JOB_ID>.jsonl
head -n 1 outputs/smoke_predictions_<JOB_ID>.jsonl
tail -n 1 outputs/smoke_predictions_<JOB_ID>.jsonl
```

Success means the model can process multiple prompts, the output file is valid JSONL, and per-prompt errors are captured instead of crashing the whole run.

Only after this stage should the repo move to 20, 100, or full challenge-scale prompts.

#### Smoke Test Pass/Fail Criteria

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

A partial pass should be documented in `docs/hpc_setup_notes.md` using the partial-pass log template there.

---

## 9. Baseline Run Illustration

The baseline run is designed to show what happens from input prompt to saved output.

### 9.1 Input Format

Example `data/sample_prompts_5.jsonl`:

```jsonl
{"id": "sample_001", "prompt": "Solve the following reasoning problem step by step: If Alice has 3 boxes and each box contains 4 red balls and 2 blue balls, how many balls are there in total?"}
{"id": "sample_002", "prompt": "A train leaves at 3 PM traveling 60 mph. Another leaves at 4 PM traveling 90 mph on the same route. When does the second train catch the first?"}
{"id": "sample_003", "prompt": "Classify whether the statement is logically valid and explain briefly: All cats are mammals. Luna is a cat. Therefore, Luna is a mammal."}
```

Each line is one independent prompt.

---

### 9.2 Baseline Logic

The baseline script should:

```text
1. Load config.
2. Read input JSONL prompts.
3. Start or connect to inference backend.
4. Send each prompt to Nemotron.
5. Record generated answer.
6. Record latency and metadata.
7. Save one JSON object per prompt.
8. Continue even if one sample fails.
```

Conceptual flow:

```text
input JSONL
   |
   v
baseline_generate.py
   |
   v
vLLM / Transformers backend
   |
   v
Nemotron 3 Nano generation
   |
   v
output JSONL
```

---

### 9.3 Example Output Format

Example `outputs/predictions_<JOB_ID>.jsonl`:

```jsonl
{"id": "sample_001", "prompt": "...", "response": "...", "latency_sec": 12.84, "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "backend": "vllm", "error": null}
{"id": "sample_002", "prompt": "...", "response": "...", "latency_sec": 18.21, "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "backend": "vllm", "error": null}
{"id": "sample_003", "prompt": "...", "response": null, "latency_sec": null, "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "backend": "vllm", "error": "RuntimeError: ..."}
```

The key idea is that the baseline run should be debuggable, even when some prompts fail.

---

## 10. Baseline Configuration

Example `configs/baseline_h200.yaml`:

```yaml
model:
  name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  dtype: bfloat16
  max_model_len: 32768
  max_tokens: 1024
  temperature: 0.0
  top_p: 1.0

runtime:
  backend: vllm
  gpu_memory_utilization: 0.90
  tensor_parallel_size: 1

data:
  input_path: data/sample_prompts_5.jsonl
  output_dir: outputs

logging:
  save_prompt: true
  save_latency: true
  save_errors: true
```

Why start with `max_model_len: 32768`?

Nemotron supports long-context usage, but very long context increases KV-cache memory pressure. For the first baseline, use a smaller context length such as 8k or 32k. Increase only after the smoke test and small baseline succeed.

---

### 10.1 Relationship Between Config and Container

The baseline config does not replace the container.

The config answers:

```text
Which model?
Which dtype?
How long is the context?
How many tokens to generate?
Where are the input/output files?
```

The container answers:

```text
Which Python?
Which CUDA runtime?
Which PyTorch?
Which vLLM?
Which Transformers?
Which compiled kernels?
```

Therefore, changing `configs/baseline_h200.yaml` changes the experiment but does not change the installed software. To change the software stack, change the Apptainer image referenced in the Slurm script.

Recommended convention:

```bash
# slurm/run_baseline.slurm
CONTAINER=$SCRATCH/containers/nemotron_vllm.sif
CONFIG=configs/baseline_h200.yaml
```

Then execute:

```bash
apptainer exec --nv \
  --bind $PROJECT_DIR:/workspace \
  --bind $SCRATCH:$SCRATCH \
  $CONTAINER \
  bash -lc "
    cd /workspace
    python3 scripts/baseline_generate.py --config $CONFIG
  "
```

This makes the provenance clear:

```text
Slurm file = resource + container provenance
YAML file  = experiment configuration
Python     = baseline logic
Output     = generated predictions and metrics
```

---

## 11. Example vLLM Baseline Slurm Script

Example `slurm/run_baseline.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=nemotron_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"

nvidia-smi

PROJECT_DIR=$HOME/Nemotron_Challenge
CONTAINER=$SCRATCH/containers/nemotron_vllm.sif

export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn

apptainer exec --nv \
  --bind $PROJECT_DIR:/workspace \
  --bind $SCRATCH:$SCRATCH \
  $CONTAINER \
  bash -lc "
    cd /workspace

    python3 scripts/check_env.py

    python3 scripts/baseline_generate.py \
      --config configs/baseline_h200.yaml \
      --output outputs/predictions_${SLURM_JOB_ID}.jsonl
  "

echo "Finished at: $(date)"
```

Submit:

```bash
sbatch slurm/run_baseline.slurm
```

---

## 12. Baseline Interpretation

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

## 13. Development Workflow

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

## 14. Common Problems

### Problem: `CUDA is not available`

Possible causes:

- Job was run on login node instead of compute node.
- Slurm script did not request GPU.
- Apptainer command missed `--nv`.
- Container CUDA/PyTorch stack is incompatible with the host driver.

Check:

```bash
nvidia-smi
apptainer exec --nv $SCRATCH/containers/nemotron_vllm.sif nvidia-smi
```

---

### Problem: Out of Memory

Possible causes:

- Context length too large.
- Batch size too high.
- `gpu_memory_utilization` too high.
- BF16 model path too memory-heavy for current settings.
- vLLM KV cache allocation too aggressive.

Try:

```yaml
model:
  max_model_len: 8192
  max_tokens: 512

runtime:
  gpu_memory_utilization: 0.80
```

---

### Problem: Hugging Face download is slow or fails

Possible causes:

- Cache path is not persistent.
- HF token is missing.
- Dataset or model terms were not accepted.
- Login node has internet but compute node does not, or vice versa.

Recommended:

```bash
export HF_HOME=$SCRATCH/huggingface
huggingface-cli login
```

If compute nodes have no internet, download/cache the model in an allowed environment first.

---

### Problem: Apptainer image cannot be built on HPC

Some clusters allow:

```bash
apptainer exec image.sif ...
```

but block:

```bash
apptainer build image.sif ...
```

Ask HPC support:

```text
Can I build Apptainer images on the cluster, or only run existing .sif images?
Is there a recommended CUDA/PyTorch/vLLM/NeMo image for H200 nodes?
```

---

## 15. Future Optimization Directions

After the baseline is stable, possible next steps include:

1. Prompt format tuning
2. Reasoning-on vs reasoning-off comparison
3. Temperature/top-p sweep
4. Self-consistency sampling
5. Tool-use evaluation for math/coding prompts
6. Output parser and verifier
7. LoRA/SFT experiment if training resources and competition rules allow
8. NeMo Evaluator Launcher integration
9. Long-context setting comparison
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
