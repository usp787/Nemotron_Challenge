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
apptainer exec --nv container.sif python script.py
```

The `--nv` flag exposes NVIDIA GPUs to the container.

---

## 5. Suggested Repository Structure

```text
nemotron-challenge/
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
$HOME/nemotron-challenge/          # code repo
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
git clone git@github.com:<YOUR_USERNAME>/nemotron-challenge.git
cd nemotron-challenge
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

### 7.3 Container Location

Expected container path:

```text
$SCRATCH/containers/nemotron_vllm.sif
```

This file is not committed to GitHub.

If the cluster allows pulling Docker images through Apptainer, one possible starting point is a vLLM OpenAI image. If the cluster has NVIDIA NGC/NeMo images available, prefer the admin-recommended image.

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
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

PROJECT_DIR=$HOME/nemotron-challenge
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
    python scripts/check_env.py
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

## 11. Example vLLM Baseline Slurm Script

Example `slurm/run_baseline.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=nemotron_baseline
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
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

PROJECT_DIR=$HOME/nemotron-challenge
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

    python scripts/check_env.py

    python scripts/baseline_generate.py \
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
git clone git@github.com:<YOUR_USERNAME>/nemotron-challenge.git
cd nemotron-challenge
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
cd $HOME/nemotron-challenge
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
