# HPC Setup Notes

Operational notes specific to running this project on the cluster.
Populate this file as the team encounters cluster-specific quirks.

## Storage

| Path | Purpose |
|---|---|
| `$HOME/Nemotron_Challenge/` | Code repo |
| `$SCRATCH/containers/` | Apptainer `.sif` images |
| `$SCRATCH/huggingface/` | Hugging Face cache |
| `$SCRATCH/nemotron_outputs/` | Large outputs (optional symlink target) |

## Environment Variables

Set inside Slurm scripts (preferred) rather than `~/.bashrc`, so the run
is self-contained and reproducible:

```bash
export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export HF_HUB_CACHE=$SCRATCH/huggingface/hub
```

## Container Provenance

The pipeline references a container in exactly one place:

```bash
# slurm/run_baseline.slurm and slurm/smoke_test.slurm
CONTAINER=$SCRATCH/containers/nemotron_vllm.sif
```

The YAML config does **not** describe the software stack. Changing
`configs/baseline_h200.yaml` changes the experiment; changing the
`.sif` referenced in the Slurm script is what changes Python / CUDA /
PyTorch / vLLM / Transformers versions.

### Source priority (use the highest available)

| Priority | Source | When to use |
|---|---|---|
| 1 | HPC-admin-provided vLLM/PyTorch/NeMo Apptainer image | If one exists, prefer it — most likely matches host driver, CUDA, and partition policy |
| 2 | Official vLLM Docker image converted to `.sif` (`apptainer pull`) | Best self-managed baseline |
| 3 | NVIDIA NGC / NeMo Framework container converted to `.sif` | If the workflow depends on NeMo Evaluator or NVIDIA tooling |
| 4 | Custom Apptainer definition file | Only after a baseline already works |
| 5 | Manually patched conda/Jupyter env | Avoid for the main run |

### Pulling an official vLLM image (Option 2)

```bash
mkdir -p $SCRATCH/containers
cd $SCRATCH/containers
apptainer pull nemotron_vllm.sif docker://vllm/vllm-openai:<PINNED_TAG>
```

Do **not** keep `:latest` long-term. After the first successful smoke
test, pin a specific tag (or digest) and record it in the provenance
log below.

### Verify GPU passthrough before any model run

```bash
apptainer exec --nv $SCRATCH/containers/nemotron_vllm.sif nvidia-smi
```

### Provenance log (fill in after each new image)

| Field | Value |
|---|---|
| Container source URI | `docker://...` or admin path |
| Tag / digest | `vX.Y.Z` or `sha256:...` |
| SIF path | `$SCRATCH/containers/nemotron_vllm.sif` |
| Build / pull date | `YYYY-MM-DD` |
| HPC node type | H200 |
| NVIDIA driver version | (from `nvidia-smi`) |
| CUDA version inside container | (from `nvidia-smi` inside container) |
| Python version | |
| PyTorch version | |
| vLLM version | |
| Transformers version | |
| Smoke test result | pass / partial / fail |
| Known issues | |

Helper command to dump most of the version info:

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

## Smoke Test

The pipeline ships a 7-stage smoke test (see README §8.3 for the full
rationale):

| Stage | Where it runs | What it verifies |
|---|---|---|
| 0 | `slurm/smoke_test.slurm` (host) | Slurm allocated a GPU node |
| 1 | `slurm/smoke_test.slurm` (host) | Apptainer `--nv` exposes the GPU |
| 2 | `scripts/smoke_test.py` | torch / transformers / vllm import |
| 3 | `scripts/smoke_test.py` | HF metadata reachable (no weight download) |
| 4 | `scripts/smoke_test.py` | Tokenizer + chat template load |
| 5 | `scripts/smoke_test.py` | Single-prompt model load + generation |
| 6 | `scripts/baseline_generate.py` w/ `configs/smoke_h200.yaml` | 5-prompt JSONL baseline |

A partial pass is still useful — record which stages passed and where
the failure was so the next attempt can target the right layer.

### Partial-pass log template

```text
Date: YYYY-MM-DD
Node type: H200
Container: $SCRATCH/containers/nemotron_vllm.sif (tag: <PINNED_TAG>)
Result: partial pass
Passed stages: 0, 1, 2, 3, 4
Failed stage: 5
Failure: CUDA OOM at max_model_len=8192
Next action: retry with max_model_len=4096, gpu_memory_utilization=0.70
```

## Slurm Cheat-Sheet

```bash
sbatch slurm/smoke_test.slurm       # submit
squeue -u $USER                     # status
scancel <JOB_ID>                    # cancel
```

Logs land in `logs/<job_name>_<JOB_ID>.out` and `.err`.

## Open Questions

Track here as we hit them:

- [ ] Can we build `.sif` images on the cluster, or only run prebuilt ones?
- [ ] Do compute nodes have outbound internet for HF downloads, or must
      models be pre-cached from a login node?
- [ ] Recommended NGC image for H200 + vLLM?
- [ ] Per-user `$SCRATCH` quota and retention policy?
