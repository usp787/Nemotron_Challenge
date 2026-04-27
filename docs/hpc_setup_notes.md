# HPC Setup Notes

Operational notes specific to running this project on the cluster.
Populate this file as the team encounters cluster-specific quirks.

## Bring-up Status (Northeastern Explorer cluster)

Last updated: 2026-04-27. Login nodes seen: `explorer-01`, `explorer-02`.

### Confirmed cluster facts

| Item | Value / Notes |
|---|---|
| Scratch path | `/scratch/$USER` (e.g. `/scratch/zha.j`). `$SCRATCH` is **not** set by default — must be exported manually. |
| Apptainer | Available on the path; no `module load` required. |
| Login-node `/tmp` | Mounted with `nodev` and small. Apptainer build/squash spills there by default and gets OOM-killed. Workaround: redirect `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR` into `$SCRATCH`. |
| GitHub auth from HPC | SSH keys are not pre-provisioned. Generate `~/.ssh/id_ed25519` on the cluster and add the public key to GitHub before `git clone git@github.com:...`. |

### What is set up

- Repo cloned at `$HOME/Nemotron_Challenge`.
- `$SCRATCH=/scratch/$USER` exported in `~/.bashrc`.
- Directories created: `$SCRATCH/containers`, `$SCRATCH/huggingface/hub`, `$SCRATCH/apptainer/{tmp,cache}`.
- `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR` exported in `~/.bashrc` to avoid login-node `/tmp`.

### Current blocker

**H200 partition quota issue** — submitting `--partition=h200` jobs (whether `salloc` or `sbatch`) is not currently usable for this account. This blocks two things:

1. The remaining `apptainer pull` of `vllm/vllm-openai:v0.12.0` if we run it on a compute node specifically through the `h200` partition.
2. All smoke-test and baseline runs as written, because [slurm/smoke_test.slurm](../slurm/smoke_test.slurm) and [slurm/run_baseline.slurm](../slurm/run_baseline.slurm) hard-code `#SBATCH --partition=h200`.

The first earlier attempt to pull from the login node was OOM-killed at the `mksquashfs` step (the `signal: killed` failure), which is why the workaround is "do the pull inside an allocation."

### Working around the blocker

**Step A — finish the container pull on any non-GPU partition.** The `apptainer pull` step does not need a GPU; it just needs RAM and disk. Use `sinfo -s` to list partitions, then run the pull on something like `general` / `compute` / `cpu`:

```bash
# Discover available partitions and their limits
sinfo -s
sinfo -o "%P %l %D %m" | head

# Allocate a non-GPU node with enough memory for mksquashfs
salloc --partition=<cpu_partition> --cpus-per-task=4 --mem=32G --time=01:00:00

# Once on the compute node, confirm env vars carried over and pull
echo "TMP=$APPTAINER_TMPDIR  CACHE=$APPTAINER_CACHEDIR"
cd $SCRATCH/containers
apptainer pull nemotron_vllm.sif docker://vllm/vllm-openai:v0.12.0
ls -lh nemotron_vllm.sif    # expect ~7-8 GB
exit
```

**Step B — figure out the right GPU partition for inference.** The smoke test (Stage 5) and the baseline both need a GPU with enough VRAM to hold Nemotron 3 Nano in BF16 (~60 GB). Options, in order of preference:

1. **Restore H200 access.** Open a ticket with Explorer support ("my account cannot submit to `h200`; please advise on quota/account configuration"). H200 (141 GB) is the most comfortable fit and is what the slurm scripts assume.
2. **Use an A100 80 GB or H100 80 GB partition** if available. Nemotron 3 Nano BF16 fits on 80 GB but is tight — start the smoke test with `gpu_memory_utilization=0.70` and `max_model_len=4096` to leave room for KV cache.
3. **Multi-GPU tensor parallel.** If only smaller GPUs are available (e.g. 2× 40 GB A100), set `tensor_parallel_size: 2` in the YAML config and request `--gres=gpu:2`. Adds complexity; only use if (1) and (2) aren't options.
4. **Quantize.** Switch to a quantized Nemotron checkpoint (FP8/AWQ if NVIDIA publishes one) — but that changes the experiment. Avoid for the baseline.

After identifying the right partition (call it `<gpu_partition>`), the slurm files need a one-line edit each:

```bash
# slurm/smoke_test.slurm and slurm/run_baseline.slurm
#SBATCH --partition=<gpu_partition>
```

If the new partition's GPUs aren't H200, also rename `configs/baseline_h200.yaml` → `configs/baseline_<gpu_name>.yaml` for honesty (and update the `--config` path in the slurm script).

### Future steps once the blocker is resolved

The end-to-end sequence after the container is pulled and a GPU partition is confirmed:

```bash
cd $HOME/Nemotron_Challenge
git pull                                  # latest code
mkdir -p logs outputs

# 1. Recreate the gitignored sample prompts (5 lines, format in README §9.1)
#    — copy from a teammate or hand-write into data/sample_prompts_5.jsonl.
ls -l data/sample_prompts_5.jsonl

# 2. Hugging Face authentication if Nemotron is gated (one-time)
huggingface-cli login

# 3. Stage 0/1 sanity check on a real GPU node
salloc --partition=<gpu_partition> --gres=gpu:1 --time=00:10:00
apptainer exec --nv $SCRATCH/containers/nemotron_vllm.sif nvidia-smi
exit

# 4. Submit the staged smoke test (stages 0-6)
sbatch slurm/smoke_test.slurm
squeue -u $USER

# 5. After the smoke job finishes, inspect logs
ls -lt logs/ | head
cat logs/nemotron_smoke_<JOB_ID>.out
cat logs/nemotron_smoke_<JOB_ID>.err

# 6. Verify Stage 6 output and run the evaluator
wc -l outputs/smoke_predictions_<JOB_ID>.jsonl
apptainer exec --nv $SCRATCH/containers/nemotron_vllm.sif \
  python scripts/evaluate.py --predictions outputs/smoke_predictions_<JOB_ID>.jsonl

# 7. Only after smoke passes — submit the full baseline
sbatch slurm/run_baseline.slurm
squeue -u $USER
cat logs/nemotron_baseline_<JOB_ID>.out
apptainer exec --nv $SCRATCH/containers/nemotron_vllm.sif \
  python scripts/evaluate.py --predictions outputs/predictions_<JOB_ID>.jsonl
```

Useful while waiting:

```bash
squeue -u $USER                                           # all your jobs
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -f logs/nemotron_baseline_<JOB_ID>.out               # live tail
scancel <JOB_ID>                                          # kill one
```

---

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

- [x] Where is scratch? → `/scratch/$USER` (must export `$SCRATCH` manually).
- [x] Can we build `.sif` images on the cluster? → Yes, but **not** on the login node (`/tmp` is small + `nodev`; `mksquashfs` gets OOM-killed). Run inside an allocation with `APPTAINER_TMPDIR=$SCRATCH/...`.
- [ ] What is the H200 quota / account configuration for this user, and how do we restore access? (Open ticket with Explorer support.)
- [ ] If H200 stays unavailable, which GPU partition do we use as the supported alternative, and does it have enough VRAM for Nemotron 30B BF16 on a single GPU?
- [ ] Do compute nodes have outbound internet for HF downloads, or must models be pre-cached from a login node?
- [ ] Recommended NGC image for H200 + vLLM (still useful as a fallback to the `vllm/vllm-openai` image)?
- [ ] Per-user `$SCRATCH` quota and retention policy?
