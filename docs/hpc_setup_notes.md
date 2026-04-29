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

**Container pull.** `nemotron_vllm.sif` (~7–8 GB) is not yet present at `$SCRATCH/containers/`. The first attempt to pull from the login node was OOM-killed at the `mksquashfs` step because the login-node `/tmp` is small and mounted `nodev`. The fix is to run the pull inside a Slurm allocation with `APPTAINER_TMPDIR=$SCRATCH/apptainer/tmp` (which is already exported in `~/.bashrc`). A CPU partition is enough — see Step A below.

The slurm scripts also still reference a non-existent `h200` partition; once the container is in place, they need a one-line edit to `--partition=gpu --gres=gpu:h200:1` (see Step B).

### Resolved: H200 access (2026-04-27)

We initially believed the account was rejected from `--partition=h200` for quota reasons. **It was a syntax mistake, not a quota issue.** Explorer has no `h200` partition; H200 GPUs are exposed as a `--gres` value on the `gpu` / `gpu-short` / `gpu-interactive` partitions. Verified by submitting a 5-minute probe:

```bash
sbatch --partition=gpu --gres=gpu:h200:1 --time=00:05:00 --mem=16G \
       --output=logs/h200_probe_%j.out --wrap="nvidia-smi"
# Job 6371194 — single H200 allocated, 143771 MiB VRAM, driver 570.86.15, CUDA 12.8
```

Lesson for next time: when a "partition" can't be found in `sinfo`, check whether the resource is actually a `--gres` value on an existing partition (use `sinfo -p <partition> -N -o "%N %P %G %m"`) before assuming a quota or permission problem.

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

**Step B — request a GPU node with enough VRAM.** The smoke test (Stage 5) and the baseline both need a GPU that can hold Nemotron 3 Nano in BF16 (~60 GB).

Important: **`h200` is not a Slurm partition on Explorer.** Earlier attempts to submit `--partition=h200` failed because the resource is requested via `--gres=gpu:h200:N` on the `gpu` partition (or `gpu-short` / `gpu-interactive` for ≤2 h jobs). The fix is the request syntax, not a quota ticket.

#### Relevant GPU nodes on Explorer

Surveyed 2026-04-27 with `sinfo -p gpu,gpu-short,gpu-interactive,courses-gpu -N -o "%N %P %G %m"`.

| Nodes | GPU | Count/node | Per-GPU VRAM | System RAM | Partitions |
|---|---|---|---|---|---|
| d4052, d4053, d4054, d4055 | H200 | 8 | 141 GB | 512 GB | gpu, gpu-short, gpu-interactive |
| d1028, d1029 | A100 | 4 | 80 GB | 512 GB | gpu, gpu-short, gpu-interactive |
| d1026 | A100 | 3 | 80 GB | 512 GB | gpu, gpu-short, gpu-interactive |

No other single GPU on the cluster has ≥80 GB VRAM (V100 PCIe / V100 SXM2 max at 32 GB, P100 and T4 at 16 GB). **Multi-GPU aggregate fallback** if H200 *and* A100 are both unavailable: V100-SXM2 nodes (d1002, d1007, d1009–d1013, d1015, d1017, d1019, d1020, d1022, d1027) have 4×32 GB = 128 GB aggregate, usable with `tensor_parallel_size=4`.

#### Interactive request (no real queuing)

`salloc` either grants the node promptly or makes you wait at the prompt — use it when you need a shell on the node *now*:

```bash
# H200 — primary target
salloc --partition=gpu --gres=gpu:h200:1 --cpus-per-task=8 --mem=64G --time=02:00:00

# A100 — backup (tight at 80 GB; use gpu_memory_utilization=0.70, max_model_len=4096)
salloc --partition=gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=02:00:00

# V100 SXM2 multi-GPU fallback (set tensor_parallel_size=4 in the YAML)
salloc --partition=gpu --gres=gpu:v100-sxm2:4 --cpus-per-task=8 --mem=64G --time=02:00:00
```

#### Queued (batch) request — recommended when H200 is busy

`sbatch` queues until resources are free. This is the same mechanism OOD's "VSCode session" form uses under the hood, so submitting from the terminal gives the same wait behavior:

```bash
# Probe job: 5-minute nvidia-smi to confirm the allocation chain works
sbatch --partition=gpu --gres=gpu:h200:1 --time=00:05:00 --mem=16G \
       --output=logs/h200_probe_%j.out --wrap="nvidia-smi"

squeue -u $USER                                            # watch queue position
sacct -j <JOB_ID> --format=JobID,State,Reason,Elapsed       # detail
cat logs/h200_probe_<JOB_ID>.out                            # should list an H200
```

Possible outcomes:

- **Runs successfully** → use `sbatch` from the terminal going forward; OOD is not needed.
- **Sits in queue with reason `Priority` or `Resources`** → normal, just wait.
- **Rejected immediately with `Invalid gres` / `QOS limit` / `Access denied`** → that account is genuinely restricted from H200; retry with `--gres=gpu:a100:1` and open a ticket about H200 access.

#### After a successful probe

Edit the submission scripts to point at the resource that worked:

```bash
# slurm/smoke_test.slurm and slurm/run_baseline.slurm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1     # or gpu:a100:1 if falling back
```

If falling back to A100, also rename `configs/baseline_h200.yaml` → `configs/baseline_a100.yaml` and tighten its memory settings (`gpu_memory_utilization=0.70`, `max_model_len=4096`).

### Future steps once the blocker is resolved

The end-to-end sequence after the container is pulled and a GPU partition is confirmed:

```bash
cd $HOME/Nemotron_Challenge
git pull                                  # latest code
mkdir -p logs outputs

# 1. Confirm the tracked smoke prompt fixture is present.
ls -l data/sample_prompts_5.jsonl
# This smoke prompt fixture is now tracked in git; if it is missing after
# `git pull`, the local checkout is stale or incomplete.

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
cat logs/nemotron_smoke_6382411.out
cat logs/nemotron_smoke_6382411.err

# 6. Verify Stage 6 output and run the evaluator
wc -l outputs/smoke_predictions_6375635.jsonl
apptainer exec --nv $SCRATCH/containers/nemotron_vllm.sif \
  python3 scripts/evaluate.py --predictions outputs/smoke_predictions_<JOB_ID>.jsonl

# 7. Prepare the AIME25 dataset (one-time, login node — needs internet)
#    Writes data/aime25.jsonl (30 problems) using MathArena/aime_2025
#    with the NeMo-Skills boxed-answer prompt template baked in.
#    Pure-stdlib (urllib + json), so no container or extra packages
#    needed; runs straight on the login node.
python3 scripts/prepare_aime25.py
wc -l data/aime25.jsonl   # expect 30

# 8. Only after smoke passes and aime25.jsonl exists — submit the baseline
sbatch slurm/run_baseline.slurm
squeue -u $USER
cat logs/nemotron_baseline_<JOB_ID>.out
apptainer exec --nv $SCRATCH/containers/nemotron_vllm.sif \
  python3 scripts/evaluate.py --predictions outputs/predictions_<JOB_ID>.jsonl
```
# baseline log check
ls -lt logs/ | head
cat logs/nemotron_baseline_6382411.out
cat logs/nemotron_baseline_6382411.err

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
- [x] What is the H200 access situation? → Confirmed working 2026-04-27 (job 6371194). Earlier failure was the partition-vs-gres syntax mistake (`--partition=h200` doesn't exist); use `--partition=gpu --gres=gpu:h200:N`. No quota ticket needed.
- [x] Which GPU partition fits Nemotron 3 Nano BF16 on a single GPU? → H200 (141 GB) on the `gpu` partition — 4 nodes (d4052–d4055), 8 GPUs each. A100 80 GB on d1026 / d1028 / d1029 is the backup.
- [ ] Do compute nodes have outbound internet for HF downloads, or must models be pre-cached from a login node?
- [ ] Recommended NGC image for H200 + vLLM (still useful as a fallback to the `vllm/vllm-openai` image)?
- [ ] Per-user `$SCRATCH` quota and retention policy?
