# Path β — Chunked Production Training Runbook

This doc describes how the Phase 6 production training run is executed
on Explorer's `gpu` partition. It covers the design, the per-chunk
operational commands, expected log signals, and recovery procedures.

## Why path β exists

Smoke job `6710260` measured **~230 sec per optimizer step** on a single
H200, fitting the THK 04-10-04-33 recipe (LoRA r=32, effective batch 64,
linear LR over a 271-step "full epoch" basis).

| Step target | Train time at 230 s/step |
|---|---|
| 245 (THK natural stop) | ~15.7 h |
| 271 (one full epoch) | ~17.4 h |

The `gpu` partition's hard `TIMELIMIT` is **8 hours** (`sinfo -p gpu -o "%P %l %D %G"`),
so a single uninterrupted run is impossible. Path β splits the work into
**3 SLURM jobs of ~82 optimizer steps each, totalling 246 cumulative steps**
(≈ THK's 245), with full state preserved between jobs.

| Chunk | Cumulative steps | Train time (chunk) | SLURM walltime request | Output dir |
|---|---|---|---|---|
| 1 | 1 → 82 | ~5.2 h | 7:00:00 | `outputs/lora_adapter_thk_chunk1` |
| 2 | 83 → 164 | ~5.2 h | 7:00:00 | `outputs/lora_adapter_thk_chunk2` |
| 3 | 165 → 246 | ~5.2 h | 7:00:00 | `outputs/lora_adapter_thk_chunk3` |

Each chunk has ~1.5 h of safety margin under the 8h cap.

## How the chunked training works

### What persists between chunks

`scripts/train_lora.py` was extended with three CLI flags (the rest of the
trainer is unchanged, so the smoke + lora_verification invocations remain
backward-compatible):

| Flag | Purpose |
|---|---|
| `--resume-from-checkpoint <dir>` | Load adapter weights + AdamW state + LR schedule state + sampler position from the previous chunk's output dir. |
| `--max-steps <N>` | Cap the **cumulative** `global_step` at `N` for this process. Each chunk's value is higher than the previous (82 / 164 / 246). |
| `--output-dir <path>` | Override the YAML's `output.adapter_dir` so each chunk writes to its own dir without clobbering. |

At end-of-chunk, the trainer writes (in `output-dir`):

- `adapter_config.json` + `adapter_model.safetensors` — the PEFT LoRA weights (~3.3 GB).
- `training_state.pt` — a Python pickle (~7 GB) containing:
  - `optimizer` — AdamW momentum (m) and variance (v) per trainable parameter.
  - `scheduler` — the LR scheduler's `last_epoch` (= cumulative global_step).
  - `global_step` — cumulative optimizer steps so far.
  - `epoch_in_progress` — which outer-epoch the trainer was iterating.
  - `samples_consumed_in_epoch` — how many micro-batches were pulled from
    this epoch's deterministic shuffle. The next chunk's data loader
    fast-forwards past these to maintain THK's **continuous-shuffle**
    semantics (each sample seen at most once across all 3 chunks).
  - `seed` — for diagnostic purposes only.

Per-chunk disk: ~10.5 GB. Three chunks plus their `_<jobid>` archives
consume ~30 GB peak on the `outputs/` directory.

### Why this preserves THK's recipe

| Concern | How path β handles it |
|---|---|
| LR schedule continuity | Scheduler state is saved + restored. The linear decay continues smoothly across chunk boundaries — chunk 2 starts at LR=1.39e-4 (continuing chunk 1's decay), not at the 2e-4 peak. |
| Optimizer momentum | AdamW's `state_dict` (m, v per param) is saved + restored. Chunks don't start with zero-momentum gradients. |
| Sample order | Each chunk continues consuming the SAME epoch's deterministic shuffle (seed=42, epoch=0) at the position the previous chunk left off. No sample is seen twice; ~10% of the dataset is never seen (matching THK's 245-of-271 partial epoch). |
| Final-step LR | Step 246 lands at LR=1.85e-5 (= 2e-4 × (25/271) on the linear decay) — close to but not at zero. Matches THK's stopping point. |

The only material deviation from a single uninterrupted run is the
~5 minute setup overhead paid per chunk (Stages 0-3 + model load +
re-tokenization). Total setup overhead across 3 chunks ≈ 15 min,
negligible against ~15.7 h of training time.

## Submission sequence

**Do NOT chain via `sbatch --dependency`.** This is a hard project rule
(memory `feedback_cluster_workflow.md`): submit each chunk by hand after
inspecting the previous chunk's log. The intent is that you have a
chance to catch a recipe drift, an OOM, or an unexpected loss spike
before propagating it through 3 chunks of training.

### Before chunk 1: confirm the path-β code is on the cluster

The path-β patches live in your git repo:

- `scripts/train_lora.py` — resume + state save logic.
- `configs/train_thk_full.yaml` — `max_steps: 246` (cumulative target).
- `slurm/train_thk_chunk.slurm` — the parameterized SLURM script.

If they were authored locally, push and pull before submitting:

```bash
# On your local box (Windows / PowerShell)
cd c:\Users\usp78\Desktop\Nemotron_Challenge\Nemotron_Challenge
git add scripts/train_lora.py configs/train_thk_full.yaml slurm/train_thk_chunk.slurm slurm/train_thk_full.slurm docs/path_beta_runbook.md
git commit -m "Add path-beta chunked training pipeline"
git push

# On the cluster (SSH session)
cd ~/Nemotron_Challenge
git pull
ls slurm/train_thk_chunk.slurm   # confirm
```

### Chunk 1

```bash
cd ~/Nemotron_Challenge
sbatch --export=ALL,THK_CHUNK_IDX=1 slurm/train_thk_chunk.slurm
# -> Submitted batch job <JOBID_1>
```

Live monitor (Ctrl+C detaches without killing the job):

```bash
tail -f logs/train_thk_chunk_<JOBID_1>.out
```

Check queue status:

```bash
squeue -u $USER -o "%.10i %.12P %.20j %.2t %.10M %.10l %.6D %R"
```

Expected wall: ~5.5h training + ~0.1h setup ≈ **~5.5 h elapsed**.

When `squeue` no longer lists the job, verify completion:

```bash
sacct -j <JOBID_1> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
# Expected: State=COMPLETED, ExitCode=0:0, Elapsed ~ 5:30:00
```

Verify chunk-1 outputs:

```bash
ls -lh outputs/lora_adapter_thk_chunk1/
# Expected files:
#   adapter_config.json         ~1.2K
#   adapter_model.safetensors   ~3.3G
#   training_state.pt           ~7-10G   <-- the resume payload
#   chat_template.jinja, tokenizer.json, tokenizer_config.json, special_tokens_map.json, README.md
```

Inspect the loss curve (all 82 steps, since `logging_steps: 1`):

```bash
grep '^\[step ' logs/train_thk_chunk_<JOBID_1>.out | head -n 5
grep '^\[step ' logs/train_thk_chunk_<JOBID_1>.out | tail -n 5
```

Sanity checks before submitting chunk 2:

- `State=COMPLETED` (NOT `TIMEOUT` or `FAILED`).
- `[step 82/82] loss=... lr=...` is the last training line (chunk hit max-steps cleanly).
- `[info] training state saved -> outputs/lora_adapter_thk_chunk1/training_state.pt (global_step=82, ...)`
- File `outputs/lora_adapter_thk_chunk1/training_state.pt` exists and is ~7-10 GB.

### Chunk 2

```bash
sbatch --export=ALL,THK_CHUNK_IDX=2 slurm/train_thk_chunk.slurm
# -> Submitted batch job <JOBID_2>
```

Critical lines to verify in the log (the path-β resume is operational):

```bash
grep -E '\[info\] (resuming|loading training|resumed|resume:|hit max_steps)' logs/train_thk_chunk_<JOBID_2>.out
```

Expected output (exact wording):

```
[info] resuming LoRA weights from: outputs/lora_adapter_thk_chunk1
[info] loading training state: outputs/lora_adapter_thk_chunk1/training_state.pt
[info] resumed: global_step=82, epoch_in_progress=0, samples_consumed_in_epoch=5248
[info] resume: fast-forwarding past 5248 consumed samples in epoch 0
[info] hit max_steps=164, stopping
```

The first `[step N/M]` line should be `[step 83/164]`, NOT `[step 1/164]`.
The LR at step 83 should be approximately `1.39e-4` (continuation of
chunk 1's linear decay from 2e-4, not a fresh peak).

### Chunk 3

```bash
sbatch --export=ALL,THK_CHUNK_IDX=3 slurm/train_thk_chunk.slurm
# -> Submitted batch job <JOBID_3>
```

Same monitoring. Chunk 3's Stage 6 promotes the live adapter to a
stable symlink:

```bash
ls -la outputs/lora_adapter
# Expected: outputs/lora_adapter -> /home/$USER/Nemotron_Challenge/outputs/lora_adapter_thk_chunk3
```

Final-step LR (step 246) is `~1.85e-5` on the linear decay over 271
steps. Adapter is now ready for evaluation.

## Expected log signals per chunk

| Stage / Signal | Where to look | Expected value |
|---|---|---|
| Stage 3 diagnostic verdict | Stage 3 console output | `PASS: tokenization fits, boundary alignment clean` |
| `trainable params:` | After Stage 4 starts | `883,873,792 / 32,461,811,136 = 2.7228%` |
| Resume detection (chunks 2-3) | `[info] resumed:` line | global_step matches end of previous chunk |
| Sampler fast-forward (chunks 2-3) | `[info] resume: fast-forwarding` line | Skip count = `82 * 64 = 5248` after chunk 1; `164 * 64 = 10496` after chunk 2 |
| Per-step loss | `[step N/M] loss=...` | Block mean decreases across the cumulative run (~0.7 early → ~0.1 late). Step-to-step bouncing is normal (category mix + wrong-answer-trace batches). |
| Final-step LR | Last `[step N/M] ... lr=...` line | Chunk 1: ~1.39e-4; Chunk 2: ~7.9e-5; Chunk 3: ~1.85e-5 |
| Adapter shape (Stage 5) | `adapter_config.json` dump | `r: 32`, `lora_alpha: 32`, 8 target modules (q/k/v/o/up/down/in/out_proj) |
| State save | End of Stage 4 console output | `[info] training state saved -> ... global_step=N` |
| Walltime | `sacct` Elapsed | ~5:30:00 per chunk |

## Troubleshooting

### Chunk N (N ≥ 2) fails with "resume dir missing adapter_config.json"

The previous chunk did not finish saving. Check the previous chunk's log
and `sacct` state. Common causes:

- Previous chunk was `TIMEOUT` (didn't reach max-steps before SLURM killed it).
- Previous chunk was OOM-killed (very unlikely given the smoke's ~58 GiB
  model footprint on the 143 GiB H200, but check `nvidia-smi` output if
  you see GPU memory pressure).

Recovery: re-submit the previous chunk. It will overwrite its own output
dir using the chunk-before's checkpoint (or, for chunk 1, fresh weights).

### Loss spikes at a chunk boundary

If the first 5-10 steps of chunk 2 or 3 show much higher loss than the
last steps of the previous chunk (say >2× higher), the optimizer state
likely did not restore correctly. Investigate:

```bash
# Confirm the resume lines fired
grep '\[info\] loading training state' logs/train_thk_chunk_<JOBID_N>.out
grep '\[info\] resumed:' logs/train_thk_chunk_<JOBID_N>.out

# Confirm the LR is NOT at peak
grep '^\[step ' logs/train_thk_chunk_<JOBID_N>.out | head -n 3
```

If the first step's LR is `2.00e-04` instead of `~1.39e-4` (for chunk 2),
the scheduler state didn't restore — investigate `training_state.pt`
contents.

### Walltime timeout

If a chunk hits the 7:00:00 SLURM cap before reaching its max-steps target,
the script is killed mid-training and `training_state.pt` is **NOT** saved.
The chunk must be re-run from scratch (from its predecessor's checkpoint,
which is intact).

Each chunk has ~1.5h of headroom at the observed 230 s/step rate, so this
should not occur. If it does, investigate whether step time has risen
(check `nvidia-smi` GPU utilization during training, look for unusual
warnings in the log).

### Disk space

Each chunk writes ~10.5 GB to `outputs/`. Three chunks plus their per-job
archive copies (`outputs/lora_adapter_thk_chunkN_<jobid>`) consume up to
~60 GB. If you re-run chunks multiple times, the archive accumulation can
grow. Periodically clean old archives:

```bash
# After all 3 chunks finished successfully, you can remove the per-job
# archives (the live dirs and the stable symlink are the durable artifacts):
rm -rf outputs/lora_adapter_thk_chunk*_<some_old_jobid>
```

## After chunk 3 completes

The trained adapter is at `outputs/lora_adapter -> outputs/lora_adapter_thk_chunk3`.

Recommended next steps (do one at a time, inspect between each):

1. **Inspect the loss curve end-to-end:**

   ```bash
   for j in <JOBID_1> <JOBID_2> <JOBID_3>; do
     grep '^\[step ' logs/train_thk_chunk_$j.out
   done | nl
   ```

   Loss block means should decrease across all 246 steps. Last few steps
   should be < 0.1 on average.

2. **Run local stratified holdout evaluation** before any Kaggle submission.
   Per `feedback_holdout_must_be_stratified.md`, the single-category holdout
   at `data/numeral_holdout.jsonl` does NOT predict the LB score; use the
   stratified eval set instead.

3. **If holdout score is in the expected range (≥ ~0.65), submit to Kaggle.**

## Backward compatibility

The path-β additions to `scripts/train_lora.py` are all optional CLI
flags. Existing invocations continue to work unchanged:

- `slurm/smoke_train.slurm` (smoke shake-down) — uses no flags; runs as before.
- `slurm/lora_verification.slurm` (verification pipeline) — uses no flags; runs as before.
- `slurm/train_thk_full.slurm` (deprecated single-shot run) — would still
  execute, but is documented as deprecated since it cannot complete inside
  the 8h partition cap.
