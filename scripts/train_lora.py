"""LoRA SFT training driver for Nemotron-3 Nano.

Stack: HF transformers + peft + accelerate. This is the same machinery
NeMo Automodel uses underneath; calling it directly is more transparent
than the Automodel YAML wrapper and avoids schema drift across NeMo
releases.

This script is intentionally small. It is a *verification* trainer: it
proves the pipeline produces a valid HF/PEFT adapter directory
(``adapter_config.json`` + ``adapter_model.safetensors``) that the Kaggle
eval harness will accept. It is not tuned for accuracy.

Why these defaults:
  - ``target_modules=[q_proj, k_proj, v_proj, o_proj]``: attention-only.
    "all-linear" was the original choice but produced an adapter that
    vLLM 0.12.0 refuses to load — its NemotronHForCausalLM lacks the
    ``get_expert_mapping`` method needed to apply LoRA to MoE experts.
    Since Kaggle scores via vLLM, the adapter must avoid expert layers
    entirely. See docs/lora_strategy.md §3 "vLLM MoE LoRA constraint".
  - 2026-05-03 update: the verification YAML now restores
    ``target_modules="all-linear"`` and evaluates with a newer vLLM image.
    The older attention-only note above describes the v0.12.0 workaround
    attempt that job 6518135 proved insufficient.
  - ``gradient_checkpointing=True``: 30B BF16 weights = ~60 GB; on a
    single H200 (143 GB), checkpointing keeps activation memory in
    range at ``max_seq_len=4096, batch=1``.
  - Loss masked on the user turn: matches the inference-time
    distribution (the model is only ever asked to generate the
    assistant turn). Toggle via ``mask_user_loss`` in the YAML.

Run from inside the NeMo (or any HF-training-capable) container.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_dataset(records: list[dict], tokenizer, max_seq_len: int, mask_user_loss: bool):
    """Render each record's chat messages and tokenize.

    For each sample we render twice:
      1. The full conversation (user + assistant) with the chat template.
      2. The same conversation truncated to the user turn + generation
         prompt (the "prompt prefix" the model would see at inference).

    The token count of (2) is the boundary at which assistant tokens
    begin in (1). Tokens before the boundary get label=-100 when
    mask_user_loss is true. This is the standard way to do SFT-on-
    assistant-only with HF tokenizers and avoids the chat-template-
    parsing fragility of the alternatives.

    Returns a list of dicts (unpadded). Padding happens per-batch in the
    collate_fn so step compute scales with the actual batch length, not
    the global max — important when raising max_seq_len from 4096 to
    8192 would otherwise pin every step at the worst-case length.
    """
    samples: list[dict] = []
    skipped = 0
    for rec in records:
        messages = rec["messages"]
        # enable_thinking=True must match scripts/baseline_generate.py and
        # the Kaggle submission notebook -- both render the prompt with the
        # `<think>\n` prefix appended to the assistant turn. Training without
        # this flag placed our trace at a position the inference distribution
        # never sees (immediately after `<|assistant|>`), so the LoRA learned
        # nothing useful for the post-`</think>` answer slot. See iteration-3
        # post-mortem: 6/200 LoRA accuracy because of this single mismatch.
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=True,
        )
        prompt_only_text = tokenizer.apply_chat_template(
            [m for m in messages if m["role"] != "assistant"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_seq_len)
        prompt = tokenizer(prompt_only_text, add_special_tokens=False, truncation=True, max_length=max_seq_len)

        full_ids = full["input_ids"]
        prompt_len = min(len(prompt["input_ids"]), len(full_ids))

        if prompt_len >= len(full_ids):
            skipped += 1
            continue

        lbl = list(full_ids)
        if mask_user_loss:
            for i in range(prompt_len):
                lbl[i] = -100

        samples.append({
            "input_ids": full_ids,
            "labels": lbl,
            "attention_mask": full["attention_mask"],
            "length": len(full_ids),
        })

    if skipped:
        print(f"[warn] skipped {skipped} samples whose prompt filled the full context")

    return samples


class ListDataset:
    """Tiny dataset wrapper so DataLoader can index into the sample list."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def make_collate_fn(pad_id: int):
    """Pad to the longest sample *in this batch*, not the global max.

    With per_device_batch_size=1 this means each step processes exactly the
    sample's own length — at max_seq_len=8192, short traces no longer pay
    the worst-case forward/backward cost. Padding tokens still incur
    compute on Mamba/MoE paths, so reducing them is the cheapest win.
    """
    import torch

    def collate(batch: list[dict]):
        max_len = max(s["length"] for s in batch)

        def pad(seq: list[int], fill: int) -> list[int]:
            return seq + [fill] * (max_len - len(seq))

        input_ids = torch.tensor([pad(s["input_ids"], pad_id) for s in batch], dtype=torch.long)
        labels = torch.tensor([pad(s["labels"], -100) for s in batch], dtype=torch.long)
        attn = torch.tensor([pad(s["attention_mask"], 0) for s in batch], dtype=torch.long)
        return input_ids, attn, labels

    return collate


def _make_length_bucketed_sampler(lengths: list[int], batch_size: int, seed: int):
    """Group similar-length samples into the same batch.

    Sort the dataset by length, chunk into mega-batches of size
    bucket_size*batch_size, shuffle inside each chunk, then yield
    individual indices. Output order is approximately sorted but with
    enough randomness that consecutive epochs don't see identical
    micro-batches. No-op at batch_size=1 in terms of padding waste, but
    keeps gradient noise comparable to plain shuffling at larger bs.

    Built lazily so we can subclass torch.utils.data.Sampler at call time
    (avoiding a top-level torch import for the small helpers above).
    """
    from torch.utils.data import Sampler

    class LengthBucketedSampler(Sampler):
        def __init__(self, lengths, batch_size, bucket_size=32, seed=0):
            super().__init__(data_source=None)
            self.lengths = lengths
            self.batch_size = batch_size
            self.bucket_size = bucket_size
            self.seed = seed
            self.epoch = 0

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __len__(self):
            return len(self.lengths)

        def __iter__(self):
            import random

            rng = random.Random(self.seed + self.epoch)
            order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
            chunk = self.batch_size * self.bucket_size
            chunks = [order[i : i + chunk] for i in range(0, len(order), chunk)]
            for c in chunks:
                rng.shuffle(c)
            rng.shuffle(chunks)
            for c in chunks:
                for i in c:
                    yield i

    return LengthBucketedSampler(lengths, batch_size, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    # Path-beta chunking flags. All optional; backward-compatible with the
    # single-shot invocation used by smoke_train.slurm / lora_verification.slurm.
    ap.add_argument(
        "--resume-from-checkpoint",
        dest="resume_from_checkpoint",
        default=None,
        help="Path to a previous chunk's output dir containing adapter_config.json + "
             "adapter_model.safetensors + training_state.pt. When set, the run "
             "continues from that state (optimizer momentum, LR schedule position, "
             "and sampler position are all restored).",
    )
    ap.add_argument(
        "--max-steps",
        dest="max_steps_cli",
        type=int,
        default=None,
        help="Override train.max_steps in the YAML for THIS process invocation. "
             "Path-beta uses this to cap each chunk's cumulative global_step "
             "(chunk 1=82, chunk 2=164, chunk 3=246).",
    )
    ap.add_argument(
        "--output-dir",
        dest="output_dir_cli",
        default=None,
        help="Override output.adapter_dir in the YAML so each chunk writes its "
             "adapter + training_state to a chunk-specific directory.",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    out_cfg = cfg["output"]

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader

    print(f"[info] loading tokenizer: {model_cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], trust_remote_code=model_cfg.get("trust_remote_code", True)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[info] loading data: {data_cfg['input_path']}")
    records = read_jsonl(data_cfg["input_path"])
    print(f"[info] {len(records)} training samples")

    samples = build_dataset(
        records,
        tokenizer,
        max_seq_len=data_cfg.get("max_seq_len", 4096),
        mask_user_loss=data_cfg.get("mask_user_loss", True),
    )
    dataset = ListDataset(samples)
    print(f"[info] tokenized dataset size: {len(dataset)}")
    lengths = [s["length"] for s in samples]
    if lengths:
        lengths_sorted = sorted(lengths)
        p50 = lengths_sorted[len(lengths_sorted) // 2]
        p90 = lengths_sorted[int(0.9 * (len(lengths_sorted) - 1))]
        print(f"[info] sample length tokens: p50={p50} p90={p90} max={lengths_sorted[-1]}")

    print(f"[info] loading base model in bf16 (this can take a few minutes)")
    dtype = torch.bfloat16 if model_cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        device_map="auto",
    )
    base.config.use_cache = False
    if train_cfg.get("gradient_checkpointing", True):
        base.gradient_checkpointing_enable()

    print(f"[info] applying LoRA: r={lora_cfg['r']} alpha={lora_cfg['alpha']} target={lora_cfg['target_modules']}")
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=lora_cfg["target_modules"],
    )
    if args.resume_from_checkpoint:
        from peft import PeftModel
        resume_dir_path = Path(args.resume_from_checkpoint)
        if not (resume_dir_path / "adapter_config.json").is_file():
            raise SystemExit(f"resume dir missing adapter_config.json: {resume_dir_path}")
        if not (resume_dir_path / "training_state.pt").is_file():
            raise SystemExit(f"resume dir missing training_state.pt: {resume_dir_path}")
        # is_trainable=True is critical: the default is inference mode, which
        # freezes the adapter weights and would silently produce a non-training
        # run that overwrites the checkpoint with unchanged weights.
        print(f"[info] resuming LoRA weights from: {resume_dir_path}")
        model = PeftModel.from_pretrained(base, str(resume_dir_path), is_trainable=True)
    else:
        model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()

    bs = train_cfg.get("per_device_batch_size", 1)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 16)
    n_epochs = train_cfg.get("num_epochs", 1)
    lr = float(train_cfg.get("learning_rate", 2e-4))
    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    sampler = _make_length_bucketed_sampler(lengths, batch_size=bs, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=bs,
        sampler=sampler,
        drop_last=False,
        collate_fn=make_collate_fn(pad_id),
    )
    steps_per_epoch = max(1, len(loader) // grad_accum)
    total_steps_full = steps_per_epoch * n_epochs
    # max_steps caps the optimizer-step count. Iteration N hit loss<0.01
    # by step 285 of 421, so the tail is overfit churn that also blew the
    # 5h SLURM walltime. The cosine schedule still spans total_steps_full
    # so the LR shape matches what we'd get without the cap.
    config_max_steps = int(train_cfg.get("max_steps", 0) or 0)
    if args.max_steps_cli is not None:
        max_steps_cfg = int(args.max_steps_cli)
        print(f"[info] CLI --max-steps={max_steps_cfg} (overrides config max_steps={config_max_steps})")
    else:
        max_steps_cfg = config_max_steps
    total_steps = min(total_steps_full, max_steps_cfg) if max_steps_cfg > 0 else total_steps_full
    warmup_steps = max(1, int(total_steps_full * float(train_cfg.get("warmup_ratio", 0.03))))

    adam_beta1 = float(train_cfg.get("adam_beta1", 0.9))
    adam_beta2 = float(train_cfg.get("adam_beta2", 0.999))
    adam_eps = float(train_cfg.get("adam_eps", 1e-8))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0) or 0.0)
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    # THK's 04-10-04-33 used StepLinearDecayLRSchedule (linear decay from
    # peak LR to 0 over total_steps, no warmup). Our default remains cosine
    # for backward compat with the iteration-2/3 verification baseline.
    scheduler_name = str(train_cfg.get("lr_scheduler", "cosine"))
    if scheduler_name == "linear":
        from transformers import get_linear_schedule_with_warmup
        sched = get_linear_schedule_with_warmup(
            optim, warmup_steps, total_steps_full
        )
    elif scheduler_name == "cosine":
        sched = get_cosine_schedule_with_warmup(
            optim, warmup_steps, total_steps_full
        )
    else:
        raise ValueError(f"unknown lr_scheduler: {scheduler_name!r}")

    # Path-beta resume: restore optimizer + scheduler state from a previous
    # chunk's training_state.pt. Optimizer state holds AdamW momentum (m) and
    # variance (v) per trainable param; without it, each chunk would start
    # with zero-momentum gradients and corrupt the recipe. Scheduler state
    # holds the LR schedule's step counter so decay continues smoothly.
    resume_global_step = 0
    resume_epoch = 0
    resume_samples_in_epoch = 0
    if args.resume_from_checkpoint:
        state_path = Path(args.resume_from_checkpoint) / "training_state.pt"
        print(f"[info] loading training state: {state_path}")
        # weights_only=False because the state dict contains optimizer
        # internals (tensors + python ints) that don't satisfy the
        # weights-only safelist. The file comes from our own previous run,
        # so trusting it is fine.
        saved_state = torch.load(state_path, map_location="cpu", weights_only=False)
        optim.load_state_dict(saved_state["optimizer"])
        sched.load_state_dict(saved_state["scheduler"])
        resume_global_step = int(saved_state["global_step"])
        resume_epoch = int(saved_state["epoch_in_progress"])
        resume_samples_in_epoch = int(saved_state["samples_consumed_in_epoch"])
        print(
            f"[info] resumed: global_step={resume_global_step}, "
            f"epoch_in_progress={resume_epoch}, "
            f"samples_consumed_in_epoch={resume_samples_in_epoch}"
        )

    log_every = int(train_cfg.get("logging_steps", 5))
    model.train()
    global_step = resume_global_step
    optim.zero_grad()
    print(
        f"[info] training: total_optim_steps={total_steps} "
        f"(uncapped={total_steps_full}, max_steps={max_steps_cfg or 'none'}, "
        f"starting_global_step={global_step}) warmup={warmup_steps}"
    )

    if resume_epoch >= n_epochs:
        raise SystemExit(
            f"[error] resume_epoch={resume_epoch} >= num_epochs={n_epochs}. "
            f"Either raise num_epochs in the config or stop training (target reached)."
        )

    # Initialize before the loop so values are defined even if the range is
    # empty (e.g., resume_epoch == n_epochs). Loop will rebind both.
    epoch = resume_epoch
    samples_consumed_in_epoch = resume_samples_in_epoch
    stop = False

    for epoch in range(resume_epoch, n_epochs):
        if stop:
            break
        sampler.set_epoch(epoch)
        loader_iter = iter(loader)

        # Path-beta fast-forward: only on the first epoch we enter after a
        # resume, skip past micro-batches the previous chunk already consumed
        # from this epoch's deterministic shuffle. This preserves the
        # "one continuous shuffle" property that THK's single-run had — each
        # sample is seen at most once across all 3 chunks, not re-shuffled.
        if epoch == resume_epoch and resume_samples_in_epoch > 0:
            samples_consumed_in_epoch = resume_samples_in_epoch
            print(
                f"[info] resume: fast-forwarding past {samples_consumed_in_epoch} "
                f"consumed samples in epoch {epoch}"
            )
            for _ in range(samples_consumed_in_epoch):
                try:
                    next(loader_iter)
                except StopIteration:
                    print(
                        f"[warn] consumed-sample count ({samples_consumed_in_epoch}) "
                        f">= epoch length; resume state inconsistent"
                    )
                    break
        else:
            samples_consumed_in_epoch = 0

        for batch in loader_iter:
            input_ids, attn_mask, labels = batch
            input_ids = input_ids.to(model.device)
            attn_mask = attn_mask.to(model.device)
            labels = labels.to(model.device)

            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss / grad_accum
            loss.backward()

            samples_consumed_in_epoch += 1

            if samples_consumed_in_epoch % grad_accum == 0:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad),
                        max_grad_norm,
                    )
                optim.step()
                sched.step()
                optim.zero_grad()
                global_step += 1
                if global_step % log_every == 0:
                    print(
                        f"[step {global_step}/{total_steps}] "
                        f"loss={(loss.item() * grad_accum):.4f} lr={sched.get_last_lr()[0]:.2e}"
                    )
                if max_steps_cfg > 0 and global_step >= max_steps_cfg:
                    print(f"[info] hit max_steps={max_steps_cfg}, stopping")
                    stop = True
                    break

        # tail: flush any remaining grad. Skip when stopping mid-epoch — the
        # next chunk's resume should pick up at the same micro-batch boundary
        # without an extra optimizer step in between.
        if not stop and samples_consumed_in_epoch % grad_accum != 0:
            optim.step()
            sched.step()
            optim.zero_grad()

    adapter_dir = Path(args.output_dir_cli or out_cfg["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] saving adapter -> {adapter_dir}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Save path-beta training state alongside the adapter. The next chunk
    # restores all three of (optimizer state, scheduler state, sampler
    # position) from this file. Even on the final chunk this is written —
    # cheap (~7 GB) and useful for post-hoc debugging.
    training_state = {
        "optimizer": optim.state_dict(),
        "scheduler": sched.state_dict(),
        "global_step": global_step,
        # If we stopped mid-epoch (max_steps hit), the next chunk continues
        # in the same epoch. If we ran to the end of the inner loop without
        # stopping (epoch fully consumed), the next chunk starts a fresh
        # epoch and the sampler skip count resets to 0.
        "epoch_in_progress": epoch if stop else (epoch + 1),
        "samples_consumed_in_epoch": samples_consumed_in_epoch if stop else 0,
        "seed": seed,
    }
    state_path = adapter_dir / "training_state.pt"
    torch.save(training_state, state_path)
    print(
        f"[info] training state saved -> {state_path} "
        f"(global_step={global_step}, "
        f"epoch_in_progress={training_state['epoch_in_progress']}, "
        f"samples_consumed_in_epoch={training_state['samples_consumed_in_epoch']})"
    )

    cfg_files = sorted(p.name for p in adapter_dir.iterdir())
    print(f"[info] adapter dir contents: {cfg_files}")
    if "adapter_config.json" not in cfg_files:
        raise SystemExit("adapter_config.json missing — Kaggle submission would be rejected.")

    print("[info] training complete.")


if __name__ == "__main__":
    main()
