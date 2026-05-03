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
    """
    import torch

    inputs_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attn: list[list[int]] = []

    skipped = 0
    for rec in records:
        messages = rec["messages"]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_only_text = tokenizer.apply_chat_template(
            [m for m in messages if m["role"] != "assistant"],
            tokenize=False,
            add_generation_prompt=True,
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

        inputs_ids.append(full_ids)
        labels.append(lbl)
        attn.append(full["attention_mask"])

    if skipped:
        print(f"[warn] skipped {skipped} samples whose prompt filled the full context")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    max_len = max(len(x) for x in inputs_ids)

    def pad(seq: list[int], fill: int) -> list[int]:
        return seq + [fill] * (max_len - len(seq))

    input_tensor = torch.tensor([pad(x, pad_id) for x in inputs_ids], dtype=torch.long)
    label_tensor = torch.tensor([pad(x, -100) for x in labels], dtype=torch.long)
    attn_tensor = torch.tensor([pad(x, 0) for x in attn], dtype=torch.long)

    from torch.utils.data import TensorDataset
    return TensorDataset(input_tensor, attn_tensor, label_tensor)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
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

    dataset = build_dataset(
        records,
        tokenizer,
        max_seq_len=data_cfg.get("max_seq_len", 4096),
        mask_user_loss=data_cfg.get("mask_user_loss", True),
    )
    print(f"[info] tokenized dataset size: {len(dataset)}")

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
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()

    bs = train_cfg.get("per_device_batch_size", 1)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 16)
    n_epochs = train_cfg.get("num_epochs", 1)
    lr = float(train_cfg.get("learning_rate", 2e-4))
    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)

    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)
    steps_per_epoch = max(1, len(loader) // grad_accum)
    total_steps = steps_per_epoch * n_epochs
    warmup_steps = max(1, int(total_steps * float(train_cfg.get("warmup_ratio", 0.03))))

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    log_every = int(train_cfg.get("logging_steps", 5))
    model.train()
    global_step = 0
    optim.zero_grad()
    print(f"[info] training: total_optim_steps={total_steps} warmup={warmup_steps}")

    for epoch in range(n_epochs):
        for micro_step, (input_ids, attn_mask, labels) in enumerate(loader):
            input_ids = input_ids.to(model.device)
            attn_mask = attn_mask.to(model.device)
            labels = labels.to(model.device)

            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss / grad_accum
            loss.backward()

            if (micro_step + 1) % grad_accum == 0:
                optim.step()
                sched.step()
                optim.zero_grad()
                global_step += 1
                if global_step % log_every == 0:
                    print(
                        f"[step {global_step}/{total_steps}] "
                        f"loss={(loss.item() * grad_accum):.4f} lr={sched.get_last_lr()[0]:.2e}"
                    )

        # tail: flush any remaining grad
        if (micro_step + 1) % grad_accum != 0:
            optim.step()
            sched.step()
            optim.zero_grad()

    adapter_dir = Path(out_cfg["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] saving adapter -> {adapter_dir}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    cfg_files = sorted(p.name for p in adapter_dir.iterdir())
    print(f"[info] adapter dir contents: {cfg_files}")
    if "adapter_config.json" not in cfg_files:
        raise SystemExit("adapter_config.json missing — Kaggle submission would be rejected.")

    print("[info] training complete.")


if __name__ == "__main__":
    main()
