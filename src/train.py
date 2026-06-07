"""
train.py — STEP 3a: train ONE configuration and log telemetry.

Usage:
    python src/train.py lora
    python src/train.py qlora

Writes:
    outputs/adapter_{mode}/        the trained LoRA adapter + tokenizer
    outputs/telemetry_{mode}.json  peak VRAM, steps/sec, runtime, final loss

COMPLETION-ONLY MASKING (robust version):
We tokenize the prompt and the answer SEPARATELY, concatenate the token ids, and
set the prompt's labels to -100. This supervises only the SQL (+EOS) without any
fragile substring search for a response template — so it can't silently mask the
whole sequence the way DataCollatorForCompletionOnlyLM can with BPE tokenizers.
"""

import sys
import os
import json
import time
import argparse

import torch
from datasets import load_dataset
from transformers import (
    set_seed,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model_loader import load_model, load_tokenizer


def build_tokenized_dataset(tokenizer):
    """Return a dataset of {input_ids, attention_mask, labels} with the prompt
    masked (-100) so loss is computed only on the SQL answer + EOS."""
    ds = load_dataset("json", data_files=config.TRAIN_FILE, split="train")
    eos = tokenizer.eos_token
    max_len = config.MAX_SEQ_LENGTH

    def tokenize(ex):
        prompt = config.build_prompt(ex["context"], ex["question"])
        answer = ex["answer"].strip() + eos
        # add_special_tokens=False: Qwen uses no BOS; we control the format fully.
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        input_ids = (p_ids + a_ids)[:max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:max_len]   # mask the prompt exactly
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    return ds.map(tokenize, remove_columns=ds.column_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["lora", "qlora"])
    args = parser.parse_args()
    mode = args.mode

    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ── Data + tokenizer ────────────────────────────────────────────────────
    tokenizer = load_tokenizer()
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise RuntimeError(
            "pad_token == eos_token: padding would mask every EOS out of the loss "
            "and the model would never learn to stop. Use a distinct pad token."
        )
    train_ds = build_tokenized_dataset(tokenizer)
    print(f"[data] {len(train_ds)} training examples")

    # Preflight: confirm the prompt mask leaves the answer tokens supervised.
    n_supervised = sum(1 for x in train_ds[0]["labels"] if x != -100)
    if n_supervised == 0:
        raise RuntimeError("Sample 0 has no supervised tokens — masking is wrong.")
    print(f"[preflight] {n_supervised} supervised tokens in sample 0 (the SQL answer). OK.")

    # Pads input_ids with pad_token_id and labels with -100; builds attention_mask.
    collator = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=-100, padding=True
    )

    # ── Model (the A/B toggle) ──────────────────────────────────────────────
    model = load_model(mode)
    torch.cuda.synchronize()
    model_footprint_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[vram] model footprint after load: {model_footprint_gb:.2f} GB")

    # ── Training config (identical for both modes) ──────────────────────────
    train_args = TrainingArguments(
        output_dir=config.adapter_dir(mode),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=config.LOGGING_STEPS,
        bf16=True,
        optim="adamw_torch",        # SAME optimizer for both modes (clean A/B)
        save_strategy="no",
        report_to="none",
        seed=config.SEED,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    # ── Train + capture telemetry ───────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    wall_clock = time.time() - t0

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    m = result.metrics
    telemetry = {
        "mode": mode,
        "model_footprint_gb": round(model_footprint_gb, 3),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "wall_clock_sec": round(wall_clock, 2),
        "train_runtime_sec": round(m.get("train_runtime", wall_clock), 2),
        "steps_per_second": round(m.get("train_steps_per_second", 0.0), 4),
        "samples_per_second": round(m.get("train_samples_per_second", 0.0), 3),
        "final_train_loss": round(m.get("train_loss", float("nan")), 4),
        "effective_batch_size": config.PER_DEVICE_BATCH_SIZE * config.GRAD_ACCUM_STEPS,
        "num_train_epochs": config.NUM_EPOCHS,
        "gradient_checkpointing": config.GRADIENT_CHECKPOINTING,
    }

    # ── Save adapter, tokenizer, telemetry ──────────────────────────────────
    trainer.save_model(config.adapter_dir(mode))
    tokenizer.save_pretrained(config.adapter_dir(mode))
    tel_path = os.path.join(config.OUTPUT_DIR, f"telemetry_{mode}.json")
    with open(tel_path, "w") as f:
        json.dump(telemetry, f, indent=2)

    print(f"\n[done] {mode}")
    print(json.dumps(telemetry, indent=2))
    print(f"[saved] adapter -> {config.adapter_dir(mode)}")
    print(f"[saved] telemetry -> {tel_path}")


if __name__ == "__main__":
    main()