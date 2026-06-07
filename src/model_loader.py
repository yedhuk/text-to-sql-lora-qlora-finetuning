"""
model_loader.py — STEP 2: the A/B toggle.

load_model("lora")  -> base weights in bfloat16 (16-bit)
load_model("qlora") -> base weights in 4-bit NF4, dequantized to bf16 on the fly

The LoRA adapter applied afterward is IDENTICAL in both cases (same config), so any
downstream quality difference is attributable to base-weight precision alone.
"""

import sys
import os
from importlib.metadata import version

import torch
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_tokenizer():
    """Tokenizer set up for training (right-padding, pad token defined)."""
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tok.pad_token is None:
        # NEVER reuse eos as pad: the LM collator masks pad positions to -100, so
        # if pad==eos the model never learns to emit eos and won't stop generating.
        # Qwen has a distinct <|endoftext|> we can borrow as pad.
        tok.pad_token = "<|endoftext|>"
    tok.padding_side = "right"
    return tok


def _build_lora_config() -> LoraConfig:
    """The SAME adapter for both runs, so any quality difference is due to base precision."""
    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        # CAUSAL_LM adapter: only the tokens after the response template are supervised, so the model learns to attend to the prompt and generate the SQL answer, but is NOT trained to reproduce the prompt itself. This is a more realistic setup for how the model will be used at eval time, and it also makes the training more efficient by focusing the loss on the SQL generation part of the sequence. The alternative would be a seq2seq-style setup where the model is trained to reproduce the entire prompt + response, but that would be less efficient and the model would spend capacity on learning to copy the prompt rather than just learning to generate the SQL. See the `DataCollatorForCompletionOnlyLM` in `train.py` for how the labels are masked to achieve this.
        task_type="CAUSAL_LM",
    )


def _assert_qlora_supported():
    """Turn a cryptic CUDA kernel crash into a clear, actionable message."""
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA GPU, but none was detected.")
    cc = torch.cuda.get_device_capability(0)  # e.g. (12, 0) on an RTX 5080
    if cc >= (12, 0):  # Blackwell / sm_120 or newer
        bnb_ver = Version(version("bitsandbytes"))
        if bnb_ver < Version("0.46.1"):
            raise RuntimeError(
                f"Detected a Blackwell GPU (sm_{cc[0]}{cc[1]}) but bitsandbytes "
                f"{bnb_ver} predates its 4-bit kernels. Fix:\n"
                f"    pip install -U 'bitsandbytes>=0.46.1'"
            )


def load_model(mode: str):
    """Load the base model for the given mode and attach the LoRA adapter.

    Args:
        mode: "lora" (16-bit base) or "qlora" (4-bit base).
    Returns:
        A PEFT model ready to hand to SFTTrainer.
    """
    mode = mode.lower()
    if mode not in ("lora", "qlora"):
        raise ValueError(f"mode must be 'lora' or 'qlora', got {mode!r}")

    if mode == "lora":
        # ── Run A: 16-bit base weights ──────────────────────────────────────
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},   # pin everything to GPU 0 (single-GPU training)
            # NOTE: not using flash_attention_2 — its JIT path is broken on
            # Blackwell and unnecessary at this scale. Default SDPA is used.
        )
        if config.GRADIENT_CHECKPOINTING:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()  # needed for ckpt + PEFT

    else:
        # ── Run B: 4-bit base weights (QLoRA) ───────────────────────────────
        _assert_qlora_supported()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,        # quantize the quant constants too
            bnb_4bit_quant_type="nf4",             # NormalFloat4
            bnb_4bit_compute_dtype=torch.bfloat16,  # dequantize to bf16 for each matmul
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map={"": 0},
        )
        # Casts layer norms to fp32, enables input grads, prepares the frozen
        # 4-bit base to backprop into the 16-bit adapter.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        )

    # ── Shared tail: the IDENTICAL adapter for both runs ────────────────────
    #get_peft_model does NOT modify the base model's weights or precision; it just adds the adapter layers on top. So the LoRA adapter is exactly the same in both runs, and any quality difference is attributable to the base model's precision alone.
    model = get_peft_model(model, _build_lora_config())
    model.config.use_cache = False  # incompatible with training; required off
    print(f"\n[model_loader] mode={mode}")
    model.print_trainable_parameters()
    return model


if __name__ == "__main__":
    # Quick smoke test: python src/model_loader.py lora   (or qlora)
    m = load_model(sys.argv[1] if len(sys.argv) > 1 else "lora")
    print("loaded OK:", type(m).__name__)