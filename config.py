"""
config.py — single source of truth for the whole benchmark.

Everything that BOTH training and evaluation must agree on lives here:
the model id, the dataset, the sizes, the LoRA/training hyperparameters, and
(critically) the prompt format. If you change the template, change it ONCE here
and both train.py and evaluate.py stay in sync automatically.
"""

import os

# ─── What we're training ─────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "b-mc2/sql-create-context"   # columns: question, context (DDL), answer (SQL)

# ─── Toy-dataset sizes (keep small so each run finishes in ~10 min on a T4) ───
N_TRAIN = 2000
N_TEST = 100
SEED = 42   # fixed seed => both runs get the *same* train/test split

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")
OUTPUT_DIR = os.path.join(ROOT, "outputs")     # adapters + telemetry land here

# Per-mode adapter output dirs. `mode` is "lora" or "qlora".
def adapter_dir(mode: str) -> str:
    return os.path.join(OUTPUT_DIR, f"adapter_{mode}")

# ─── LoRA hyperparameters (IDENTICAL for both runs — that's the whole point) ──
# Only the base-model *loading* differs between LoRA and QLoRA; the adapter is
# the same shape in both, so any quality gap is attributable to quantization.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ─── Training hyperparameters ─────────────────────────────────────────────────
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2          # effective batch = 4 * 2 = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512          # schema + question + SQL comfortably fits
LOGGING_STEPS = 10
WARMUP_RATIO = 0.03

# Gradient checkpointing trades ~20-30% training speed for large VRAM savings.
# A 1.5B model fits on 16GB WITHOUT it, so we default OFF to keep the LoRA-vs-QLoRA
# SPEED comparison clean (no checkpointing overhead muddying the steps/sec numbers).
# Whatever you set here is applied to BOTH runs identically — never to just one.
GRADIENT_CHECKPOINTING = False


# ─── Generation settings used at eval time ───────────────────────────────────
MAX_NEW_TOKENS = 256

# ─── THE PROMPT TEMPLATE (single source of truth) ────────────────────────────
# Plaintext template exactly as specified in the project brief. We deliberately
# do NOT use Qwen's chat template here so the A/B test isolates quantization.
RESPONSE_TEMPLATE = "[SQL]\n"   # everything before this is "prompt", after is "completion"

def build_prompt(context: str, question: str) -> str:
    """The template up to (and including) [SQL]\\n. Fed to the model at eval time."""
    return (
        f"[SCHEMA]\n{context.strip()}\n"
        f"[QUESTION]\n{question.strip()}\n"
        f"{RESPONSE_TEMPLATE}"
    )

def build_full_text(context: str, question: str, answer: str) -> str:
    """Prompt + gold SQL. The model is trained on this full string.

    Note: the tokenizer's EOS token is appended in train.py (where the tokenizer
    is loaded) so the model learns where to STOP — without it the model never
    terminates generation cleanly.
    """
    return build_prompt(context, question) + answer.strip()