# Text-to-SQL: LoRA vs QLoRA A/B Benchmark

Fine-tune the same model **twice** — once with LoRA (16-bit base), once with QLoRA
(4-bit base) — on a Text-to-SQL task, then measure what quantization actually costs
and saves: peak VRAM, training speed, and SQL accuracy.

The point of the project isn't to "win" — it's to produce an honest, reproducible
side-by-side and understand *where* QLoRA's famous memory savings come from (and where
they don't).

## The task

Given a database schema (DDL) and a natural-language question, output a correct SQL
query. Every example is formatted into one strict template:

```
[SCHEMA]
CREATE TABLE Users (id INT, name VARCHAR, age INT)
[QUESTION]
How many users are over 30?
[SQL]
SELECT COUNT(*) FROM Users WHERE age > 30;
```

The model is trained to produce only the part after `[SQL]`.

## Stack

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Dataset:** `b-mc2/sql-create-context` (schema + question + gold SQL; ships no rows)
- **Libraries:** `transformers`, `peft`, `bitsandbytes`, `datasets` (pinned — see below)

## Project layout

```
text-to-sql-benchmark/
├── requirements.txt        Pinned deps (HF stack); torch/bnb installed separately
├── config.py               Single source of truth: prompt template + all hyperparams
├── data/
│   └── prepare_data.py      Step 1 — 2000 train / 100 holdout, formatted to JSONL
├── src/
│   ├── model_loader.py      Step 2 — the A/B toggle: load_model("lora"|"qlora")
│   ├── train.py             Step 3a — train + capture VRAM/speed telemetry
│   └── evaluate.py          Step 3b — execute generated SQL on SQLite, score accuracy
└── run_benchmark.py        Orchestrator → comparison table + benchmark_summary.{md,json}
```

Two design choices keep the comparison honest: the prompt format lives in exactly one
place (`config.py`), so training and eval can never drift; and the LoRA adapter is
*identical* in both runs (same rank, alpha, target modules) — only the base-weight
precision differs, so any gap is attributable to quantization alone.

## Setup

This was developed on an **RTX 5080 (Blackwell, `sm_120`)**, which requires recent CUDA
kernels. Install `torch` from the CUDA 12.8 index **first**, then the rest:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verify the GPU is driven natively (`capability` must print `(12, 0)` on a 5080):

```python
import torch
print(torch.cuda.get_device_capability(0))   # (12, 0)
print(torch.cuda.is_available())              # True
```

**Notes on versions.** The Hugging Face libraries are pinned because their APIs change
between minor releases (`trl` especially). `bitsandbytes` is a floor (`>=0.46.1`) rather
than an exact pin because 4-bit Blackwell kernels are hardware-dependent — anything
before that lacks `sm_120` support. This run used `bitsandbytes 0.49.2` and
`torch 2.11.0+cu128`; record `pip freeze | grep bitsandbytes` for exact reproducibility.

## Running

Full pipeline (both modes, end to end):

```bash
python run_benchmark.py
```

Each stage runs as an **isolated subprocess** so one run's VRAM never leaks into the
next run's measurement. Useful flags: `--skip-data` (reuse existing JSONL),
`--report-only` (rebuild the table from existing result files without retraining).

Or run stages by hand:

```bash
python data/prepare_data.py
python src/train.py lora    && python src/evaluate.py lora
python src/train.py qlora   && python src/evaluate.py qlora
```

## Outputs (in `outputs/`)

- `telemetry_{mode}.json` — peak VRAM, model footprint, steps/sec, runtime, final loss
- `eval_{mode}.json` — accuracy summary + every one of the 100 examples (gold vs generated)
- `eval_{mode}.md` — the same dump, readable, for eyeballing failures side by side
- `benchmark_summary.{md,json}` — the LoRA-vs-QLoRA comparison table (the deliverable)

### How accuracy is measured

The dataset ships **no table rows**, so each schema is built as an in-memory SQLite DB
and populated with a few deterministic synthetic rows; both the gold and generated
queries run against that identical DB. Two metrics:

- **valid-SQL rate** — query executes without error. This is the *hallucination detector*
  and is robust to the no-data problem (a nonexistent column throws an error regardless).
- **exec-match rate** — generated result equals gold result (rows compared
  order-insensitively).

Because many queries filter on specific values the synthetic rows won't contain, both
queries often return empty and "match" trivially — so the **absolute** exec-match number
is a soft proxy. Trust the **LoRA-vs-QLoRA delta** and the hallucination counts more than
the headline percentage. Generation is **greedy** (`do_sample=False`) so the eval is
deterministic and re-runnable.

## Results (RTX 5080, 1.5B model, 2000 train / 100 test, 3 epochs)

```
Metric                   LoRA   QLoRA  Δ (QLoRA vs LoRA)
--------------------------------------------------------
Peak VRAM (GB)           6.71    6.75              +0.7%
Model footprint (GB)     3.11    1.64             -47.3%
Train steps/sec         7.469   5.465             -26.8%
Train runtime (s)      100.42  137.23             +36.7%
Final train loss       0.0685  0.0676             -0.001
Valid-SQL rate          0.990   0.990                 +0
Exec-match rate         0.929   0.939             +0.010
Hallucinations              0       1                 +1
Syntax errors               1       0                 -1
```

### What it means

**Quality is a tie.** The loss curves are nearly identical and both models produce valid,
executable SQL ~99% of the time. The exec-match gap (1 example out of 98) and the
hallucination/syntax differences (single examples) are noise, not real effects. A 4-bit
base + 16-bit adapter learned this task just as well as a 16-bit base — the central QLoRA
claim, reproduced.

**Speed cost is real.** QLoRA trained ~27% slower. Its 4-bit weights are dequantized to
bf16 on the fly for every matmul, and that overhead is exactly the price paid.

**The memory result is the interesting one.** QLoRA cut the *resident weight footprint*
by 47% (3.11 → 1.64 GB) — but **peak training VRAM was unchanged** (6.71 → 6.75 GB).
The reason: peak memory ≈ weights + activations (gradients/optimizer are negligible since
the base is frozen). Quantization shrinks only the **weights**; **activations** —
which depend on batch × sequence × layers, not weight precision — are untouched and here
they dominate the peak. The ~1.5 GB saved on weights gets spent right back on
dequantization scratch buffers. Net: a wash.

### The takeaway

> QLoRA shrinks the *static* cost (weights), never the *dynamic* cost (activations).
> It pays off precisely when weights are the term about to break you — i.e. **large**
> models, where resident weights dominate peak memory and a 4× cut is the difference
> between OOM and fitting. At 1.5B on 16 GB, weights were a minority of peak, so we paid
> the speed cost and saw essentially none of the memory benefit.

This is *scale-dependent*: the exact same code on a 13B/70B model would show QLoRA
preventing an OOM that LoRA cannot avoid.

## Extending the experiment

- **Make QLoRA earn its keep:** raise `PER_DEVICE_BATCH_SIZE` aggressively on the QLoRA
  run only and watch LoRA OOM first. Run it as a *separate labeled* experiment — don't
  edit the shared config, or the core A/B is no longer apples-to-apples.
- **Activation memory lever:** flip `GRADIENT_CHECKPOINTING = True` (applied to both runs)
  to see the activation term drop — at a compute cost. This is the orthogonal tool to
  quantization.
- **Bigger model:** swap `MODEL_NAME` for a 7B+ model to move into the
  weights-dominate regime where the peak-VRAM gap becomes dramatic.

## Caveats

- Synthetic-data evaluation makes the *absolute* accuracy a soft proxy (see above).
- "Deterministic" greedy decoding holds on the same machine + library versions; tiny
  floating-point differences across GPUs can occasionally flip a near-tie token.
- The benchmark trains on a 2000-row toy subset for speed; numbers are illustrative of
  the *tradeoff shape*, not production accuracy.