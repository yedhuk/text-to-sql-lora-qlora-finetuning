# Text-to-SQL: LoRA vs QLoRA A/B Benchmark

Fine-tune the same model **twice** — once with LoRA (16-bit base), once with QLoRA
(4-bit base) — on a Text-to-SQL task, then measure what quantization actually costs
and saves: peak VRAM, training speed, and SQL accuracy.

The point of the project isn't to "win" — it's to produce an honest, reproducible
side-by-side and understand *where* QLoRA's famous memory savings come from (and,
just as importantly, where they don't). The headline result below is a good example:
the obvious metric was misleading, and the truth only showed up once the *right*
quantity was measured the *right* way.

**LoRA Train Monitoring**
<p>
  <img src="https://raw.githubusercontent.com/yedhuk/text-to-sql-lora-qlora-finetuning/main/monitoring/lora_train.png" alt="lora_train_monitoring">
</p>

**LoRA Eval Monitoring**
<p>
  <img src="https://raw.githubusercontent.com/yedhuk/text-to-sql-lora-qlora-finetuning/main/monitoring/lora_eval.png" alt="lora_eval_monitoring">
</p>

**QLoRA Train Monitoring**
<p>
  <img src="https://raw.githubusercontent.com/yedhuk/text-to-sql-lora-qlora-finetuning/main/monitoring/qlora_train.png" alt="qlora_train_monitoring">
</p>

**QLoRA Eval Monitoring**
<p>
  <img src="https://raw.githubusercontent.com/yedhuk/text-to-sql-lora-qlora-finetuning/main/monitoring/qlora_eval.png" alt="qlora_eval_monitoring">
</p>


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

Developed on an **RTX 5080 (Blackwell, `sm_120`)**, which requires recent CUDA kernels.
Install `torch` from the CUDA 12.8 index **first**, then the rest:

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
`--report-only` (rebuild the report from existing result files without retraining),
`--baseline-only` (evaluate the un-tuned base model only, then report).

Or run stages by hand:

```bash
python data/prepare_data.py
python src/train.py lora    && python src/evaluate.py lora
python src/train.py qlora   && python src/evaluate.py qlora
python src/evaluate.py base_strict     # baseline: raw template, no adapter
python src/evaluate.py base_fair       # baseline: instruction prompt, no adapter
```

### Monitoring

`train.py` self-reports both `allocated` and `reserved` peak VRAM (see
[Measurement notes](#measurement-notes-read-this)). To watch the GPU live in parallel,
`nvtop` (per-process GPU detail) + `btop` (whole-system context) work well, or log a
time series for later plotting:

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm \
           --format=csv -l 1 > gpu_log_lora.csv
```

## Outputs (in `outputs/`)

- `telemetry_{mode}.json` — speed + three memory figures:
  `model_footprint_gb` (resident weights), `peak_vram_allocated_gb` (live tensors),
  `peak_vram_reserved_gb` (the driver-visible pool that nvtop shows and that OOMs you)
- `eval_{mode}.json` — accuracy summary + every one of the 100 examples (gold vs generated);
  `mode` is `lora`, `qlora`, `base_strict`, or `base_fair`
- `eval_{mode}.md` — the same dump, readable, for eyeballing failures side by side
- `benchmark_summary.{md,json}` — the deliverable: the LoRA-vs-QLoRA training-cost table
  **and** the accuracy table across all four configs (incl. the un-tuned baselines)

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

**Baselines (controls).** The un-tuned base model is evaluated two ways, to separate
*format-following* from *SQL capability*. `base_strict` feeds the exact raw template the
tuned models use, so it measures the adapter's full value; `base_fair` uses a proper chat
instruction ("reply with only one SQLite query") plus lenient extraction, so it measures
what Qwen2.5 can already do when prompted well. The gap between them is how much of
fine-tuning's value was simply teaching the output format.

## Results (RTX 5080, 1.5B model, 2000 train / 100 test, 3 epochs)

| Metric | LoRA | QLoRA | Δ (QLoRA vs LoRA) |
|---|---:|---:|---:|
| Peak VRAM reserved (GB) | 15.06 | 15.03 | −0.2% |
| Peak VRAM allocated (GB) | 6.71 | 6.75 | +0.7% |
| Model footprint (GB) | 3.11 | 1.64 | **−47.3%** |
| Train steps/sec | 7.527 | 5.300 | **−29.6%** |
| Train runtime (s) | 99.64 | 141.51 | **+42.0%** |
| Final train loss | 0.0685 | 0.0676 | −0.001 |
| Valid-SQL rate | 0.990 | 0.990 | +0 |
| Exec-match rate | 0.929 | 0.939 | +0.01 |
| Hallucinations | 0 | 1 | +1 |
| Syntax errors | 1 | 0 | −1 |
| Gold-unexecutable (context) | 2 | 2 | — |

Inference memory (from live monitoring during eval): **LoRA ≈ 3.5 GB, QLoRA ≈ 1.5 GB.**

**Accuracy across all four configurations** (exec-match against the synthetic SQLite DBs):

| Configuration | Valid-SQL | Exec-match | Hallucinations | Syntax errors |
|---|---:|---:|---:|---:|
| Base — strict template | 0.775 | 0.735 | 4 | 17 |
| Base — instruction prompt | 0.969 | 0.918 | 2 | 1 |
| LoRA fine-tuned | 0.990 | 0.929 | 0 | 1 |
| QLoRA fine-tuned | 0.990 | 0.939 | 1 | 0 |

### What it means

**Training memory: tied — and the cause is *not* quantization.** Both runs drove
PyTorch's reserved pool to ~15 GB, while only ~6.7 GB of tensors were ever live
(allocated) at any instant — in *both* runs. That ~8 GB gap between reserved and
allocated is caching-allocator overhead, and it appears equally under LoRA and QLoRA.
The likely driver is fragmentation: variable-length padded batches create
constantly-changing activation tensor sizes, and with ~15 GB of free VRAM the allocator
greedily grabs large segments and never releases them. None of that depends on 4-bit vs
16-bit weights, so both runs land at the same ceiling. **QLoRA provided no peak-VRAM
benefit during training at this scale.**

**Quality: a tie.** Loss curves are nearly identical and both models produce valid,
executable SQL ~99% of the time. The exec-match gap (1 example out of 98) and the
hallucination/syntax differences (single examples) are noise. A 4-bit base + 16-bit
adapter learned this task just as well as a 16-bit base — the central QLoRA claim,
reproduced.

**Prompting was a bigger lever than fine-tuning.** This is the most striking result, and
it only appears because of the baseline control. Switching the *base* model from the raw
template to a proper instruction prompt lifted exec-match **+18 points** (0.735 → 0.918)
and cut syntax errors from **17 to 1** — at zero training cost. Fine-tuning on top of a
good prompt then added only ~+1 point (0.918 → 0.929), within noise on 98 examples. Read
like-for-like, though, fine-tuning *is* real: a tuned model on the **bare** template
(0.929) matches the base model given a **careful instruction** (0.918). So fine-tuning's
genuine product here was **internalizing the prompt engineering** — the tuned models emit
clean SQL from a minimal `[SQL]` cue with no instruction needed — plus a small reliability
gain (valid-SQL 0.969 → 0.990). It did not teach new SQL reasoning the base model lacked.
For an easy, synthetic dataset and a capable instruct model, that's expected — and a
useful reminder to try better prompting before reaching for fine-tuning.

**Speed: QLoRA's one real, robust cost.** QLoRA trained ~30% slower per step (+42%
wall-clock). Its 4-bit weights are dequantized to bf16 on the fly for every matmul, and
that overhead is exactly the price paid. This is the only large, unambiguous difference
between the two runs.

**Where QLoRA actually wins** is *outside* the training peak: the static weight footprint
(−47%, 3.11 → 1.64 GB) and **inference memory** (~1.5 vs ~3.5 GB at eval). Inference has
no backward pass and runs at batch 1, so there's almost no allocator churn to drown the
4-bit saving — the footprint advantage shows up cleanly. QLoRA's memory edge is
fundamentally a *deployment/inference* property here, not a training one.

### Measurement notes (read this)

This benchmark's headline conclusion flipped **three times** depending on what was
measured, and that's the most useful lesson in the whole project:

1. `max_memory_allocated()` (live tensors) → "memory is a wash" (~6.7 GB both).
2. A live `nvtop` **snapshot** mid-run → "QLoRA uses ~2×" (caught LoRA at a low instant).
3. `max_memory_reserved()` **maxed over the full run** → "tied at ~15 GB; it's fragmentation."

Only #3 answers "will this OOM?". A single live snapshot can miss the high-water mark
entirely; `allocated` undercounts the driver-visible footprint. **Quote whole-run
`reserved` for capacity planning, and always cross-check allocated vs reserved** — their
divergence tells you whether you're weight-bound or allocator/fragmentation-bound.

### The takeaway

> QLoRA shrinks the *static* cost (weights), never the *dynamic* cost
> (activations + allocator behaviour). For **training** at 1.5B on a 16 GB card it was
> pure downside — same peak memory, same accuracy, ~30% slower. Its real payoff is
> (a) at **large scale**, where resident weights dominate peak memory and a 4× cut is the
> difference between OOM and fitting, and (b) at **inference**, where the footprint
> dominates and there's no backward-pass churn to mask it.

This is *scale-dependent*: the same code on a 13B/70B model would show QLoRA preventing an
OOM that LoRA cannot avoid.

And on accuracy:

> A good prompt got ~92%; fine-tuning got ~93%. For this task and model, **prompting was
> the bigger lever** — fine-tuning's real value was making clean SQL automatic from a
> minimal prompt, not adding capability. Try prompt engineering before fine-tuning; reach
> for fine-tuning when you need format reliability, lower latency, shorter prompts, or
> behaviour a prompt can't buy.

## Extending the experiment

- **Unmask the hidden training benefit (do this first):** the ~15 GB ceiling is likely
  fragmentation, not a hard requirement. Re-run with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_benchmark.py --skip-data`.
  If reserved drops toward ~7 GB (close to allocated) for both runs, fragmentation is
  confirmed — and QLoRA's lower footprint should *then* translate into lower reserved,
  giving it the training headroom that's currently masked.
- **Make QLoRA earn its keep (do this second):** with fragmentation handled, raise
  `PER_DEVICE_BATCH_SIZE` aggressively on the QLoRA run only and watch LoRA OOM first.
  Run it as a *separate labeled* experiment — don't edit the shared config. (As-is, both
  runs sit at ~15 GB, so a bigger batch would OOM both together — QLoRA has no headroom
  edge until fragmentation is removed.)
- **Activation memory lever:** flip `GRADIENT_CHECKPOINTING = True` (applied to both runs)
  to attack the activation term directly, at a compute cost. Orthogonal to quantization.
- **Bigger model:** swap `MODEL_NAME` for a 7B+ model to move into the weights-dominate
  regime where the peak-VRAM gap becomes dramatic.

## Caveats

- Synthetic-data evaluation makes the *absolute* accuracy a soft proxy (see above);
  trust deltas and hallucination counts.
- "Deterministic" greedy decoding holds on the same machine + library versions; tiny
  floating-point differences across GPUs can occasionally flip a near-tie token.
- The benchmark trains on a 2000-row toy subset for speed; numbers illustrate the
  *tradeoff shape*, not production accuracy.
- All numbers are tied to this GPU (16 GB) and model size (1.5B). Memory conclusions in
  particular are scale-dependent; the speed and quality conclusions generalize better.