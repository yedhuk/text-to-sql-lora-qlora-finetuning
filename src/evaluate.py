"""
evaluate.py — STEP 3b: measure the accuracy of a trained adapter.

Usage:
    python src/evaluate.py lora
    python src/evaluate.py qlora

For each of the 100 holdout examples it:
  1. generates SQL (greedy) from the schema+question prompt,
  2. builds an in-memory SQLite DB from the CREATE TABLE context and populates it
     with a few DETERMINISTIC synthetic rows (the dataset ships no data),
  3. executes BOTH the gold and the generated query against that same DB,
  4. compares results order-insensitively and categorizes any errors.

Metrics written:
  valid_sql_rate   : generated query executes without error (the hallucination detector)
  exec_match_rate  : generated result == gold result (same rows, order-insensitive)

Outputs:
  outputs/eval_{mode}.json   summary + all 100 records (the full dump)
  outputs/eval_{mode}.md     human-readable side-by-side of every example

NOTE: torch/transformers/peft are imported lazily inside load/generate so the
pure-SQL logic below can be unit-tested without a GPU.
"""

import sys
import os
import re
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SYNTH_ROWS = 8           # synthetic rows inserted per table
SYNTH_SEED = 12345       # deterministic population => re-runnable eval

# ─────────────────────────────────────────────────────────────────────────────
# PURE SQL LOGIC (no torch) — unit-testable on its own
# ─────────────────────────────────────────────────────────────────────────────
_CREATE_RE = re.compile(
    r'create\s+table\s+(?P<name>[`"\[]?\w+[`"\]]?)\s*\((?P<cols>.+?)\)\s*(?:;|$)',
    re.IGNORECASE | re.DOTALL,
)
_CONSTRAINT_KW = ("primary", "foreign", "key", "unique", "constraint", "check")


def _clean_ident(tok: str) -> str:
    return tok.strip().strip('`"[]')


def parse_schema(ddl: str):
    """Return [(table_name, [(col_name, col_type), ...]), ...] from CREATE TABLE DDL."""
    tables = []
    for m in _CREATE_RE.finditer(ddl):
        name = _clean_ident(m.group("name"))
        cols = []
        for part in m.group("cols").split(","):
            tokens = part.strip().split()
            if not tokens:
                continue
            if tokens[0].lower() in _CONSTRAINT_KW:
                continue  # skip table-level constraints
            col_name = _clean_ident(tokens[0])
            col_type = tokens[1] if len(tokens) > 1 else "TEXT"
            cols.append((col_name, col_type))
        tables.append((name, cols))
    return tables


def _gen_value(col_type: str, row_idx: int, rng: random.Random):
    t = col_type.upper()
    if "INT" in t:
        return row_idx + 1                       # 1,2,3,... predictable for aggregates
    if any(k in t for k in ("REAL", "FLOA", "DOUB", "DEC", "NUM")):
        return round(row_idx + 0.5, 2)
    return rng.choice(["alpha", "bravo", "charlie", "delta", "echo"])


def build_db(ddl: str):
    """Create + populate an in-memory SQLite DB from the schema DDL. Returns a
    connection, or raises if the DDL itself is unparseable/invalid."""
    import sqlite3
    con = sqlite3.connect(":memory:")
    con.executescript(ddl)                       # may raise on malformed DDL
    rng = random.Random(SYNTH_SEED)
    for tbl, cols in parse_schema(ddl):
        if not cols:
            continue
        colnames = ",".join(f'"{c}"' for c, _ in cols)
        placeholders = ",".join(["?"] * len(cols))
        for r in range(SYNTH_ROWS):
            vals = [_gen_value(ct, r, rng) for _, ct in cols]
            try:
                con.execute(f'INSERT INTO "{tbl}" ({colnames}) VALUES ({placeholders})', vals)
            except Exception:
                pass  # constraint violation on synthetic data — skip this row
    return con


def run_sql(con, query: str):
    """Execute a query. Returns (status, rows, error_msg)."""
    try:
        cur = con.execute(query)
        return ("ok", cur.fetchall(), None)
    except Exception as e:
        return ("error", None, f"{type(e).__name__}: {e}")


def classify_error(msg: str) -> str:
    m = msg.lower()
    if "no such column" in m or "no such table" in m:
        return "hallucination"
    if "syntax error" in m or "near " in m:
        return "syntax_error"
    return "other_error"


def same_rows(a, b) -> bool:
    """Order-insensitive multiset comparison of result rows."""
    return sorted(map(repr, a)) == sorted(map(repr, b))


def extract_sql(text: str) -> str:
    """Pull the SQL out of the model's continuation."""
    text = text.strip()
    if ";" in text:
        return text[: text.index(";") + 1].strip()
    return text.split("\n")[0].strip()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL (lazy heavy imports)
# ─────────────────────────────────────────────────────────────────────────────
BASE_MODES = ("base_strict", "base_fair")  # raw base model, no adapter (the control)


def load_model_for_eval(mode: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    if mode in BASE_MODES:
        # Raw base model, no adapter. bf16 — the natural "as shipped" baseline.
        tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        if tok.pad_token is None:
            tok.pad_token = "<|endoftext|>"
        base = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
        base.eval()
        return base, tok

    adapter_path = config.adapter_dir(mode)
    tok = AutoTokenizer.from_pretrained(adapter_path)

    # Load the base in the SAME precision it was trained on (evaluate as deployed).
    if mode == "lora":
        base = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
    else:  # qlora
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME, quantization_config=bnb, device_map={"": 0}
        )

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tok


_SQL_START = re.compile(r"(?is)\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b")


def extract_sql_freeform(text: str) -> str:
    """Lenient extraction for the base model's chatty / possibly fenced output."""
    text = text.replace("```sql", "```").replace("```SQL", "```")
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:               # take the first fenced block
            text = parts[1]
    m = _SQL_START.search(text)           # jump to the first SQL keyword
    if m:
        text = text[m.start():]
    text = text.strip()
    if ";" in text:
        return text[: text.index(";") + 1].strip()
    return text.split("\n")[0].strip()


# Instruction for the "fair" base prompt (the base model never saw our template).
_BASE_FAIR_SYSTEM = (
    "You are a SQLite expert. Given a table schema and a question, reply with ONLY "
    "one valid SQLite query and nothing else — no explanation, no markdown fences."
)


def generate_sql(model, tok, context: str, question: str, mode: str) -> str:
    import torch
    if mode == "base_fair":
        messages = [
            {"role": "system", "content": _BASE_FAIR_SYSTEM},
            {"role": "user", "content": f"Schema:\n{context}\n\nQuestion: {question}"},
        ]
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        # lora / qlora / base_strict all use the raw training template.
        prompt = config.build_prompt(context, question)
        input_ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

    input_ids = input_ids.to(model.device)
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,                 # greedy => deterministic eval
            pad_token_id=tok.pad_token_id,
        )
    gen_ids = out[0][input_ids.shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return extract_sql_freeform(text) if mode == "base_fair" else extract_sql(text)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["lora", "qlora", "base_strict", "base_fair"])
    args = parser.parse_args()
    mode = args.mode

    with open(config.TEST_FILE) as f:
        test = [json.loads(line) for line in f]
    print(f"[eval] {len(test)} holdout examples | mode={mode}")

    model, tok = load_model_for_eval(mode)

    records = []
    counts = {"exec_match": 0, "valid_sql": 0, "gold_ok": 0,
              "hallucination": 0, "syntax_error": 0, "other_error": 0, "gold_error": 0}

    for i, ex in enumerate(test):
        ctx, q, gold = ex["context"], ex["question"], ex["answer"]
        gen = generate_sql(model, tok, ctx, q, mode)

        rec = {"i": i, "question": q, "gold_sql": gold, "generated_sql": gen}

        # Build the DB once; run both queries against it.
        try:
            con = build_db(ctx)
        except Exception as e:
            rec.update(status="db_build_error", error=f"{type(e).__name__}: {e}")
            counts["gold_error"] += 1
            records.append(rec)
            continue

        g_status, g_rows, g_err = run_sql(con, gold)
        p_status, p_rows, p_err = run_sql(con, gen)
        con.close()

        if g_status != "ok":
            # Gold itself doesn't run on our synthetic DB — exclude from accuracy.
            rec.update(status="gold_error", error=g_err)
            counts["gold_error"] += 1
        else:
            counts["gold_ok"] += 1
            if p_status == "ok":
                counts["valid_sql"] += 1
                if same_rows(g_rows, p_rows):
                    rec.update(status="match",
                               gold_rows=g_rows[:10], generated_rows=p_rows[:10])
                    counts["exec_match"] += 1
                else:
                    rec.update(status="mismatch",
                               gold_rows=g_rows[:10], generated_rows=p_rows[:10])
            else:
                cat = classify_error(p_err)
                counts[cat] += 1
                rec.update(status=cat, error=p_err)
        records.append(rec)

        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(test)}")

    # ── Metrics ─────────────────────────────────────────────────────────────
    gold_ok = max(counts["gold_ok"], 1)
    summary = {
        "mode": mode,
        "n_total": len(test),
        "n_gold_executable": counts["gold_ok"],
        "valid_sql_rate": round(counts["valid_sql"] / gold_ok, 4),
        "exec_match_rate": round(counts["exec_match"] / gold_ok, 4),
        "hallucinations": counts["hallucination"],
        "syntax_errors": counts["syntax_error"],
        "other_errors": counts["other_error"],
        "gold_unexecutable": counts["gold_error"],
    }

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(config.OUTPUT_DIR, f"eval_{mode}.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2, default=str)

    _write_markdown(mode, summary, records)

    print("\n[eval summary]")
    print(json.dumps(summary, indent=2))
    print(f"[saved] {json_path}")


def _write_markdown(mode, summary, records):
    md_path = os.path.join(config.OUTPUT_DIR, f"eval_{mode}.md")
    lines = [f"# Eval — {mode}", "", "## Summary", "```json",
             json.dumps(summary, indent=2), "```", "", "## Per-example", ""]
    for r in records:
        lines.append(f"### {r['i']}  —  `{r.get('status', '?')}`")
        lines.append(f"**Q:** {r['question']}  ")
        lines.append(f"**gold:** `{r['gold_sql']}`  ")
        lines.append(f"**gen :** `{r['generated_sql']}`  ")
        if "error" in r:
            lines.append(f"**error:** `{r['error']}`  ")
        elif r.get("status") in ("match", "mismatch"):
            lines.append(f"**gold rows:** `{r.get('gold_rows')}`  ")
            lines.append(f"**gen rows :** `{r.get('generated_rows')}`  ")
        lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()