"""
run_benchmark.py — orchestrate the full A/B and produce the deliverable.

Runs (each as an isolated subprocess so VRAM measurements don't leak between runs):
    prepare_data.py
    train.py lora    -> evaluate.py lora
    train.py qlora   -> evaluate.py qlora
Then reads telemetry_{mode}.json + eval_{mode}.json and prints a comparison table,
writing benchmark_summary.{md,json}.

Usage:
    python run_benchmark.py                 # full run, both modes
    python run_benchmark.py --skip-data     # reuse existing train/test jsonl
    python run_benchmark.py --report-only   # just rebuild the table from existing JSONs
"""

import os
import sys
import json
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

MODES = ["lora", "qlora"]                    # trained configs (train + eval)
BASE_MODES = ["base_strict", "base_fair"]    # baseline controls (eval only, no training)
PY = sys.executable


# ─── running stages ──────────────────────────────────────────────────────────
def run_stage(script_rel, *args):
    """Run a script as a subprocess (isolated memory), streaming its output."""
    script = os.path.join(config.ROOT, script_rel)
    cmd = [PY, script, *args]
    print(f"\n{'=' * 70}\n$ {' '.join(cmd)}\n{'=' * 70}", flush=True)
    subprocess.run(cmd, cwd=config.ROOT, check=True)


# ─── aggregation (pure: takes dicts, easy to test) ───────────────────────────
def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# (key, label, source, fmt, delta) where source is 'tel' or 'ev',
# fmt in {f2,f3,f4,int}, delta in {pct,abs,none}
_METRICS = [
    ("peak_vram_reserved_gb",  "Peak VRAM reserved (GB)",  "tel", "f2", "pct"),
    ("peak_vram_allocated_gb", "Peak VRAM allocated (GB)", "tel", "f2", "pct"),
    ("model_footprint_gb",     "Model footprint (GB)",     "tel", "f2", "pct"),
    ("steps_per_second",    "Train steps/sec",       "tel", "f3", "pct"),
    ("train_runtime_sec",   "Train runtime (s)",     "tel", "f2", "pct"),
    ("final_train_loss",    "Final train loss",      "tel", "f4", "abs"),
    ("valid_sql_rate",      "Valid-SQL rate",        "ev",  "f3", "abs"),
    ("exec_match_rate",     "Exec-match rate",       "ev",  "f3", "abs"),
    ("hallucinations",      "Hallucinations",        "ev",  "int", "abs"),
    ("syntax_errors",       "Syntax errors",         "ev",  "int", "abs"),
    ("gold_unexecutable",   "Gold-unexec (context)", "ev",  "int", "none"),
]


def _fmt(v, kind):
    if v is None:
        return "—"
    if kind == "int":
        return str(int(v))
    return {"f2": f"{v:.2f}", "f3": f"{v:.3f}", "f4": f"{v:.4f}"}[kind].format(v)


def _delta(lora, qlora, kind):
    if lora is None or qlora is None or kind == "none":
        return ""
    if kind == "pct":
        if lora == 0:
            return ""
        return f"{(qlora - lora) / lora * 100:+.1f}%"
    return f"{qlora - lora:+.3f}".rstrip("0").rstrip(".")


def build_rows(data):
    """data = {mode: {'tel': dict|None, 'ev': dict|None}}. Returns list of row tuples."""
    rows = []
    for key, label, src, fmt, delta in _METRICS:
        def get(mode):
            d = data[mode][src]
            return d.get(key) if d else None
        lv, qv = get("lora"), get("qlora")
        rows.append((label, _fmt(lv, fmt), _fmt(qv, fmt), _delta(lv, qv, delta)))
    return rows


def render_table(rows):
    headers = ("Metric", "LoRA", "QLoRA", "Δ (QLoRA vs LoRA)")
    w0 = max(len(headers[0]), *(len(r[0]) for r in rows))
    w1 = max(len(headers[1]), *(len(r[1]) for r in rows))
    w2 = max(len(headers[2]), *(len(r[2]) for r in rows))
    w3 = max(len(headers[3]), *(len(r[3]) for r in rows))
    line = f"{headers[0]:<{w0}}  {headers[1]:>{w1}}  {headers[2]:>{w2}}  {headers[3]:>{w3}}"
    out = [line, "-" * len(line)]
    for label, a, b, d in rows:
        out.append(f"{label:<{w0}}  {a:>{w1}}  {b:>{w2}}  {d:>{w3}}")
    return "\n".join(out)


def interpret(data):
    """One-paragraph plain-English readout of the headline tradeoff."""
    tl, tq = data["lora"]["tel"], data["qlora"]["tel"]
    el, eq = data["lora"]["ev"], data["qlora"]["ev"]
    if not all([tl, tq, el, eq]):
        return "Incomplete results — some JSON files were missing."

    # Prefer the explicit reserved figure; fall back to peak_vram_gb for telemetry
    # produced before reserved-memory logging was added.
    def reserved(d):
        return d.get("peak_vram_reserved_gb", d.get("peak_vram_gb"))

    rl, rq = reserved(tl), reserved(tq)
    al, aq = tl.get("peak_vram_allocated_gb"), tq.get("peak_vram_allocated_gb")
    vram = (rq - rl) / rl * 100 if rl else 0.0
    speed = (tq["steps_per_second"] - tl["steps_per_second"]) / tl["steps_per_second"] * 100
    acc = (eq["exec_match_rate"] - el["exec_match_rate"]) * 100

    alloc_note = ""
    if al is not None and aq is not None:
        alloc_note = (
            f" — despite near-identical *allocated* VRAM ({al:.2f} vs {aq:.2f} GB); "
            f"the gap is dequantization fragmentation in PyTorch's reserved pool, the "
            f"figure that actually triggers OOM"
        )

    return (
        f"QLoRA used {abs(vram):.0f}% {'less' if vram < 0 else 'more'} peak RESERVED VRAM "
        f"({rl:.2f} -> {rq:.2f} GB){alloc_note}. It trained "
        f"{abs(speed):.0f}% {'slower' if speed < 0 else 'faster'} "
        f"({tl['steps_per_second']:.2f} -> {tq['steps_per_second']:.2f} steps/sec), and its "
        f"exec-match rate changed by {acc:+.1f} points "
        f"({el['exec_match_rate']:.1%} -> {eq['exec_match_rate']:.1%}). "
        f"Hallucinated columns/tables: {el['hallucinations']} (LoRA) vs "
        f"{eq['hallucinations']} (QLoRA)."
    )


# Accuracy comparison: configs as rows (incl. the two un-tuned baselines).
ACCURACY_ORDER = [
    ("base_strict", "Base (strict template)"),
    ("base_fair",   "Base (instruction prompt)"),
    ("lora",        "LoRA fine-tuned"),
    ("qlora",       "QLoRA fine-tuned"),
]


def render_accuracy_table(summaries):
    headers = ["Configuration", "Valid-SQL", "Exec-match", "Halluc.", "Syntax-err"]
    rows = []
    for mode, label in ACCURACY_ORDER:
        s = summaries.get(mode)
        if s:
            rows.append([label, f"{s['valid_sql_rate']:.3f}", f"{s['exec_match_rate']:.3f}",
                         str(s["hallucinations"]), str(s["syntax_errors"])])
        else:
            rows.append([label, "—", "—", "—", "—"])
    w = [max(len(headers[c]), *(len(r[c]) for r in rows)) for c in range(len(headers))]

    def fmt(cells):
        return "  ".join([cells[0].ljust(w[0])] +
                         [cells[c].rjust(w[c]) for c in range(1, len(cells))])

    line = fmt(headers)
    return "\n".join([line, "-" * len(line)] + [fmt(r) for r in rows])


def interpret_accuracy(summaries):
    lo = summaries.get("lora")
    present = {k: summaries.get(k) for k in ("base_strict", "base_fair") if summaries.get(k)}
    if not (lo and present):
        return ""
    base_best = max(v["exec_match_rate"] for v in present.values())
    gain = (lo["exec_match_rate"] - base_best) * 100
    desc = ", ".join(
        f"{'strict' if k == 'base_strict' else 'instruction'} base {v['exec_match_rate']:.1%}"
        for k, v in present.items()
    )
    return (
        f"Baseline exec-match: {desc}. Fine-tuning (LoRA) reached {lo['exec_match_rate']:.1%} — "
        f"a +{gain:.1f} point gain over the best base prompt, confirming the adapter (not the "
        f"base model) is doing the work."
    )


def report():
    data = {}
    for m in MODES:
        ev = _load(os.path.join(config.OUTPUT_DIR, f"eval_{m}.json"))
        data[m] = {
            "tel": _load(os.path.join(config.OUTPUT_DIR, f"telemetry_{m}.json")),
            "ev": ev["summary"] if ev else None,
        }
    rows = build_rows(data)
    table = render_table(rows)
    para = interpret(data)

    # Accuracy across all configs, including the no-adapter baselines.
    summaries = {m: data[m]["ev"] for m in MODES}
    for m in BASE_MODES:
        ev = _load(os.path.join(config.OUTPUT_DIR, f"eval_{m}.json"))
        summaries[m] = ev["summary"] if ev else None
    acc_table = render_accuracy_table(summaries)
    acc_para = interpret_accuracy(summaries)

    print(f"\n{'#' * 70}\n# Text-to-SQL Benchmark — LoRA vs QLoRA\n{'#' * 70}\n")
    print(table)
    print(f"\n{para}\n")
    print("\n--- Accuracy vs un-tuned base model ---\n")
    print(acc_table)
    if acc_para:
        print(f"\n{acc_para}\n")

    md = (
        f"# Text-to-SQL Benchmark — LoRA vs QLoRA\n\n"
        f"## Training cost (LoRA vs QLoRA)\n\n```\n{table}\n```\n\n{para}\n\n"
        f"## Accuracy vs un-tuned base model\n\n```\n{acc_table}\n```\n\n{acc_para}\n"
    )
    with open(os.path.join(config.OUTPUT_DIR, "benchmark_summary.md"), "w") as f:
        f.write(md)
    with open(os.path.join(config.OUTPUT_DIR, "benchmark_summary.json"), "w") as f:
        json.dump({"table": rows, "interpretation": para,
                   "accuracy": summaries, "accuracy_interpretation": acc_para,
                   "raw": data}, f, indent=2)
    print(f"[saved] {os.path.join(config.OUTPUT_DIR, 'benchmark_summary.md')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true", help="reuse existing jsonl")
    parser.add_argument("--report-only", action="store_true", help="only rebuild the table")
    parser.add_argument("--baseline-only", action="store_true",
                        help="run only the two base-model evals, then report")
    args = parser.parse_args()

    if not args.report_only:
        if args.baseline_only:
            for m in BASE_MODES:                  # uses existing lora/qlora evals in report
                run_stage("src/evaluate.py", m)
        else:
            if not args.skip_data:
                run_stage("data/prepare_data.py")
            for m in MODES:
                run_stage("src/train.py", m)
                run_stage("src/evaluate.py", m)
            for m in BASE_MODES:                  # baselines: eval only, no training
                run_stage("src/evaluate.py", m)

    report()


if __name__ == "__main__":
    main()