"""
prepare_data.py — STEP 1: build the toy dataset.

Downloads b-mc2/sql-create-context, takes a small disjoint train/test split,
formats each example into the strict template (via config.build_*), and writes
data/train.jsonl and data/test.jsonl.

Run:  python data/prepare_data.py
"""

import json
import sys
import os

# Make the repo root importable so `import config` works regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
import config


def main():
    print(f"Loading {config.DATASET_NAME} ...")
    # This dataset only has a 'train' split; we'll slice our own test set out of it.
    ds = load_dataset(config.DATASET_NAME, split="train")
    print(f"  full dataset: {len(ds):,} rows | columns: {ds.column_names}")

    # Shuffle once with the fixed seed so the split is deterministic & reproducible.
    ds = ds.shuffle(seed=config.SEED)

    need = config.N_TRAIN + config.N_TEST
    if len(ds) < need:
        raise ValueError(f"Dataset has {len(ds)} rows but we need {need}.")

    train_rows = ds.select(range(config.N_TRAIN))
    test_rows = ds.select(range(config.N_TRAIN, config.N_TRAIN + config.N_TEST))
    print(f"  -> {len(train_rows)} train / {len(test_rows)} test (disjoint)")

    def to_record(row):
        ctx, q, ans = row["context"], row["question"], row["answer"]
        return {
            # raw fields (evaluate.py needs context + answer to run SQL)
            "context": ctx,
            "question": q,
            "answer": ans,
            # pre-formatted views so you can eyeball the template on disk
            "prompt": config.build_prompt(ctx, q),
            "completion": ans.strip(),
            "text": config.build_full_text(ctx, q, ans),  # EOS appended in train.py
        }

    os.makedirs(config.DATA_DIR, exist_ok=True)
    for split_rows, path in [(train_rows, config.TRAIN_FILE), (test_rows, config.TEST_FILE)]:
        with open(path, "w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(to_record(row), ensure_ascii=False) + "\n")
        print(f"  wrote {path}")

    # Show one formatted example so you can confirm the template is correct.
    print("\n─── sample formatted training example ───")
    print(to_record(train_rows[0])["text"])
    print("─────────────────────────────────────────")


if __name__ == "__main__":
    main()