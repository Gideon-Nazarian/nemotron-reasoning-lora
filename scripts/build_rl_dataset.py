"""Build the RL training dataset for GRPO.

GRPO doesn't need solver-generated reasoning — only (prompt, gold_answer) pairs
plus a verifiable reward function. So unlike SFT (where we filter to solver-
matching examples), here we can use ALL of train.csv as the prompt pool. This
unlocks the categories where our solvers don't work (bit_manipulation 90%
uncovered, transformation_rules 100% uncovered).

Output format: a JSONL file with one record per problem:
  {
    "prompt": [{"role": "user", "content": "<prompt text + PROMPT_SUFFIX>"}],
    "gold_answer": "<expected boxed answer>",
    "category": "<puzzle category>",
    "problem_id": "<train.csv id>"
  }

TRL's GRPOTrainer expects "prompt" in conversational format (list of message
dicts). The "gold_answer" column gets passed to our reward function as a kwarg.

Usage:
    uv run python scripts/build_rl_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path(__file__).parent.parent
TRAIN_CSV = ROOT / "data" / "train.csv"
CATEGORIES_CSV = ROOT / "notes" / "train-categories.csv"
OUT_JSONL = ROOT / "data" / "rl-training-data.jsonl"

PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def main() -> int:
    if not TRAIN_CSV.exists():
        print(f"missing {TRAIN_CSV}", file=sys.stderr)
        return 1

    df = pd.read_csv(TRAIN_CSV)
    cats = pd.read_csv(CATEGORIES_CSV)
    df = df.merge(cats[["id", "category_by_prompt"]], on="id")

    records = []
    by_cat: Counter[str] = Counter()
    for _, row in df.iterrows():
        records.append({
            "prompt": [{"role": "user", "content": row["prompt"] + PROMPT_SUFFIX}],
            "gold_answer": str(row["answer"]).strip(),
            "category": row["category_by_prompt"],
            "problem_id": row["id"],
        })
        by_cat[row["category_by_prompt"]] += 1

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(records)} RL training records to {OUT_JSONL}")
    print("\nBy category:")
    for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat:25s}: {n:>5d} ({n / len(records) * 100:.1f}%)")

    # Compare to SFT coverage so we can see the unlock
    sft_path = ROOT / "data" / "training-data.jsonl"
    if sft_path.exists():
        sft_ids = set()
        with sft_path.open() as f:
            for line in f:
                sft_ids.add(json.loads(line)["problem_id"])
        rl_ids = {r["problem_id"] for r in records}
        only_rl = rl_ids - sft_ids
        print(f"\nProblems covered by RL but NOT by SFT: {len(only_rl)} ({len(only_rl)/len(records)*100:.1f}%)")
        # Per-category breakdown of the unlock
        sft_cats: Counter[str] = Counter()
        with sft_path.open() as f:
            for line in f:
                sft_cats[json.loads(line)["category"]] += 1
        print("\nCoverage gain (RL has, SFT lacks) by category:")
        for cat, n_total in by_cat.most_common():
            n_sft = sft_cats.get(cat, 0)
            gap = n_total - n_sft
            print(f"  {cat:25s}: SFT had {n_sft:>5d} of {n_total:>5d} ({n_sft/n_total*100:>5.1f}%), RL adds {gap:>5d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
