"""Generate SFT training data by running all solvers over train.csv and
keeping only entries where the solver's answer exactly matches the gold.

Output: data/training-data.jsonl (one JSON record per training example).
Each record:
  {
    "problem_id": "...",
    "category": "<solver_category>",
    "prompt": "<original train.csv prompt>",
    "reasoning": "<solver-generated concise CoT>",
    "answer": "<solver answer == gold>",
    "gold": "<train.csv answer>"
  }

The training notebook will then format each record into:
  user_text:      prompt + PROMPT_SUFFIX
  assistant_text: reasoning + "\n</think>\n\\boxed{" + answer + "}"

Usage:
    uv run python scripts/build_training_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

from solvers import (
    bit_manipulation,
    cipher,
    gravity,
    roman_numeral,
    transformation_rules,
    unit_conversion,
)
from solvers.types import Solution

ROOT = Path(__file__).parent.parent
TRAIN_CSV = ROOT / "data" / "train.csv"
CATEGORIES_CSV = ROOT / "notes" / "train-categories.csv"
OUT_JSONL = ROOT / "data" / "training-data.jsonl"

SOLVERS = {
    "unit_conversion": unit_conversion,
    "gravity": gravity,
    "numeral_system": roman_numeral,
    "bit_manipulation": bit_manipulation,
    "cipher": cipher,
    "transformation_rules": transformation_rules,
}


def main() -> int:
    if not TRAIN_CSV.exists():
        print(f"missing {TRAIN_CSV}", file=sys.stderr)
        return 1

    df = pd.read_csv(TRAIN_CSV)
    cats = pd.read_csv(CATEGORIES_CSV)
    df = df.merge(cats[["id", "category_by_prompt"]], on="id")

    records = []
    by_cat: Counter[str] = Counter()
    by_outcome: Counter[str] = Counter()

    for _, row in df.iterrows():
        cat = row["category_by_prompt"]
        solver = SOLVERS.get(cat)
        if solver is None:
            by_outcome[f"skipped_no_solver"] += 1
            continue

        sol: Solution = solver.solve(str(row["prompt"]))
        if not sol.success:
            by_outcome["solver_failed"] += 1
            continue

        gold = str(row["answer"]).strip()
        if sol.answer.strip() != gold:
            by_outcome["wrong_answer"] += 1
            continue

        # Keep this clean example
        records.append({
            "problem_id": row["id"],
            "category": cat,
            "prompt": row["prompt"],
            "reasoning": sol.reasoning,
            "answer": sol.answer,
            "gold": gold,
        })
        by_cat[cat] += 1
        by_outcome["kept"] += 1

    # Write JSONL
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(records)} training records to {OUT_JSONL}")
    print(f"\nBy category:")
    for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat:25s}: {n}")
    print(f"\nBy outcome:")
    for outcome, n in sorted(by_outcome.items(), key=lambda x: -x[1]):
        print(f"  {outcome:25s}: {n}")
    print(f"\nKept / total processed: {by_outcome['kept']} / {len(df)} = {by_outcome['kept']/len(df)*100:.1f}%")

    # Reasoning length stats (chars, since we don't have a tokenizer here)
    if records:
        lens = [len(r["reasoning"]) for r in records]
        print(f"\nReasoning length (chars): min={min(lens)}, median={sorted(lens)[len(lens)//2]}, "
              f"mean={sum(lens)/len(lens):.0f}, max={max(lens)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
