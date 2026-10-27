"""Run programmatic solvers over train.csv and report per-category accuracy.

Validates that our solvers actually solve the problems they're supposed to
before we use their outputs as training data.

Usage:
    uv run python scripts/generate_reasoning.py                  # all categories
    uv run python scripts/generate_reasoning.py unit_conversion  # just one
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from solvers import bit_manipulation, cipher, gravity, roman_numeral, unit_conversion
from solvers.types import Solution

ROOT = Path(__file__).parent.parent
TRAIN_CSV = ROOT / "data" / "train.csv"
CATEGORIES_CSV = ROOT / "notes" / "train-categories.csv"


# Map category name → solver module (each module must have a `solve(prompt)` function)
SOLVERS = {
    "unit_conversion": unit_conversion,
    "gravity": gravity,
    "numeral_system": roman_numeral,  # category name from train-categories.csv
    "bit_manipulation": bit_manipulation,
    "cipher": cipher,
}


def main() -> int:
    # Filter to requested category if specified
    requested = sys.argv[1] if len(sys.argv) > 1 else None
    if requested and requested not in SOLVERS:
        print(f"Unknown category: {requested}. Available: {list(SOLVERS)}", file=sys.stderr)
        return 1
    categories = [requested] if requested else list(SOLVERS)

    # Load problems + their assigned categories
    df = pd.read_csv(TRAIN_CSV)
    cats = pd.read_csv(CATEGORIES_CSV)
    df = df.merge(cats[["id", "category_by_prompt"]], on="id")

    for category in categories:
        solver = SOLVERS[category]
        subset = df[df["category_by_prompt"] == category].reset_index(drop=True)
        print(f"\n{'='*70}\nCategory: {category}  (n={len(subset)})\n{'='*70}")

        n_attempted = 0
        n_succeeded = 0
        n_correct = 0
        failures = []
        for _, row in subset.iterrows():
            sol: Solution = solver.solve(str(row["prompt"]))
            n_attempted += 1
            if sol.success:
                n_succeeded += 1
                if sol.answer.strip() == str(row["answer"]).strip():
                    n_correct += 1
                else:
                    if len(failures) < 5:
                        failures.append((row["id"], row["answer"], sol.answer, sol.notes))
            else:
                if len(failures) < 5:
                    failures.append((row["id"], row["answer"], "(solver failed)", sol.notes))

        print(f"Attempted:           {n_attempted}")
        print(f"Solver claimed success: {n_succeeded}  ({n_succeeded/max(n_attempted,1)*100:.1f}%)")
        print(f"Actually correct:    {n_correct}  ({n_correct/max(n_attempted,1)*100:.1f}%)")
        print(f"Accuracy where solver succeeded: {n_correct/max(n_succeeded,1)*100:.1f}%")

        if failures:
            print(f"\nFirst {len(failures)} mismatches:")
            for pid, gold, predicted, notes in failures:
                print(f"  id={pid}  gold={gold!r}  predicted={predicted!r}  notes={notes}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
