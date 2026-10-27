"""Build a stratified subset of training-data.jsonl.

Takes N examples per category (default 150 — the minimum any category has),
shuffles deterministically with seed=42, writes to data/training-stratified.jsonl.

Usage:
    uv run python scripts/build_stratified_subset.py [n_per_category]
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
IN_PATH = ROOT / "data" / "training-data.jsonl"
OUT_PATH = ROOT / "data" / "training-stratified.jsonl"


def main() -> int:
    n_per_cat = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    seed = 42

    by_cat: dict[str, list[dict]] = defaultdict(list)
    with IN_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            by_cat[rec["category"]].append(rec)

    print(f"Loaded {sum(len(v) for v in by_cat.values())} records across {len(by_cat)} categories")
    print(f"Sampling {n_per_cat} per category (seed={seed})")

    rng = random.Random(seed)
    chosen: list[dict] = []
    for cat, records in sorted(by_cat.items()):
        n_avail = len(records)
        n_take = min(n_per_cat, n_avail)
        sample = rng.sample(records, n_take)
        chosen.extend(sample)
        print(f"  {cat:25s}: took {n_take} of {n_avail}")

    rng.shuffle(chosen)  # final shuffle across categories so training sees mixed types

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for rec in chosen:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(chosen)} stratified records to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
