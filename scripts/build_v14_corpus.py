"""Build v14-C training corpus: dgxchen non-cipher + our cipher v2 + Tong augmenters.

v14-C design (locked 2026-05-20):
  - Base: dgxchen's recipe (unchanged from v13)
  - Delta 1 (cipher swap): replace dgxchen's 1,656 cipher CoTs with our cipher v2's 1,576
    Wonderland-wordlist-fallback CoTs (100% gold coverage on train.csv).
  - Delta 2 (augmenters): add Tong's 8,463 augmenter records — pure format/pattern teaching,
    from his public 0.85 LB submission.

Output schema (JSONL):
  {
    "id": str,
    "prompt": str,
    "answer": str,
    "type": str,             # category (cipher, bit_manipulation, ..., lstrip, matching, ...)
    "generated_cot": str,    # reasoning text for reasoner records; "" for augmenter records
    "record_kind": "reasoner" | "augmenter"   # discriminates template
  }

The v14 notebook uses `record_kind` to apply the right template:
  - reasoner: assistant_content = generated_cot + "\\n</think>\\n\\boxed{<answer>}"
              user_content = prompt + "\\nPlease put your final answer inside `\\boxed{}`..."
  - augmenter: assistant_content = answer (raw, no wrapping)
               user_content = prompt (no boxed suffix)

Usage:
  python3 scripts/build_v14_corpus.py
  → writes data/v14-training.jsonl
"""
from __future__ import annotations
import json
import sys
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DGXCHEN_CSV = REPO / "data" / "dgxchen-corpus" / "problem_ids_matched.csv"
OUR_SOLVER_JSONL = REPO / "data" / "training-data.jsonl"
OUTPUT_JSONL = REPO / "data" / "v14-training.jsonl"

# Make Tong augmenters importable
sys.path.insert(0, str(REPO / "reference" / "tonghuikang-nemotron"))


def load_dgxchen_non_cipher() -> list[dict]:
    """Load dgxchen's CSV and drop cipher rows."""
    df = pd.read_csv(DGXCHEN_CSV)
    print(f"[dgxchen] Loaded {len(df)} total records")
    print(f"[dgxchen] Cipher count: {(df['type']=='cipher').sum()}")
    df = df[df["type"] != "cipher"].copy()
    print(f"[dgxchen] After dropping cipher: {len(df)} records")
    print(f"[dgxchen] Per-type counts:")
    for t, n in df["type"].value_counts().items():
        print(f"  {t}: {n}")
    records = []
    for _, row in df.iterrows():
        records.append({
            "id": str(row["id"]),
            "prompt": str(row["prompt"]),
            "answer": str(row["answer"]),
            "type": str(row["type"]),
            "generated_cot": str(row["generated_cot"]),
            "record_kind": "reasoner",
        })
    return records


def load_our_cipher_v2() -> list[dict]:
    """Load our cipher v2 records from training-data.jsonl, map to dgxchen schema."""
    records = []
    with open(OUR_SOLVER_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if d.get("category") != "cipher":
                continue
            records.append({
                "id": d["problem_id"],
                "prompt": d["prompt"],
                "answer": d["answer"],
                "type": "cipher",
                "generated_cot": d["reasoning"],
                "record_kind": "reasoner",
            })
    print(f"[our cipher v2] Loaded {len(records)} records")
    return records


def load_tong_augmenters(matching_limit: int | None = None) -> list[dict]:
    """Generate Tong's 5 augmenter categories via his Python scripts.

    matching_limit: if set, downsample the 'matching' augmenter to this many records.
        Reproducibly seeded for determinism. Use to keep the corpus within Kaggle's
        12-hr kernel-runtime cap. Default None = use all 4,515 matching records.
    """
    from augmenters import concatenation, lstrip, matching, spelling, splitting
    import random as _r

    records = []
    for name, module in [
        ("spelling", spelling),
        ("lstrip", lstrip),
        ("concatenation", concatenation),
        ("splitting", splitting),
        ("matching", matching),
    ]:
        gen = module.generate()
        if name == "matching" and matching_limit is not None and len(gen) > matching_limit:
            rng = _r.Random(42)
            gen = rng.sample(gen, matching_limit)
            print(f"[Tong augmenter] matching downsampled to {matching_limit}")
        for p in gen:
            # All augmenters yield {id, prompt, completion or answer, category}
            answer = p.get("completion") or p.get("answer")
            if answer is None:
                raise RuntimeError(f"Augmenter {name} record {p.get('id')} has no completion/answer")
            records.append({
                "id": str(p["id"]),
                "prompt": str(p["prompt"]),
                "answer": str(answer),
                "type": str(p.get("category", name)),
                "generated_cot": "",
                "record_kind": "augmenter",
            })
        print(f"[Tong augmenter] {name}: {len(gen)} records")
    print(f"[Tong augmenter] Total: {len(records)}")
    return records


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching-limit", type=int, default=None,
                    help="Downsample matching augmenter to N records (default: all 4515)")
    args = ap.parse_args()

    print(f"=== Building v14-C training corpus (matching_limit={args.matching_limit}) ===\n")

    dgxchen = load_dgxchen_non_cipher()
    print()
    cipher_v2 = load_our_cipher_v2()
    print()
    augmenters = load_tong_augmenters(matching_limit=args.matching_limit)
    print()

    all_records = dgxchen + cipher_v2 + augmenters
    print(f"\nTotal combined records: {len(all_records)}")
    print(f"  reasoner records: {sum(1 for r in all_records if r['record_kind']=='reasoner')}")
    print(f"  augmenter records: {sum(1 for r in all_records if r['record_kind']=='augmenter')}")

    # Per-type breakdown
    from collections import Counter
    type_counts = Counter(r["type"] for r in all_records)
    print(f"\nPer-type counts:")
    for t, n in sorted(type_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {t}: {n}")

    # Sanity: dedupe by id-within-type (paranoid check)
    seen = set()
    dup = 0
    for r in all_records:
        key = (r["type"], r["id"])
        if key in seen:
            dup += 1
        seen.add(key)
    if dup:
        print(f"\nWARNING: {dup} duplicate (type,id) pairs detected")

    # Write
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    size_mb = OUTPUT_JSONL.stat().st_size / 1e6
    print(f"\nWrote {OUTPUT_JSONL} ({size_mb:.1f} MB, {len(all_records)} records)")

    # Quick text-length stats per record kind (to spot anomalies before training)
    import statistics
    for kind in ["reasoner", "augmenter"]:
        recs = [r for r in all_records if r["record_kind"] == kind]
        if not recs:
            continue
        lens = [
            len(r["prompt"]) + len(r["generated_cot"]) + len(r["answer"])
            for r in recs
        ]
        print(f"\n{kind} char-length distribution (n={len(recs)}):")
        print(f"  median: {statistics.median(lens):,.0f}")
        print(f"  p90:    {sorted(lens)[int(0.9*len(lens))]:,}")
        print(f"  p99:    {sorted(lens)[int(0.99*len(lens))]:,}")
        print(f"  max:    {max(lens):,}")


if __name__ == "__main__":
    main()
