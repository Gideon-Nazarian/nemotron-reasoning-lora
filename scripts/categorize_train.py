"""Categorize each row of train.csv by problem type.

We've observed at least four distinct problem types in the competition:
  - bit_manipulation: 8-bit binary in -> 8-bit binary out
  - cipher: substitution cipher decryption (cipher_phrase output)
  - numeric: scaling / unit conversion / gravity problems (decimal output)
  - roman_numeral: write a number in a Roman-numeral-like system

This script classifies each problem on TWO axes:
  1. answer_format: pattern of the gold answer string
  2. prompt_keyword: keyword found in the prompt

Cross-tabulating these tells us how clean the categorization is, and
whether prompt-based classification matches answer-based classification.

Output: prints distribution, saves to notes/train-categories.csv for
downstream use (e.g., stratified sampling for fine-tuning).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
TRAIN_CSV = ROOT / "data" / "train.csv"
OUT_CSV = ROOT / "notes" / "train-categories.csv"


def classify_by_answer(answer: str) -> str:
    a = str(answer).strip()
    if re.fullmatch(r"[01]{8}", a):
        return "bit_manipulation"
    if re.fullmatch(r"[IVXLCDM]+", a):
        return "roman_numeral"
    if re.fullmatch(r"-?\d+(\.\d+)?", a):
        # Numeric — split into integer vs decimal
        return "numeric_int" if "." not in a else "numeric_decimal"
    if re.fullmatch(r"[a-z]+( [a-z]+)*", a):
        return "cipher_phrase"
    if re.fullmatch(r"[A-Z]{2,}", a):
        return "uppercase_word"  # e.g. potential abbreviations
    return "other"


def classify_by_prompt(prompt: str) -> str:
    p = prompt.lower()
    # Order matters — most specific first
    if "bit manipulation" in p or "bit shift" in p or "8-bit binary" in p:
        return "bit_manipulation"
    if "encryption rule" in p or "cipher" in p or "decrypt" in p:
        return "cipher"
    if "numeral system" in p:
        return "numeral_system"
    if "gravitational constant" in p:
        return "gravity"
    if "secret unit" in p or "different unit" in p or "secretly converted" in p:
        return "unit_conversion"
    if "secret set of transformation rules" in p or "transformation rules is applied" in p:
        return "transformation_rules"
    if "secret addition" in p or "secret arithmetic" in p:
        return "secret_arithmetic"
    return "other_prompt"


def main() -> int:
    if not TRAIN_CSV.exists():
        print(f"train.csv not found at {TRAIN_CSV}", file=sys.stderr)
        return 1

    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} rows from {TRAIN_CSV}")

    df["category_by_answer"] = df["answer"].apply(classify_by_answer)
    df["category_by_prompt"] = df["prompt"].apply(classify_by_prompt)
    df["prompt_len_chars"] = df["prompt"].str.len()
    df["answer_len_chars"] = df["answer"].astype(str).str.len()

    # ---- Distributions ----
    print("\n=== Distribution by ANSWER format ===")
    counts = df["category_by_answer"].value_counts()
    for cat, n in counts.items():
        print(f"  {cat:25s}: {n:5d}  ({n/len(df)*100:5.1f}%)")

    print("\n=== Distribution by PROMPT keyword ===")
    counts = df["category_by_prompt"].value_counts()
    for cat, n in counts.items():
        print(f"  {cat:25s}: {n:5d}  ({n/len(df)*100:5.1f}%)")

    # ---- Cross-tab ----
    print("\n=== Cross-tab: prompt × answer category ===")
    ct = pd.crosstab(df["category_by_prompt"], df["category_by_answer"])
    print(ct.to_string())

    # ---- Inspect the 'other' / mismatches ----
    print("\n=== Sample: 'other' answer formats (5 examples) ===")
    other = df[df["category_by_answer"] == "other"]
    print(f"Total in 'other': {len(other)}")
    for _, r in other.head(5).iterrows():
        a = str(r["answer"])
        p = str(r["prompt"])[:200]
        print(f"  id={r['id']}  answer={a!r}  prompt[:200]={p!r}")

    print("\n=== Sample: 'other_prompt' (5 examples) ===")
    other_p = df[df["category_by_prompt"] == "other_prompt"]
    print(f"Total: {len(other_p)}")
    for _, r in other_p.head(5).iterrows():
        a = str(r["answer"])
        p = str(r["prompt"])[:300]
        print(f"  id={r['id']}  answer={a!r}  prompt[:300]={p!r}")
        print()

    # ---- Length stats per category ----
    print("\n=== Prompt-length stats per answer category ===")
    grouped = df.groupby("category_by_answer")["prompt_len_chars"].agg(["min", "median", "mean", "max", "count"])
    print(grouped.round(0).to_string())

    # ---- Save ----
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[["id", "category_by_answer", "category_by_prompt", "prompt_len_chars", "answer_len_chars"]].to_csv(OUT_CSV, index=False)
    print(f"\nWrote categorization to {OUT_CSV}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
