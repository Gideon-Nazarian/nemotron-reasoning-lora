#!/usr/bin/env python3
"""
build_tong_corpus.py — reconstruct Tong Huikang's EXACT winning training corpus
in OUR pipeline's record format.

Tong (Progress Prize winner) ships his corpus as an INDEX (corpus.jsonl, 17,963
rows of metadata) plus the raw TEXT split across two directories:

  1. reasoning/<id>.txt   (9,500 files)  — multi-step CoT, answer for the prompt
                                            looked up from train.csv (id,prompt,answer).
  2. augmentations/<id>.txt (8,463 files) — [category]/[prompt]/[completion] sections;
                                            the completion IS the structural answer,
                                            so there is NO \\boxed{} wrapper.

Total = 9,500 + 8,463 = 17,963 records (matches corpus.jsonl exactly).

This script rebuilds the per-record TEXT exactly the way Tong's own corpus.py does
(verified by reading reference/tonghuikang-nemotron/corpus.py), with ONE deliberate
adaptation to our format: we DO NOT append the `<|im_end|>` token to the assistant
content. In our SFTTrainer pipeline that EOS token is added by the chat template at
train time (every record in data/v18-corpus.jsonl ends WITHOUT `<|im_end|>` in the
content). So:

  REASONING record (Tong corpus.py lines 180-187, sans <|im_end|>):
    reasoning_text  = read(reasoning/<id>.txt).rstrip("\\n")
    boxed           = re.findall(r"\\boxed\\{([^}]*)\\}", reasoning_text)
    reasoning_answer= boxed[-1] if boxed else train.csv answer  # last inline boxed wins
    user      = train.csv prompt + PROMPT_SUFFIX
    assistant = f"{reasoning_text}\\n</think>\\n\\boxed{{{reasoning_answer}}}"
    type      = category (problems.jsonl by id; identical to corpus.jsonl)

  AUGMENTATION record (Tong corpus.py lines 234-240, sans <|im_end|>):
    category   = text between "[category]\\n" and "\\n[prompt]\\n"
    prompt     = text between "[prompt]\\n"    and "\\n[completion]\\n"
    completion = text after   "\\n[completion]\\n", rstrip("\\n")
    user      = prompt + "" (EMPTY suffix — NOT PROMPT_SUFFIX)
    assistant = f"{completion}\\n</think>"     # NO \\boxed wrapper
    type      = category

PROMPT_SUFFIX is byte-identical to Tong's corpus.py AND our build_v18_corpus.py:
    "\\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

Output: data/tong-corpus.jsonl  (one {messages:[user,assistant], type} per line).
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "reference", "tonghuikang-nemotron")
TRAIN_CSV = os.path.join(REF, "train.csv")
PROBLEMS_INDEX = os.path.join(REF, "problems.jsonl")
REASONING_DIR = os.path.join(REF, "reasoning")
AUGMENTATIONS_DIR = os.path.join(REF, "augmentations")
OUT = os.path.join(ROOT, "data", "tong-corpus.jsonl")

# Byte-identical to reference/tonghuikang-nemotron/corpus.py AND scripts/build_v18_corpus.py.
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def reasoning_record(prompt: str, csv_answer: str, reasoning_text: str, category: str) -> dict:
    """Tong corpus.py lines 180-187, omitting the <|im_end|> token (our format adds EOS
    via the chat template at train time)."""
    reasoning_text = reasoning_text.rstrip("\n")
    boxed = BOXED_RE.findall(reasoning_text)
    # last inline \boxed{} wins; fall back to train.csv answer if none present
    reasoning_answer = boxed[-1] if boxed else csv_answer
    user_content = prompt + PROMPT_SUFFIX
    assistant_content = f"{reasoning_text}\n</think>\n\\boxed{{{reasoning_answer}}}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "type": str(category),
    }


def augmentation_record(text: str) -> dict:
    """Tong corpus.py lines 234-240, omitting the <|im_end|> token. NO \\boxed wrapper;
    EMPTY prompt suffix; the completion is the structural answer."""
    category = text.split("[category]\n", 1)[1].split("\n[prompt]\n", 1)[0]
    prompt_text = text.split("[prompt]\n", 1)[1].split("\n[completion]\n", 1)[0]
    completion = text.split("\n[completion]\n", 1)[1].rstrip("\n")
    user_content = prompt_text  # empty suffix
    assistant_content = f"{completion}\n</think>"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "type": str(category),
    }


def main() -> None:
    # --- load lookups ---
    prompts: dict[str, str] = {}
    answers: dict[str, str] = {}
    with open(TRAIN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            prompts[row["id"]] = row["prompt"]
            answers[row["id"]] = row["answer"]

    problem_cats: dict[str, str] = {p["id"]: p["category"] for p in load_jsonl(PROBLEMS_INDEX)}

    records: list[dict] = []
    counts = {"reasoning": 0, "augmentation": 0}
    skipped: list[str] = []

    # --- 1) reasoning records (9,500) ---
    # Mirror Tong's selection: a reasoning .txt that also has prompt + category.
    reasoning_ids = sorted(
        pid
        for pid in problem_cats
        if os.path.exists(os.path.join(REASONING_DIR, f"{pid}.txt")) and pid in prompts
    )
    for pid in reasoning_ids:
        with open(os.path.join(REASONING_DIR, f"{pid}.txt")) as f:
            reasoning_text = f.read()
        rec = reasoning_record(prompts[pid], answers[pid], reasoning_text, problem_cats[pid])
        records.append(rec)
        counts["reasoning"] += 1

    # Any reasoning file lacking a prompt/category is an integrity problem — surface it.
    all_reasoning_files = {
        fn[:-4] for fn in os.listdir(REASONING_DIR) if fn.endswith(".txt")
    }
    missing_meta = sorted(all_reasoning_files - set(reasoning_ids))
    if missing_meta:
        skipped.extend(f"reasoning/{m}.txt (no prompt/category)" for m in missing_meta)

    # --- 2) augmentation records (8,463) ---
    aug_files = sorted(fn for fn in os.listdir(AUGMENTATIONS_DIR) if fn.endswith(".txt"))
    for fn in aug_files:
        with open(os.path.join(AUGMENTATIONS_DIR, fn)) as f:
            text = f.read()
        rec = augmentation_record(text)
        records.append(rec)
        counts["augmentation"] += 1

    # --- write ---
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # --- report ---
    print("=== tong corpus built ===")
    print(f"  reasoning records   : {counts['reasoning']}")
    print(f"  augmentation records: {counts['augmentation']}")
    print(f"  TOTAL records       : {len(records)}")
    print(f"  written -> {OUT}")
    if skipped:
        print(f"\n  WARNING — {len(skipped)} reasoning files skipped (missing meta):")
        for s in skipped[:20]:
            print(f"    {s}")

    # --- in-script self-checks ---
    import collections

    bytype = collections.Counter(r["type"] for r in records)
    print("\n=== by type ===")
    for k, v in sorted(bytype.items()):
        print(f"  {k}: {v}")

    n_reason = counts["reasoning"]
    reasoning_recs = records[:n_reason]
    aug_recs = records[n_reason:]

    reason_boxed_ok = sum(
        1
        for r in reasoning_recs
        if r["messages"][1]["content"].rstrip().endswith("}")
        and "\\boxed{" in r["messages"][1]["content"].rsplit("</think>", 1)[-1]
    )
    aug_has_boxed = sum(
        1 for r in aug_recs if "\\boxed" in r["messages"][1]["content"]
    )
    aug_think_ok = sum(
        1 for r in aug_recs if r["messages"][1]["content"].rstrip().endswith("</think>")
    )
    print("\n=== format self-check ===")
    print(f"  reasoning ending in \\boxed{{...}} after </think>: {reason_boxed_ok}/{len(reasoning_recs)}")
    print(f"  augmentation records containing \\boxed (MUST be 0): {aug_has_boxed}/{len(aug_recs)}")
    print(f"  augmentation records ending in </think>: {aug_think_ok}/{len(aug_recs)}")

    ok = (
        len(records) == 17963
        and reason_boxed_ok == len(reasoning_recs)
        and aug_has_boxed == 0
        and aug_think_ok == len(aug_recs)
        and not skipped
    )
    print("\n  ALL CHECKS PASSED" if ok else "\n  *** CHECKS FAILED — inspect above ***")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
