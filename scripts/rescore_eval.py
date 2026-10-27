"""Re-score an eval-results CSV with multiple extraction strategies.

The original eval harness uses a single strict extraction (e.g. \\boxed{...} for v3,
content-after-</think> for v4). This script applies multiple alternative strategies
to the saved completions and reports which strategies work for which problems.

Use this to estimate "what's the actual model capability if we had perfect
extraction?" without paying GPU to re-run inference.

Usage:
    uv run python scripts/rescore_eval.py notes/eval-runs/2026-05-10-v4-results.csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# ----------------------------- extraction strategies -----------------------------


def strat_v3_boxed(text: str) -> str:
    """Original v3 strategy: contents of last \\boxed{...}."""
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    return m.group(1).strip() if m else ""


def strat_v4_after_think(text: str) -> str:
    """Original v4 strategy: content between </think> and <|im_end|>."""
    if "</think>" not in text:
        return ""
    post = text.split("</think>", 1)[1]
    if "<|im_end|>" in post:
        post = post.split("<|im_end|>", 1)[0]
    return post.strip()


def strat_explicit_answer(text: str) -> str:
    """Look for 'answer is X' / 'final answer: X' / 'answer X' phrases."""
    patterns = [
        r"final answer\s*(?:is|:)\s*([^\n.]{1,150})",
        r"answer\s*(?:is|:)\s*([^\n.]{1,150})",
        r"so the answer is\s+([^\n.]{1,150})",
        r"thus\s+(?:the\s+)?answer\s*(?:is|:)\s*([^\n.]{1,150})",
    ]
    last = ""
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            last = m.group(1).strip().rstrip(".\"' ")
    return last


def strat_last_numeric(text: str) -> str:
    """Last numeric value in the text. Useful for math problems."""
    matches = list(re.finditer(r"-?\d+\.\d+|-?\d+", text))
    return matches[-1].group(0) if matches else ""


def strat_last_binary8(text: str) -> str:
    """Last 8-bit binary string. Useful for bit-manipulation problems."""
    matches = list(re.finditer(r"\b[01]{8}\b", text))
    return matches[-1].group(0) if matches else ""


def strat_last_roman(text: str) -> str:
    """Last Roman-numeral-looking token. Useful for Roman-numeral problems."""
    # Look for >= 2 chars in the Roman numeral character set, on word boundary
    matches = list(re.finditer(r"\b[IVXLCDM]{2,}\b", text))
    return matches[-1].group(0) if matches else ""


def strat_so_about(text: str) -> str:
    """'So about X' / 'approximately X' / '= X' patterns near end."""
    patterns = [
        r"so\s+(?:about|approximately)?\s*([\d.]+)\s*(?:m|ft|s|kg|cm|km)?\b",
        r"=\s*([\d.]+)\s*(?:m|ft|s|kg|cm|km)?\s*\.?\s*$",
        r"approx(?:imately)?\s+([\d.]+)",
    ]
    last = ""
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            last = m.group(1).strip()
    return last


def strat_gold_substring(text: str, gold: str) -> str:
    """Most permissive: does the gold answer appear anywhere in the completion?
    Returns gold if found (so scoring matches), else empty string."""
    g = gold.strip()
    if g and g in text:
        return g
    # Case-insensitive fallback for non-numeric
    if g and not re.fullmatch(r"-?\d+\.?\d*", g) and g.lower() in text.lower():
        return g
    return ""


def strat_gold_format_aware(text: str, gold: str) -> str:
    """Use gold format to pick the right extractor. APPROXIMATES grader logic."""
    g = gold.strip()
    if re.fullmatch(r"[01]{8}", g):
        return strat_last_binary8(text)
    if re.fullmatch(r"[IVXLCDM]+", g):
        return strat_last_roman(text)
    if re.fullmatch(r"-?\d+\.?\d*", g):
        return strat_last_numeric(text)
    # Multi-word phrase — try explicit answer markers first
    if " " in g:
        return strat_explicit_answer(text)
    return strat_explicit_answer(text)


STRATEGIES = {
    "v3_boxed": strat_v3_boxed,
    "v4_after_think": strat_v4_after_think,
    "explicit_answer": strat_explicit_answer,
    "last_numeric": strat_last_numeric,
    "so_about": strat_so_about,
    "gold_format_aware": "needs_gold",
    "gold_substring": "needs_gold",  # loosest: did the gold appear anywhere in the completion?
}


# ----------------------------- scoring -----------------------------


def score(extracted: str, gold: str) -> bool:
    """Strict-ish equality, tolerant of whitespace and case for non-numeric."""
    if not extracted:
        return False
    e, g = extracted.strip(), gold.strip()
    if e == g:
        return True
    # Try case-insensitive for non-numeric
    if not re.fullmatch(r"-?\d+\.?\d*", g) and e.lower() == g.lower():
        return True
    return False


def main(csv_path: Path) -> int:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}\n")

    # Per-problem results: which strategies hit
    rows = []
    for _, r in df.iterrows():
        completion = str(r.get("completion", ""))
        gold = str(r["gold"])
        result = {"id": r["id"], "gold": gold, "completion_len": len(completion)}
        for name, fn in STRATEGIES.items():
            if fn == "needs_gold":
                # Dispatch the right gold-aware strategy by name
                if name == "gold_substring":
                    ext = strat_gold_substring(completion, gold)
                else:
                    ext = strat_gold_format_aware(completion, gold)
            else:
                ext = fn(completion)
            result[f"{name}_extracted"] = ext
            result[f"{name}_correct"] = score(ext, gold)
        rows.append(result)

    out = pd.DataFrame(rows)

    # Per-strategy summary
    print("=== Per-strategy accuracy ===")
    for name in STRATEGIES:
        n_correct = int(out[f"{name}_correct"].sum())
        print(f"  {name:25s}: {n_correct}/{len(out)} = {n_correct/len(out):.3f}")

    # "Any strategy hits" — upper bound on what better extraction could give
    any_correct_cols = [f"{name}_correct" for name in STRATEGIES]
    any_correct = out[any_correct_cols].any(axis=1)
    print(f"  {'ANY strategy':25s}: {int(any_correct.sum())}/{len(out)} = {any_correct.sum()/len(out):.3f}")

    # Per-row detail
    print("\n=== Per-row detail ===")
    for _, r in out.iterrows():
        print(f"\n--- {r['id']} (gold={r['gold']!r}, completion_len={r['completion_len']}) ---")
        for name in STRATEGIES:
            ext = r[f"{name}_extracted"]
            mark = "✓" if r[f"{name}_correct"] else " "
            ext_display = (ext[:80] + "...") if len(str(ext)) > 80 else ext
            print(f"  [{mark}] {name:25s} -> {ext_display!r}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/rescore_eval.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1])))
