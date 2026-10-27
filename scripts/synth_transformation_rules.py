"""Synthetic data generator for the `transformation_rules` category.

Why: train.csv has 1555 transformation_rules problems, and our programmatic
solvers covered ZERO of them (the rules are bespoke per-problem, mixing
arithmetic obfuscation and pure-symbol manipulation). The model has never
seen this category in training. This generator produces synthetic examples
that teach the META-SKILL of "look at examples, hypothesize a rule, verify
against all examples, apply to question".

We don't attempt to match the test set's exact rules (they're unknowable
without solving them). Instead we produce verifiable patterns of the same
SHAPE — pairs of inputs and outputs under a hidden rule — and write
hypothesis-testing reasoning text.

Distribution mirrors the real train.csv split (~50/50 numeric vs symbolic).

Output: data/synth-transformation-rules.jsonl
Format: same as data/training-data.jsonl (problem_id, category, prompt,
reasoning, answer, gold).
"""

from __future__ import annotations

import json
import random
import string
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).parent.parent
OUT_PATH = ROOT / "data" / "synth-transformation-rules.jsonl"

# Operator symbols (avoid clashes with arithmetic that would confuse the
# generator — we want the symbol to look like an obscure operator, not be one).
OP_SYMBOLS = ["$", "@", "~", "|", "\\", "^", "&", "?", ":", ";", "<", ">", "+", "-", "*", "/", "%", "#", "!"]

# Symbol alphabet for pure-symbolic problems
SYMBOL_ALPHABET = list("`!@#$%^&*()_-+={}[]|\\:;\"'<>?,./~")


# ============================================================
# Numeric rules — operand-int -> operand-int -> result-int
# Each takes (a, b) where a, b are ints; returns int or None if invalid.
# ============================================================

def r_add(a, b): return a + b
def r_sub(a, b): return a - b if a >= b else None  # avoid negatives
def r_mul(a, b): return a * b
def r_div_floor(a, b): return a // b if b > 0 else None
def r_mod(a, b): return a % b if b > 0 else None
def r_concat(a, b): return int(f"{a}{b}")
def r_concat_swap(a, b): return int(f"{b}{a}")
def r_digit_sum(a, b): return sum(int(c) for c in f"{a}{b}")
def r_digit_diff(a, b):
    sa = sum(int(c) for c in f"{a}")
    sb = sum(int(c) for c in f"{b}")
    return abs(sa - sb)
def r_max(a, b): return max(a, b)
def r_min(a, b): return min(a, b)
def r_avg_floor(a, b): return (a + b) // 2
def r_xor_digit(a, b):
    """Digit-wise XOR mod 10 of paired digits (zero-padded to same length)."""
    sa, sb = f"{a}", f"{b}"
    n = max(len(sa), len(sb))
    sa, sb = sa.zfill(n), sb.zfill(n)
    return int("".join(str((int(x) ^ int(y)) % 10) for x, y in zip(sa, sb)))


NUMERIC_RULES: dict[str, tuple[Callable, str, str]] = {
    # name: (fn, human description, reasoning template hint)
    "add":         (r_add,         "addition",                       "a + b"),
    "subtract":    (r_sub,         "subtraction (larger minus smaller)", "a - b"),
    "multiply":    (r_mul,         "multiplication",                 "a × b"),
    "div_floor":   (r_div_floor,   "integer division (a ÷ b, floored)", "a // b"),
    "mod":         (r_mod,         "modulo (remainder a mod b)",     "a % b"),
    "concat":      (r_concat,      "concatenation (a then b as a single number)", "concat(a, b)"),
    "concat_swap": (r_concat_swap, "swap-concatenation (b then a)",  "concat(b, a)"),
    "digit_sum":   (r_digit_sum,   "sum of all digits in both numbers", "sum of all digits"),
    "digit_diff":  (r_digit_diff,  "absolute difference of digit sums", "|sum(digits_a) - sum(digits_b)|"),
    "max":         (r_max,         "maximum of the two numbers",     "max(a, b)"),
    "min":         (r_min,         "minimum of the two numbers",     "min(a, b)"),
    "avg_floor":   (r_avg_floor,   "average (floored)",              "(a + b) // 2"),
    "xor_digit":   (r_xor_digit,   "digit-wise XOR mod 10",          "digit-XOR"),
}


# ============================================================
# Symbolic rules — string -> string
# Operate on short symbol strings (5 chars).
# ============================================================

def s_reverse(s): return s[::-1]
def s_first_n(s, n): return s[:n]
def s_last_n(s, n): return s[-n:]
def s_drop_at(s, pos): return s[:pos] + s[pos+1:]
def s_keep_positions(s, positions): return "".join(s[p] for p in positions if p < len(s))
def s_swap_pos(s, i, j):
    if i >= len(s) or j >= len(s): return None
    lst = list(s); lst[i], lst[j] = lst[j], lst[i]; return "".join(lst)
def s_double_at(s, pos):
    if pos >= len(s): return None
    return s[:pos] + s[pos]*2 + s[pos+1:]
def s_every_other(s, start=0): return s[start::2]
def s_take_unique(s):
    seen = set(); out = []
    for c in s:
        if c not in seen:
            seen.add(c); out.append(c)
    return "".join(out)


SYMBOLIC_RULES: dict[str, tuple[Callable, str]] = {
    "reverse":         ((lambda s: s_reverse(s)),               "reverse the string"),
    "first_2":         ((lambda s: s_first_n(s, 2)),            "take the first 2 characters"),
    "first_3":         ((lambda s: s_first_n(s, 3)),            "take the first 3 characters"),
    "last_2":          ((lambda s: s_last_n(s, 2)),             "take the last 2 characters"),
    "last_3":          ((lambda s: s_last_n(s, 3)),             "take the last 3 characters"),
    "drop_first":      ((lambda s: s_drop_at(s, 0)),            "drop the first character"),
    "drop_last":       ((lambda s: s_drop_at(s, len(s)-1)),     "drop the last character"),
    "drop_middle":     ((lambda s: s_drop_at(s, len(s)//2)),    "drop the middle character"),
    "keep_0_2_4":      ((lambda s: s_keep_positions(s, [0,2,4])), "keep characters at positions 0, 2, 4"),
    "keep_1_3":        ((lambda s: s_keep_positions(s, [1,3])),  "keep characters at positions 1 and 3"),
    "swap_0_last":     ((lambda s: s_swap_pos(s, 0, len(s)-1)), "swap first and last characters"),
    "double_first":    ((lambda s: s_double_at(s, 0)),          "double the first character"),
    "every_other":     ((lambda s: s_every_other(s, 0)),        "take every other character starting from position 0"),
    "every_other_1":   ((lambda s: s_every_other(s, 1)),        "take every other character starting from position 1"),
    "take_unique":     ((lambda s: s_take_unique(s)),           "take unique characters (preserve order, drop duplicates)"),
}


# ============================================================
# Generators
# ============================================================

PROMPT_HEAD = (
    "In Alice's Wonderland, a secret set of transformation rules is applied "
    "to equations. Below are a few examples:\n"
)
QUESTION_HEAD = "Now, determine the result for: "


def gen_numeric_problem(rng: random.Random) -> dict | None:
    """Generate one numeric arithmetic-style transformation_rules problem."""
    rule_name = rng.choice(list(NUMERIC_RULES.keys()))
    rule_fn, desc, hint = NUMERIC_RULES[rule_name]
    op = rng.choice(OP_SYMBOLS)

    # Generate examples with valid inputs
    examples = []
    attempts = 0
    while len(examples) < 5 and attempts < 50:
        attempts += 1
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        if rule_name == "subtract":
            a, b = max(a, b), min(a, b)
            if a == b: continue
        r = rule_fn(a, b)
        if r is None or r < 0: continue
        examples.append((a, b, r))
    if len(examples) < 4: return None

    # Generate question
    attempts = 0
    while attempts < 50:
        attempts += 1
        qa = rng.randint(10, 99)
        qb = rng.randint(10, 99)
        if rule_name == "subtract":
            qa, qb = max(qa, qb), min(qa, qb)
            if qa == qb: continue
        q_result = rule_fn(qa, qb)
        if q_result is None or q_result < 0: continue
        # Avoid trivial overlap with examples
        if not any(qa == a and qb == b for a, b, _ in examples):
            break
    else:
        return None

    # Build the prompt
    example_lines = "\n".join(f"{a}{op}{b} = {r}" for a, b, r in examples)
    prompt = f"{PROMPT_HEAD}{example_lines}\n{QUESTION_HEAD}{qa}{op}{qb}"

    # Build hypothesis-testing reasoning
    reasoning = build_numeric_reasoning(rule_name, examples, qa, qb, q_result, op)

    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": str(q_result),
        "rule": rule_name,
    }


def gen_symbolic_problem(rng: random.Random) -> dict | None:
    """Generate one pure-symbol transformation_rules problem."""
    rule_name = rng.choice(list(SYMBOLIC_RULES.keys()))
    rule_fn, desc = SYMBOLIC_RULES[rule_name]
    op = rng.choice(OP_SYMBOLS)
    # Make sure operator isn't in our symbol alphabet (so it stays distinct)
    alphabet = [c for c in SYMBOL_ALPHABET if c != op]

    # Generate 5 example pairs
    examples = []
    seen_inputs = set()
    attempts = 0
    while len(examples) < 5 and attempts < 50:
        attempts += 1
        inp = "".join(rng.choices(alphabet, k=5))
        if inp in seen_inputs: continue
        seen_inputs.add(inp)
        out = rule_fn(inp)
        if out is None or out == "": continue
        examples.append((inp, out))
    if len(examples) < 4: return None

    # Generate question
    attempts = 0
    while attempts < 50:
        attempts += 1
        q_in = "".join(rng.choices(alphabet, k=5))
        if q_in in seen_inputs: continue
        q_out = rule_fn(q_in)
        if q_out is None or q_out == "": continue
        break
    else:
        return None

    # For symbolic problems we typically have a single string per side, no
    # operator-between-operands. But to match the real format, we use:
    # `<input>` (just the input string, since the "rule" is unary)
    # Actually real problems have "A<op>B = C" structure even when symbolic.
    # Let's split input into two halves with an operator between.
    # 5 chars -> 2 chars + op + 3 chars? Or we could just write input = output.
    # Real examples have things like " `!*[{ = '"[ " — one symbol string -> another.
    # We'll match that: write input = output.

    example_lines = "\n".join(f"{i} = {o}" for i, o in examples)
    prompt = f"{PROMPT_HEAD}{example_lines}\n{QUESTION_HEAD}{q_in}"

    reasoning = build_symbolic_reasoning(rule_name, examples, q_in, q_out)

    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": q_out,
        "rule": rule_name,
    }


# ============================================================
# Reasoning templates — hypothesis-test-apply pattern
# ============================================================

def build_numeric_reasoning(rule_name, examples, qa, qb, q_result, op):
    rule_fn, desc, hint = NUMERIC_RULES[rule_name]
    a0, b0, r0 = examples[0]

    lines = []
    lines.append(
        f"Looking at the first example, {a0}{op}{b0} produces {r0}. "
        f"I need to figure out what operation '{op}' represents."
    )

    # Show 1-2 wrong hypotheses being tested and rejected (teaches search behavior)
    rejected_hypotheses = []
    for try_name in ["add", "multiply", "concat"]:
        if try_name == rule_name: continue
        try_fn = NUMERIC_RULES[try_name][0]
        try_result = try_fn(a0, b0)
        if try_result is not None and try_result != r0:
            try_desc = NUMERIC_RULES[try_name][1]
            rejected_hypotheses.append((try_desc, try_result, r0))
            if len(rejected_hypotheses) >= 1: break

    for try_desc, try_result, actual in rejected_hypotheses:
        lines.append(f"Testing if it's {try_desc}: {a0}, {b0} → {try_result}, but the example shows {actual}. So '{op}' is not {try_desc}.")

    # Verify our actual rule across all examples
    lines.append(f"Testing if it's {desc}:")
    for a, b, r in examples:
        verify = rule_fn(a, b)
        lines.append(f"  {a}, {b}: {hint} = {verify}. Expected {r}. {'✓' if verify == r else '✗'}")
    lines.append(f"All {len(examples)} examples match {desc}. The rule is confirmed.")

    # Apply to question
    lines.append(f"Applying to the question {qa}{op}{qb}: {hint} = {q_result}.")

    return "\n".join(lines)


def build_symbolic_reasoning(rule_name, examples, q_in, q_out):
    rule_fn, desc = SYMBOLIC_RULES[rule_name]
    in0, out0 = examples[0]

    lines = []
    lines.append(
        f"Looking at the first example, '{in0}' → '{out0}'. The input has "
        f"{len(in0)} characters; the output has {len(out0)}. "
        f"Let me hypothesize a transformation rule."
    )

    # Try one rejected alternative
    rejected_tried = False
    for try_name in ["reverse", "first_2", "last_2"]:
        if try_name == rule_name: continue
        try_fn = SYMBOLIC_RULES[try_name][0]
        try_desc = SYMBOLIC_RULES[try_name][1]
        try_result = try_fn(in0)
        if try_result is not None and try_result != out0:
            lines.append(f"Hypothesis '{try_desc}': '{in0}' → '{try_result}', but the example output is '{out0}'. Rejected.")
            rejected_tried = True
            break

    # Verify the actual rule
    lines.append(f"Hypothesis '{desc}':")
    for inp, out in examples:
        check = rule_fn(inp)
        lines.append(f"  '{inp}' → '{check}'. Expected '{out}'. {'✓' if check == out else '✗'}")
    lines.append(f"All {len(examples)} examples match. The rule is to {desc}.")

    # Apply
    lines.append(f"Applying to '{q_in}': the result is '{q_out}'.")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main() -> int:
    n_numeric = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    n_symbolic = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    seed = 42

    rng = random.Random(seed)
    records = []
    rule_counts = {}

    # Generate numeric
    attempts = 0
    while sum(1 for r in records if r["rule"] in NUMERIC_RULES) < n_numeric and attempts < n_numeric * 3:
        attempts += 1
        rec = gen_numeric_problem(rng)
        if rec is None: continue
        # Self-verify: the answer must be derivable from the rule. Already verified inside generator.
        records.append(rec)
        rule_counts[rec["rule"]] = rule_counts.get(rec["rule"], 0) + 1

    # Generate symbolic
    attempts = 0
    while sum(1 for r in records if r["rule"] in SYMBOLIC_RULES) < n_symbolic and attempts < n_symbolic * 3:
        attempts += 1
        rec = gen_symbolic_problem(rng)
        if rec is None: continue
        records.append(rec)
        rule_counts[rec["rule"]] = rule_counts.get(rec["rule"], 0) + 1

    # Wrap into our standard SFT record format and write
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for i, rec in enumerate(records):
            out = {
                "problem_id": f"synth-tr-{i:04d}",
                "category": "transformation_rules",
                "prompt": rec["prompt"],
                "reasoning": rec["reasoning"],
                "answer": rec["answer"],
                "gold": rec["answer"],
            }
            f.write(json.dumps(out) + "\n")

    print(f"Wrote {len(records)} synthetic transformation_rules records to {OUT_PATH}")
    print("\nBy rule:")
    for rule, n in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"  {rule:20s}: {n}")

    if records:
        rls = [len(r["reasoning"]) for r in records]
        print(f"\nReasoning length (chars): min={min(rls)}, median={sorted(rls)[len(rls)//2]}, mean={sum(rls)//len(rls)}, max={max(rls)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
