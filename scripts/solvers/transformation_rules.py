"""Solver for `transformation_rules` problems (numeric subset).

Prompt structure:
    In Alice's Wonderland, a secret set of transformation rules is applied
    to equations. Below are a few examples:
        34/44 = 1
        41/32 = 9
        34|25 = 69
        87\\64 = 8853
    Now, determine the result for: 69/52

Each example is `<digits><op><digits> = <output>`. The `<op>` is a single
non-digit symbol (`/`, `|`, `\\`, `*`, `-`, `+`, `}`, `{`, `&`, etc.). The
RULE depends on which op symbol is used — different op symbols can represent
totally different transformations. The question's op must therefore have
examples to infer from.

Algorithm (ported with simplifications from Tong Huikang's reasoners/
equation_numeric.py):
  1. Parse each example into (a, op, b, out).
  2. Group examples by op.
  3. For the question's op (q_op), search a fixed list of candidate
     transformations: concat, reverse concat, addition, subtraction (both
     directions), abs difference, multiplication, ±1 variants, integer
     division, modulo, and (for 2-digit operands) cross-multiplications,
     digit-wise mod10 add/sub, digit absolute diff, determinant, digit
     product diff/sum.
  4. Also handle four output formats: plain numeric, sign-prefixed (`-NNN`),
     sign-suffixed-via-op (`NNN<op>` indicating negative), and similar.
  5. Also try reversing the operands and/or reversing the result digits.
  6. Accept the first candidate whose rule reproduces ALL examples in the
     q_op group.

This is the SYMBOLIC subset is NOT handled (823 of 1555 transformation_rules
problems are pure-punctuation — those need a different family of rules). Of
the 732 NUMERIC problems, ~596 have q_op present in examples; we aim to
cover most of those with the candidate list above.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable

from .types import Solution

_EXAMPLE_RE = re.compile(r"^(\d+)(\D)(\d+)$")
_QUESTION_RE = re.compile(
    r"determine the result for:\s*(\S+)", re.IGNORECASE
)


def _common_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Common operations tried first (most likely to match)."""
    out: list[tuple[str, str]] = [
        ("concatenation", sa + sb),
        ("reverse concatenation", sb + sa),
        ("addition", str(a + b)),
        ("absolute difference", str(abs(a - b))),
        ("negated absolute difference", str(-abs(a - b))),
        ("subtraction (a-b)", str(a - b)),
        ("reverse subtraction (b-a)", str(b - a)),
        ("multiplication", str(a * b)),
    ]
    return out


def _rare_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Rare operations, tried if common ones don't match."""
    out: list[tuple[str, str]] = [
        ("multiply+1", str(a * b + 1)),
        ("multiply-1", str(a * b - 1)),
        ("add+1", str(a + b + 1)),
        ("add-1", str(a + b - 1)),
        ("sub+1", str(a - b + 1)),
        ("sub-1", str(a - b - 1)),
    ]
    if a != 0 and b != 0:
        big, small = max(a, b), min(a, b)
        out.append(("max mod min", str(big % small)))
    if b != 0:
        out.append(("integer division (a/b)", str(a // b)))
        out.append(("modulo (a mod b)", str(a % b)))
    if a != 0:
        out.append(("reverse division (b/a)", str(b // a)))
        out.append(("reverse modulo (b mod a)", str(b % a)))
    if len(sa) == 2 and len(sb) == 2:
        d1, d2, d3, d4 = int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1])
        out.append(("digit absolute diff", str(abs(d1 - d3)) + str(abs(d2 - d4))))
        out.append(("digit add mod10", str((d1 + d3) % 10) + str((d2 + d4) % 10)))
        out.append(("digit sub mod10", str((d1 - d3) % 10) + str((d2 - d4) % 10)))
        out.append(("cross multiply", str(d1 * d3 + d2 * d4)))
        out.append(("cross multiply rev", str(d1 * d4 + d2 * d3)))
        out.append(("digit multiply", str(d1 * d3) + str(d2 * d4)))
        out.append(("digit multiply rev", str(d1 * d4) + str(d2 * d3)))
        out.append(("digit sum diff", str((d1 + d2) - (d3 + d4))))
        out.append(("digit sum sum", str((d1 + d2) + (d3 + d4))))
        out.append(("digit product diff", str(d1 * d2 - d3 * d4)))
        out.append(("digit product sum", str(d1 * d2 + d3 * d4)))
        det_val = d1 * d4 - d2 * d3
        out.append(("determinant", str(det_val)))
        out.append(("abs determinant", str(abs(det_val))))
    return out


def _all_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    return _common_candidates(a, b, sa, sb) + _rare_candidates(a, b, sa, sb)


def _rev_str(s: str) -> str:
    """Reverse a (possibly signed) digit string."""
    if s.startswith("-"):
        return "-" + s[1:][::-1]
    return s[::-1]


def _find_op_for_group(
    group: list[tuple[str, str, str]],
    op_char: str,
) -> tuple[str, bool, bool, str] | None:
    """Search for an operation that reproduces every output in `group`.

    Returns (op_name, rev_ops, rev_res, output_fmt) or None.
      - rev_ops: did we reverse the operands before applying?
      - rev_res: did we reverse the result digits?
      - output_fmt: "plain" | "neg_suffix" | "neg_prefix"
        - "neg_suffix": negative outputs are written `NNN<op_char>` (e.g. `123-`)
        - "neg_prefix": negative outputs are written `<op_char>NNN` (e.g. `-123`
          where `-` is also the op char; or with the op char prefixed)
    """
    # Detect output-format markers based on the group
    # (Per Tong: if any output ends with the op_char and isn't just the op_char,
    # that's the negative sign marker; same idea for prefix.)
    any_suf = any(out.endswith(op_char) and len(out) > 1 for _, _, out in group)
    any_pre = any(out.startswith(op_char) and len(out) > 1 for _, _, out in group)
    # Plain `-` sign in output (when op_char isn't `-`)
    any_neg_suf = op_char != "-" and any(out.endswith("-") and len(out) > 1 for _, _, out in group)
    any_neg_pre = op_char != "-" and any(out.startswith("-") and len(out) > 1 for _, _, out in group)

    fmt_candidates: list[str] = ["plain"]
    if any_neg_suf or any_neg_pre or any_suf or any_pre:
        fmt_candidates.append("neg_suffix")
        fmt_candidates.append("neg_prefix")

    # For each format, transform group outputs and search candidates
    for fmt in fmt_candidates:
        transformed: list[tuple[str, str, str]] = []
        ok = True
        for a, b, out in group:
            t_out = out
            if fmt == "neg_suffix":
                # Try stripping op_char or '-' suffix and prefixing '-'
                if out.endswith(op_char) and len(out) > 1:
                    t_out = "-" + out[: -len(op_char)]
                elif out.endswith("-") and len(out) > 1 and op_char != "-":
                    t_out = "-" + out[:-1]
            elif fmt == "neg_prefix":
                if out.startswith(op_char) and len(out) > 1:
                    t_out = "-" + out[len(op_char):]
                elif out.startswith("-") and len(out) > 1 and op_char != "-":
                    pass  # already in -NNN form
            transformed.append((a, b, t_out))
        # Now search candidates with operand/result reversal flags
        for rev_ops in (False, True):
            for rev_res in (False, True):
                # Try every candidate and see if it matches all
                op_names_to_try: list[str] = []
                # collect unique op names from first pair's candidates
                first_a, first_b, _ = transformed[0]
                ta = first_a[::-1] if rev_ops else first_a
                tb = first_b[::-1] if rev_ops else first_b
                for name, _ in _all_candidates(int(ta), int(tb), ta, tb):
                    op_names_to_try.append(name)
                seen = set()
                op_names_to_try = [n for n in op_names_to_try if not (n in seen or seen.add(n))]

                for op_name in op_names_to_try:
                    all_match = True
                    for a, b, t_out in transformed:
                        ta_i = a[::-1] if rev_ops else a
                        tb_i = b[::-1] if rev_ops else b
                        raw = None
                        for name, res in _all_candidates(int(ta_i), int(tb_i), ta_i, tb_i):
                            if name == op_name:
                                raw = res
                                break
                        if raw is None:
                            all_match = False
                            break
                        final = _rev_str(raw) if rev_res else raw
                        if final != t_out:
                            all_match = False
                            break
                    if all_match:
                        return op_name, rev_ops, rev_res, fmt
    return None


def _apply_op(
    op_name: str, rev_ops: bool, rev_res: bool, fmt: str, op_char: str,
    a_str: str, b_str: str,
) -> tuple[str, list[str]]:
    """Apply the discovered operation to (a_str, b_str). Returns (output, steps)."""
    steps: list[str] = []
    ta = a_str[::-1] if rev_ops else a_str
    tb = b_str[::-1] if rev_ops else b_str

    if rev_ops and rev_res:
        steps.append(f"Operands reversed [{a_str}->{ta}, {b_str}->{tb}], result will also be reversed.")
    elif rev_ops:
        steps.append(f"Operands reversed [{a_str}->{ta}, {b_str}->{tb}].")
    elif rev_res:
        steps.append("Result will be reversed at the end.")

    raw_result = None
    for name, res in _all_candidates(int(ta), int(tb), ta, tb):
        if name == op_name:
            raw_result = res
            break
    if raw_result is None:
        raise RuntimeError(f"op_name {op_name!r} not found among candidates")

    steps.append(f"Applying '{op_name}' to ({ta}, {tb}) gives {raw_result}.")
    final = _rev_str(raw_result) if rev_res else raw_result
    if rev_res:
        steps.append(f"After reversing the result: {final}")

    # Format-specific post-processing for negative results
    if fmt == "neg_suffix":
        if final.startswith("-"):
            stripped = final[1:]
            final = stripped + op_char
            steps.append(f"Result was negative, so write the suffix {op_char}: {final}")
    elif fmt == "neg_prefix":
        if final.startswith("-"):
            stripped = final[1:]
            final = op_char + stripped
            steps.append(f"Result was negative, so write the prefix {op_char}: {final}")
    return final, steps


def solve(prompt: str) -> Solution:
    # Find the example block
    if "Below are a few examples:" not in prompt or "Now, determine" not in prompt:
        return Solution(False, "", "", notes="no example block found")
    body = prompt.split("Below are a few examples:", 1)[1].split("Now, determine", 1)[0]

    parsed: list[tuple[str, str, str, str]] = []
    for line in body.strip().split("\n"):
        if " = " not in line:
            continue
        lhs, rhs = line.strip().split(" = ", 1)
        m = _EXAMPLE_RE.match(lhs.strip())
        if not m:
            return Solution(False, "", "", notes=f"non-numeric example: {lhs!r}")
        parsed.append((m.group(1), m.group(2), m.group(3), rhs.strip()))
    if not parsed:
        return Solution(False, "", "", notes="no examples parsed")

    # Parse question
    q_match = _QUESTION_RE.search(prompt)
    if not q_match:
        return Solution(False, "", "", notes="no question parsed")
    q_text = q_match.group(1).strip()
    qm = _EXAMPLE_RE.match(q_text)
    if not qm:
        return Solution(False, "", "", notes=f"non-numeric question: {q_text!r}")
    q_a, q_op, q_b = qm.group(1), qm.group(2), qm.group(3)

    # Group by op
    by_op: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for a, op, b, out in parsed:
        by_op[op].append((a, b, out))

    if q_op not in by_op:
        return Solution(False, "", "", notes=f"question op {q_op!r} not in examples")

    # Search for a matching op for the q_op group
    found = _find_op_for_group(by_op[q_op], q_op)
    if found is None:
        return Solution(False, "", "", notes=f"no candidate op matches {q_op!r} group ({len(by_op[q_op])} examples)")
    op_name, rev_ops, rev_res, fmt = found

    answer, query_steps = _apply_op(op_name, rev_ops, rev_res, fmt, q_op, q_a, q_b)

    # Build reasoning trace
    lines: list[str] = []
    lines.append(
        "Each example is `a<op>b = output`. The transformation depends on which "
        "operator symbol is used — different operators encode different rules. "
        "I'll group by operator, find the rule for the question's operator, "
        "then apply it."
    )
    lines.append("")
    lines.append("Examples:")
    for a, op, b, out in parsed:
        lines.append(f"  {a}{op}{b} = {out}")
    lines.append("")
    lines.append(f"Grouping by operator: {dict((k, len(v)) for k, v in by_op.items())}")
    lines.append("")
    lines.append(f"Question: {q_a}{q_op}{q_b}. The operator is `{q_op}`, with these examples:")
    for a, b, out in by_op[q_op]:
        lines.append(f"  {a}{q_op}{b} = {out}")
    lines.append("")
    lines.append(f"Testing candidate operations against the `{q_op}` group:")
    lines.append(f"Found: `{op_name}`" +
                 (f" with operands reversed" if rev_ops else "") +
                 (f", result reversed" if rev_res else "") +
                 (f", output format = {fmt}" if fmt != "plain" else "")
                 + ".")
    # Verify on each example with the trace
    for a, b, out in by_op[q_op]:
        v, _ = _apply_op(op_name, rev_ops, rev_res, fmt, q_op, a, b)
        lines.append(f"  Verify: {a}{q_op}{b} -> {v}  (gold {out})  {'✓' if v == out else '✗'}")
    lines.append("")
    lines.append(f"Applying to the question {q_a}{q_op}{q_b}:")
    for s in query_steps:
        lines.append(f"  {s}")
    lines.append(f"Result: {answer}")

    reasoning = "\n".join(lines)
    return Solution(
        success=True, answer=answer, reasoning=reasoning,
        notes=(
            f"op={op_name}, rev_ops={rev_ops}, rev_res={rev_res}, fmt={fmt}, "
            f"q_op={q_op!r}, n_examples_in_group={len(by_op[q_op])}"
        ),
    )
