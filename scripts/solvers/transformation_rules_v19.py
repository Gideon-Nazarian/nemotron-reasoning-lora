"""v19 procedural reasoning for transformation_rules (operator-redefinition puzzles).

The v18/current trace announces the winner without the search: "Testing candidate
operations... Found: `concatenation`." — the model never sees the test-and-eliminate
that the solver actually performs (critique #1, the invisible-loop illusion).

v19 exposes the solver's REAL candidate search (critique fixes applied):
  #1 EXPLICIT failed tests: show 2-3 candidate operations that FAIL on the first example
     in the operator's group ("try addition: 64+65=129, but output is 201 -> no") before
     the one that matches, so the model learns to test-and-reject rather than hallucinate.
  #2 SHOW THE ARITHMETIC inline per candidate (small numbers, so horizontal is fine here —
     unlike bit_manip's 8-bit ops; the risk the critique flagged was multi-bit, not 2-digit add).
  #3 UNIFORM procedural style across all solved problems (no oracle fallback).

Key honesty: operator-redefinition has a GENUINE search (group by operator, test a fixed
candidate list, accept the first reproducing ALL examples in that group) — the richest of
the three weak categories to expose. Token budget ~300-450 tok.
Originality: our solver's own candidate search exposed; answer==gold gated.
"""
from __future__ import annotations

import textwrap
from collections import defaultdict

from . import transformation_rules as tr
from .types import Solution


def _wrap(text, width=100):
    return "\n".join(textwrap.wrap(text, width=width)) if text else text


def _candidate_value(a_str, b_str, op_name, rev_ops, rev_res, fmt, op_char):
    """Compute what a named candidate op produces on (a_str,b_str) under the given flags,
    matching the solver's _apply_op logic. Returns the formatted output string or None."""
    try:
        out, _ = tr._apply_op(op_name, rev_ops, rev_res, fmt, op_char, a_str, b_str)
        return out
    except Exception:
        return None


def _failed_candidate_narration(group, q_op, winner):
    """Fix #1: show 2-3 plain candidate ops that FAIL on the first group example, then stop.
    Uses the plain/no-reversal candidate list so the failures are simple and truthful."""
    a0, b0, out0 = group[0]
    lines = []
    shown = 0
    win_name = winner[0]
    for name, res in tr._all_candidates(int(a0), int(b0), a0, b0):
        if name == win_name:
            continue  # don't show the winner among the failures
        if res == out0:
            continue  # this candidate also matches ex0 — skip (not a clean "fail")
        # a genuine failure on example 0
        lines.append(f"  - Try **{name}**: on `{a0}{q_op}{b0}`, that gives `{res}`, "
                     f"but the output is `{out0}` -> no.")
        shown += 1
        if shown >= 3:
            break
    return lines


def solve(prompt: str) -> Solution:
    # reuse the base parser/search by calling it, then RE-narrate procedurally
    base = tr.solve(prompt)
    if not base.success:
        return base  # unsolved — dropped from corpus anyway

    # re-parse to get structured pieces (mirror tr.solve)
    body = prompt.split("Below are a few examples:", 1)[1].split("Now, determine", 1)[0]
    parsed = []
    for line in body.strip().split("\n"):
        if " = " not in line:
            continue
        lhs, rhs = line.strip().split(" = ", 1)
        m = tr._EXAMPLE_RE.match(lhs.strip())
        if not m:
            return base
        parsed.append((m.group(1), m.group(2), m.group(3), rhs.strip()))
    q_match = tr._QUESTION_RE.search(prompt)
    qm = tr._EXAMPLE_RE.match(q_match.group(1).strip())
    q_a, q_op, q_b = qm.group(1), qm.group(2), qm.group(3)

    by_op = defaultdict(list)
    for a, op, b, out in parsed:
        by_op[op].append((a, b, out))
    group = by_op[q_op]

    found = tr._find_op_for_group(group, q_op)
    op_name, rev_ops, rev_res, fmt = found

    N = [_wrap("Each line is `a <op> b = output`, and the rule depends on WHICH operator symbol "
               "is used — different symbols encode different transformations. So I first group "
               "the examples by operator, then search for the operation that reproduces every "
               "example sharing the question's operator.")]

    N.append("\n**Group the examples by operator:**")
    for op, items in by_op.items():
        N.append(f"  - `{op}` : " + ", ".join(f"{a}{op}{b} = {o}" for a, b, o in items))

    N.append(_wrap(f"\nThe question is `{q_a}{q_op}{q_b}`, so I only need the rule for operator "
                   f"`{q_op}` ({len(group)} example{'s' if len(group)!=1 else ''})."))

    N.append(f"\n**Search candidate operations against the `{q_op}` examples** "
             f"(reject any that fail the first one):")
    fails = _failed_candidate_narration(group, q_op, found)
    N += fails

    # show the winner matching, with per-example verification
    a0, b0, out0 = group[0]
    win_val0 = _candidate_value(a0, b0, op_name, rev_ops, rev_res, fmt, q_op)
    extra = ""
    if rev_ops and rev_res:
        extra = " (operands and result both read reversed)"
    elif rev_ops:
        extra = " (operands read reversed)"
    elif rev_res:
        extra = " (result read reversed)"
    N.append(f"  - Try **{op_name}**{extra}: on `{a0}{q_op}{b0}` that gives `{win_val0}` "
             f"= `{out0}` ✓.")
    if len(group) > 1:
        N.append("  Verify it holds on the rest of the group:")
        for a, b, out in group[1:]:
            v = _candidate_value(a, b, op_name, rev_ops, rev_res, fmt, q_op)
            N.append(f"    - `{a}{q_op}{b}` -> `{v}` (gold `{out}`) {'✓' if v==out else '✗'}")
    N.append(f"  All {len(group)} match. The rule for `{q_op}` is **{op_name}**{extra}.")

    # apply to the question, showing steps
    answer, steps = tr._apply_op(op_name, rev_ops, rev_res, fmt, q_op, q_a, q_b)
    N.append(f"\n**Apply to `{q_a}{q_op}{q_b}`:**")
    for s in steps:
        N.append(f"  - {s}")
    N.append(f"\nThe answer is `{answer}`.")

    return Solution(True, answer, "\n".join(N),
                    notes=f"v19:transform:op={op_name},q_op={q_op!r},n_group={len(group)}")
