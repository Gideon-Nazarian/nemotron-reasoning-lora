"""Solver for `bit_manipulation` problems — v3 (full shift composer).

The prompt says: "operations like bit shifts, rotations, XOR, AND, OR, NOT,
and possibly majority or choice functions."

**Rule grammar (covers ~1,423 of 1,428 = ~99% of compact-rule bit_manip
problems per Tong's investigations):**

    TERM      = [NOT] OP(n)                where OP ∈ {SHL, SHR, ROT}, n ∈ 1..7
    SINGLE    = TERM                        (148 problems in Tong's set)
    TWO_TERM  = TERM COMB TERM              (888 problems)
    THREE_TERM = (TERM COMB TERM) COMB TERM (387 problems)

`COMB ∈ {XOR, AND, OR}`. `ROT` is left-rotation (Tong's notation). All three
combiners are commutative, so we enforce lexicographic ordering on commutative
sub-pairs to halve the search.

Search-space sizes:
- TERM:       3 ops × 7 shifts × 2 (NOT yes/no) = 42 candidates
- SINGLE:     42 candidates
- TWO_TERM:   42 × 3 × 42 ≈ 5.3k candidates (de-duped: ~2.7k)
- THREE_TERM: ~2.7k inner × 3 × 42 ≈ 340k candidates

For each problem we try SINGLE, TWO_TERM, then THREE_TERM in order. We
precompute every TERM's value on every example input ONCE per problem, then
combine values via table lookups (vector ops). First-pair early-exit kills
~99.6% of candidates before evaluating the rest of the pairs.

Reasoning text is generated in Tong's investigation style: state the rule,
show 2 worked examples, then walk the query forward step-by-step.

The choice-function and majority-function rules (~240 prose-rule problems in
Tong's set, e.g. `ch(a,b,c) = (a&b) | (~a&c)`) are NOT covered by this
grammar — they require a different family of templates.
"""

from __future__ import annotations

import re
from typing import Optional

from .types import Solution

PAIR_RE = re.compile(r"^([01]{8})\s*->\s*([01]{8})\s*$", re.MULTILINE)
QUESTION_RE = re.compile(r"determine the output for:\s*([01]{8})", re.IGNORECASE)


# --- Atomic bit ops (8-bit, all wrapped to 0xFF) ---


def shl(x: int, n: int) -> int:
    return (x << n) & 0xFF


def shr(x: int, n: int) -> int:
    return (x >> n) & 0xFF


def rotl(x: int, n: int) -> int:
    n %= 8
    if n == 0:
        return x & 0xFF
    return ((x << n) | (x >> (8 - n))) & 0xFF


OP_FNS = {"SHL": shl, "SHR": shr, "ROT": rotl}
OPS = ("SHL", "SHR", "ROT")
COMBINERS = ("XOR", "AND", "OR")
COMB_FNS = {
    "XOR": lambda a, b: (a ^ b) & 0xFF,
    "AND": lambda a, b: (a & b) & 0xFF,
    "OR":  lambda a, b: (a | b) & 0xFF,
}


def _apply_term(x: int, op: str, n: int, negate: bool) -> int:
    """Apply a single TERM = [NOT] OP(n) to x."""
    v = OP_FNS[op](x, n)
    if negate:
        v = (~v) & 0xFF
    return v


def _term_str(op: str, n: int, negate: bool) -> str:
    """Human-readable form of a TERM, matching Tong's notation."""
    inner = f"{op}({n})"
    return f"NOT {inner}" if negate else inner


def _iter_terms():
    """Yield (op, n, negate) for every TERM in the grammar."""
    for op in OPS:
        for n in range(1, 8):
            for neg in (False, True):
                yield op, n, neg


def _matches_all(pairs, fn) -> bool:
    return all(fn(a) == b for a, b in pairs)


def _try_single(pairs):
    """Try every SINGLE = TERM rule. Return (op, n, negate) or None."""
    for op, n, neg in _iter_terms():
        if _matches_all(pairs, lambda x, op=op, n=n, neg=neg: _apply_term(x, op, n, neg)):
            return op, n, neg
    # Also try identity (rare but appears as `I XOR NOT` = NOT in Tong's set —
    # though that's just `NOT` of identity, which is the negate-only case below)
    if _matches_all(pairs, lambda x: x & 0xFF):
        return None  # identity — useless, skip
    return None


def _precompute_term_vals(inputs):
    """Return (terms_list, term_vals_matrix).

    `terms_list[i]` is a (op, n, neg) tuple. `term_vals_matrix[i][k]` is the
    value of `terms_list[i]` applied to `inputs[k]`. Lets the two/three-term
    search reuse work — each TERM is applied to each example exactly once.
    """
    terms = list(_iter_terms())
    vals = [[_apply_term(x, op, n, neg) for x in inputs] for op, n, neg in terms]
    return terms, vals


def _try_two_term(pairs, terms=None, term_vals=None, golds=None):
    """Try every TWO_TERM = TERM COMB TERM rule.

    Returns ((op1, n1, neg1), comb, (op2, n2, neg2)) or None.

    AND/OR/XOR are commutative, so we de-duplicate by enforcing `i <= j`
    on the term-index pair — about 2× speedup.
    """
    if terms is None:
        inputs = [a for a, _ in pairs]
        terms, term_vals = _precompute_term_vals(inputs)
    if golds is None:
        golds = [b for _, b in pairs]
    n = len(pairs)
    for i in range(len(terms)):
        ti = term_vals[i]
        for j in range(i, len(terms)):
            tj = term_vals[j]
            for comb in COMBINERS:
                fn = COMB_FNS[comb]
                # First-pair early exit
                if fn(ti[0], tj[0]) != golds[0]:
                    continue
                if all(fn(ti[k], tj[k]) == golds[k] for k in range(1, n)):
                    return terms[i], comb, terms[j]
    return None


def _try_three_term(pairs):
    """Try every THREE_TERM = (TERM COMB TERM) COMB TERM rule.

    Returns ((t1, c1, t2), c2, t3) or None.

    Outer combiner is also commutative, but the parenthesized inner pair is
    NOT swappable with the outer term (different operator), so we only dedupe
    the inner pair.
    """
    inputs = [a for a, _ in pairs]
    golds = [b for _, b in pairs]
    n = len(pairs)
    terms, term_vals = _precompute_term_vals(inputs)
    L = len(terms)

    for i in range(L):
        ti = term_vals[i]
        for j in range(i, L):  # commutative inner
            tj = term_vals[j]
            for cin in COMBINERS:
                fin = COMB_FNS[cin]
                # Compute inner values for ALL example inputs
                inner = [fin(ti[k], tj[k]) for k in range(n)]
                for kk in range(L):
                    tk = term_vals[kk]
                    for cout in COMBINERS:
                        fout = COMB_FNS[cout]
                        # First-pair early exit
                        if fout(inner[0], tk[0]) != golds[0]:
                            continue
                        ok = True
                        for kk2 in range(1, n):
                            if fout(inner[kk2], tk[kk2]) != golds[kk2]:
                                ok = False
                                break
                        if ok:
                            return (terms[i], cin, terms[j]), cout, terms[kk]
    return None


def _format_single_reasoning(pairs, op, n, neg, q_input, q_output):
    """Tong-style reasoning text for a single-term rule."""
    rule_str = _term_str(op, n, neg)
    samples = []
    for a, b in pairs[:3]:
        v = _apply_term(a, op, n, neg)
        assert v == b
        samples.append(
            f"  - For input `{a:08b}`: {rule_str.replace(op + '(' + str(n) + ')', f'{op}({a:08b}, {n}) = {OP_FNS[op](a, n):08b}')} → output `{v:08b}`."
        )
    op_verb = {"SHL": "logical left shift", "SHR": "logical right shift", "ROT": "rotate left"}[op]
    inner_val = OP_FNS[op](q_input, n)
    if neg:
        step_query = (
            f"1. Start with `x = {q_input:08b}`.\n"
            f"2. Apply {op}({n}) ({op_verb} by {n}): {inner_val:08b}.\n"
            f"3. Apply NOT: {q_output:08b}."
        )
    else:
        step_query = (
            f"1. Start with `x = {q_input:08b}`.\n"
            f"2. Apply {op}({n}) ({op_verb} by {n}): {q_output:08b}."
        )
    return (
        f"The transformation is `{rule_str}` — i.e. {op_verb} by {n}"
        f"{', then bitwise NOT' if neg else ''}.\n\n"
        f"Verifying on the examples:\n"
        + "\n".join(samples)
        + f"\n\nApplying to the query:\n{step_query}"
    )


def _format_two_term_reasoning(pairs, term1, comb, term2, q_input, q_output):
    """Tong-style reasoning text for a two-term rule."""
    op1, n1, neg1 = term1
    op2, n2, neg2 = term2
    rule_str = f"{_term_str(op1, n1, neg1)} {comb} {_term_str(op2, n2, neg2)}"
    samples = []
    for a, b in pairs[:2]:
        v1 = _apply_term(a, op1, n1, neg1)
        v2 = _apply_term(a, op2, n2, neg2)
        out = COMB_FNS[comb](v1, v2)
        assert out == b
        samples.append(
            f"  - For `{a:08b}`: "
            f"`{_term_str(op1, n1, neg1)} = {v1:08b}`, "
            f"`{_term_str(op2, n2, neg2)} = {v2:08b}`, "
            f"`{v1:08b} {comb} {v2:08b} = {out:08b}` ✓"
        )

    op1_verb = {"SHL": "left-shifting", "SHR": "right-shifting", "ROT": "left-rotating"}[op1]
    op2_verb = {"SHL": "left-shifting", "SHR": "right-shifting", "ROT": "left-rotating"}[op2]

    v1_q = _apply_term(q_input, op1, n1, neg1)
    v2_q = _apply_term(q_input, op2, n2, neg2)
    inner1_q = OP_FNS[op1](q_input, n1)
    inner2_q = OP_FNS[op2](q_input, n2)

    step_lines = [f"1. Start with `x = {q_input:08b}`."]
    step_lines.append(f"2. Compute `{op1}({n1})`: {op1_verb} `x` by {n1} → `{inner1_q:08b}`.")
    if neg1:
        step_lines.append(f"3. Apply NOT to that: `{v1_q:08b}` (the `{_term_str(op1, n1, neg1)}` term).")
        next_step = 4
    else:
        next_step = 3
    step_lines.append(f"{next_step}. Compute `{op2}({n2})`: {op2_verb} `x` by {n2} → `{inner2_q:08b}`.")
    if neg2:
        step_lines.append(f"{next_step + 1}. Apply NOT to that: `{v2_q:08b}` (the `{_term_str(op2, n2, neg2)}` term).")
        next_step += 2
    else:
        next_step += 1
    step_lines.append(f"{next_step}. Combine: `{v1_q:08b} {comb} {v2_q:08b} = {q_output:08b}`.")

    return (
        f"The transformation is `{rule_str}`.\n\n"
        f"Verifying on two examples:\n"
        + "\n".join(samples)
        + f"\n\nApplying to the query:\n"
        + "\n".join(step_lines)
    )


def _format_three_term_reasoning(pairs, term1, cin, term2, cout, term3, q_input, q_output):
    """Tong-style reasoning text for a three-term rule (T1 cin T2) cout T3."""
    op1, n1, neg1 = term1
    op2, n2, neg2 = term2
    op3, n3, neg3 = term3
    inner_str = f"{_term_str(op1, n1, neg1)} {cin} {_term_str(op2, n2, neg2)}"
    rule_str = f"({inner_str}) {cout} {_term_str(op3, n3, neg3)}"

    samples = []
    for a, b in pairs[:2]:
        v1 = _apply_term(a, op1, n1, neg1)
        v2 = _apply_term(a, op2, n2, neg2)
        inner_v = COMB_FNS[cin](v1, v2)
        v3 = _apply_term(a, op3, n3, neg3)
        out = COMB_FNS[cout](inner_v, v3)
        assert out == b
        samples.append(
            f"  - For `{a:08b}`: "
            f"`{_term_str(op1, n1, neg1)}={v1:08b}`, "
            f"`{_term_str(op2, n2, neg2)}={v2:08b}`, "
            f"inner=`{v1:08b} {cin} {v2:08b} = {inner_v:08b}`; "
            f"`{_term_str(op3, n3, neg3)}={v3:08b}`, "
            f"final=`{inner_v:08b} {cout} {v3:08b} = {out:08b}` ✓"
        )

    v1_q = _apply_term(q_input, op1, n1, neg1)
    v2_q = _apply_term(q_input, op2, n2, neg2)
    inner_q = COMB_FNS[cin](v1_q, v2_q)
    v3_q = _apply_term(q_input, op3, n3, neg3)

    steps = [
        f"1. Start with `x = {q_input:08b}`.",
        f"2. Compute `{_term_str(op1, n1, neg1)} = {v1_q:08b}`.",
        f"3. Compute `{_term_str(op2, n2, neg2)} = {v2_q:08b}`.",
        f"4. Inner combination: `{v1_q:08b} {cin} {v2_q:08b} = {inner_q:08b}`.",
        f"5. Compute `{_term_str(op3, n3, neg3)} = {v3_q:08b}`.",
        f"6. Final combination: `{inner_q:08b} {cout} {v3_q:08b} = {q_output:08b}`.",
    ]
    return (
        f"The transformation is `{rule_str}`.\n\n"
        f"Verifying on two examples:\n"
        + "\n".join(samples)
        + f"\n\nApplying to the query:\n"
        + "\n".join(steps)
    )


def solve(prompt: str) -> Solution:
    pairs_str = PAIR_RE.findall(prompt)
    if len(pairs_str) < 4:
        return Solution(False, "", "", notes=f"only {len(pairs_str)} pairs parsed")
    pairs = [(int(a, 2), int(b, 2)) for a, b in pairs_str]

    q_match = QUESTION_RE.search(prompt)
    if not q_match:
        return Solution(False, "", "", notes="no question parsed")
    q_input_str = q_match.group(1)
    q_input = int(q_input_str, 2)

    # Tier 1: single TERM
    single = _try_single(pairs)
    if single is not None:
        op, n, neg = single
        q_output = _apply_term(q_input, op, n, neg)
        answer = format(q_output, "08b")
        reasoning = _format_single_reasoning(pairs, op, n, neg, q_input, q_output)
        return Solution(
            success=True, answer=answer, reasoning=reasoning,
            notes=f"rule=single:{_term_str(op, n, neg)}, n_pairs={len(pairs)}",
        )

    # Tier 2: two TERMs combined
    two = _try_two_term(pairs)
    if two is not None:
        term1, comb, term2 = two
        op1, n1, neg1 = term1
        op2, n2, neg2 = term2
        v1 = _apply_term(q_input, op1, n1, neg1)
        v2 = _apply_term(q_input, op2, n2, neg2)
        q_output = COMB_FNS[comb](v1, v2)
        answer = format(q_output, "08b")
        reasoning = _format_two_term_reasoning(pairs, term1, comb, term2, q_input, q_output)
        return Solution(
            success=True, answer=answer, reasoning=reasoning,
            notes=f"rule=two_term:{_term_str(*term1)} {comb} {_term_str(*term2)}, n_pairs={len(pairs)}",
        )

    # Tier 3: three TERMs parenthesized (T1 cin T2) cout T3
    three = _try_three_term(pairs)
    if three is not None:
        (term1, cin, term2), cout, term3 = three
        op1, n1, neg1 = term1
        op2, n2, neg2 = term2
        op3, n3, neg3 = term3
        v1 = _apply_term(q_input, op1, n1, neg1)
        v2 = _apply_term(q_input, op2, n2, neg2)
        inner_q = COMB_FNS[cin](v1, v2)
        v3 = _apply_term(q_input, op3, n3, neg3)
        q_output = COMB_FNS[cout](inner_q, v3)
        answer = format(q_output, "08b")
        reasoning = _format_three_term_reasoning(
            pairs, term1, cin, term2, cout, term3, q_input, q_output
        )
        return Solution(
            success=True, answer=answer, reasoning=reasoning,
            notes=(
                f"rule=three_term:({_term_str(*term1)} {cin} {_term_str(*term2)}) "
                f"{cout} {_term_str(*term3)}, n_pairs={len(pairs)}"
            ),
        )

    return Solution(False, "", "", notes=f"no rule matched all {len(pairs)} pairs")
