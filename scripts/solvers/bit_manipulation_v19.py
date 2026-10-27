"""v19 procedural-search reasoning for bit_manipulation (critique-hardened).

Wraps the v3 solver's search (scripts/solvers/bit_manipulation.py) but emits reasoning
that the LLM can actually EXECUTE token-by-token at inference, not a Python loop's summary.

Design fixes (from the 2026-06-01 external critique, each validated against our history):
  #1 INVISIBLE-LOOP fix: show 2-3 EXPLICIT failed candidate tests with real values
     ("I test ROT(1): 01010001 -> 10100010, compare to 11011101 -> no") before the match,
     so the model learns to USE its tokens as scratchpad, not hallucinate "42 candidates".
     (Validated: our eval notes show the model hallucinates summary phrases w/o doing work.)
  #2 VERTICAL BIT-MATH fix: lay XOR/AND/OR out bit-column-by-bit-column, not inline
     horizontal. (Validated: dgxchen's 0.84 bit_manip traces use exactly this columnar style;
     our notes show inline arithmetic produces off-by-one errors at inference.)
  #3 TIER-3 CONTAMINATION fix: three-operation problems ALSO narrate the search procedurally
     (singles fail -> pairs fail -> triples -> converge). NO fallback to v18 oracle style.
     (Validated: mixing two reasoning styles = v14's -0.16 dual-template collapse.)

Token budget: ~700 tok/trace (vs 283 terse). Keeps corpus mean rise ~6%, real run ~9h < 12h cap.
Originality: OUR solver's OWN search exposed. answer==gold gated. No distillation.
"""
from __future__ import annotations

import textwrap

from . import bit_manipulation as bm
from .types import Solution

_term_str = bm._term_str


def _wrap(text: str, width: int = 100) -> str:
    """Wrap a prose paragraph to <=width cols so no line runs long enough to soft-wrap
    in a viewer/training display and visually break the bit-column tables below it."""
    return "\n".join(textwrap.wrap(text, width=width)) if text else text


def _bits(x: int):
    """MSB-first list of 8 bits."""
    return [(x >> (7 - i)) & 1 for i in range(8)]


def _vertical_combine(label_a, va, label_b, vb, comb):
    """Render `va COMB vb` as a vertical bit-column table (fix #2).

    Returns (result_int, text_block). Columns are MSB(bit7)..LSB(bit0) top-to-bottom,
    so the model lines up bits like long-hand arithmetic on paper.
    """
    fn = bm.COMB_FNS[comb]
    res = fn(va, vb)
    ba, bb, br = _bits(va), _bits(vb), _bits(res)
    sym = {"XOR": "^", "AND": "&", "OR": "|"}[comb]
    # Operator row carries the combiner symbol after the label; the first operand,
    # "bit", and "result" rows have no symbol. Build the left gutter as
    # "<label> <sym>" and pad ALL gutters to one fixed width so the 8 bit columns
    # line up character-for-character regardless of label length (fix: true vertical
    # alignment — long labels like "NOT ROT(1)" previously shifted their row right).
    g_bit = "bit"
    g_a = label_a
    g_b = f"{label_b} {sym}"
    g_res = "result"
    w = max(len(g_bit), len(g_a), len(g_b), len(g_res))
    def row(gutter, cells, tail=""):
        return f"  {gutter:>{w}s} : {' '.join(cells)}{tail}"
    lines = [
        row(g_bit, ["7", "6", "5", "4", "3", "2", "1", "0"]),
        row(g_a, [str(x) for x in ba]),
        row(g_b, [str(x) for x in bb]),
        f"  {'':>{w}s}   {'-' * 15}",
        row(g_res, [str(x) for x in br], tail=f"   = {res:08b}"),
    ]
    return res, "\n".join(lines)


def _explicit_single_failures(pairs, max_show=3):
    """Fix #1: yield up-to-3 concrete failed single-op tests on example 0, then the verdict."""
    a0, b0 = pairs[0]
    shown = []
    matched0 = []  # candidates that match ex0 (worth fully checking)
    for op, n, neg in bm._iter_terms():
        v0 = bm._apply_term(a0, op, n, neg)
        if v0 == b0:
            matched0.append((op, n, neg))
        elif len(shown) < max_show:
            shown.append((op, n, neg, v0))
    lines = []
    for op, n, neg, v0 in shown:
        lines.append(f"  - Test `{_term_str(op,n,neg)}`: `{a0:08b}` -> `{v0:08b}`. "
                     f"Compare to target `{b0:08b}` -> mismatch, reject.")
    return lines, matched0


def _try_single_verified(pairs):
    for op, n, neg in bm._iter_terms():
        if bm._matches_all(pairs, lambda x, op=op, n=n, neg=neg: bm._apply_term(x, op, n, neg)):
            return (op, n, neg)
    return None


def solve(prompt: str) -> Solution:
    pairs_str = bm.PAIR_RE.findall(prompt)
    if len(pairs_str) < 4:
        return Solution(False, "", "", notes=f"only {len(pairs_str)} pairs parsed")
    pairs = [(int(a, 2), int(b, 2)) for a, b in pairs_str]
    q_match = bm.QUESTION_RE.search(prompt)
    if not q_match:
        return Solution(False, "", "", notes="no question parsed")
    q_input = int(q_match.group(1), 2)
    a0, b0 = pairs[0]

    N = [_wrap("I need to find the hidden bit rule by testing candidate operations against the "
               "examples and rejecting the ones that don't reproduce them. I work bit-by-bit so "
               "I don't make shift or XOR mistakes.") + "\n",
         _wrap(f"**Step 1 — single operations** `[NOT] OP(n)`, OP in {{SHL, SHR, ROT}}, n in 1..7. "
               f"I check each against the first example `{a0:08b} -> {b0:08b}`:")]

    fail_lines, matched0 = _explicit_single_failures(pairs)
    N += fail_lines

    single = _try_single_verified(pairs)
    if single is not None:
        op, n, neg = single
        # show it matching ex0, then confirm across all
        v0 = bm._apply_term(a0, op, n, neg)
        N.append(f"  - Test `{_term_str(op,n,neg)}`: `{a0:08b}` -> `{v0:08b}` = target `{b0:08b}` ✓. "
                 f"It also holds on all {len(pairs)} examples. **Rule found.**")
        q_output = bm._apply_term(q_input, op, n, neg)
        N.append(f"\n**Apply to query** `{q_input:08b}`: `{_term_str(op,n,neg)}` = `{q_output:08b}`.")
        return Solution(True, format(q_output, "08b"), "\n".join(N), notes=f"v19:single:{_term_str(op,n,neg)}")

    N.append(f"  - After testing the single operations, none reproduces every example "
             f"(a single shift/rotate is insufficient).")

    # Tier 2 — two-term, with vertical math (fix #2)
    inputs = [a for a, _ in pairs]; golds = [b for _, b in pairs]
    terms, term_vals = bm._precompute_term_vals(inputs)
    N.append("\n" + _wrap(f"**Step 2 — two operations** `TERM1 COMB TERM2`, COMB in {{XOR, AND, OR}}. "
                          f"I look for a pair whose combination matches `{a0:08b} -> {b0:08b}`:"))
    two = bm._try_two_term(pairs, terms, term_vals, golds)
    if two is not None:
        t1, comb, t2 = two
        v1_0 = bm._apply_term(a0, *t1); v2_0 = bm._apply_term(a0, *t2)
        res0, vtable = _vertical_combine(_term_str(*t1), v1_0, _term_str(*t2), v2_0, comb)
        N.append(f"  Try `{_term_str(*t1)} {comb} {_term_str(*t2)}` on the first example, bit by bit:")
        N.append(vtable)
        N.append(f"  That equals the target `{b0:08b}` ✓, and it holds on all {len(pairs)} examples. **Rule found.**")
        v1 = bm._apply_term(q_input, *t1); v2 = bm._apply_term(q_input, *t2)
        qres, qtable = _vertical_combine(_term_str(*t1), v1, _term_str(*t2), v2, comb)
        N.append(f"\n**Apply to query** `{q_input:08b}`:")
        N.append(qtable)
        return Solution(True, format(qres, "08b"), "\n".join(N), notes=f"v19:two:{_term_str(*t1)} {comb} {_term_str(*t2)}")

    N.append("  - No two-operation combination matches every example either.")

    # Tier 3 — three-term, FULLY procedural + vertical (fix #3: no oracle fallback)
    three = bm._try_three_term(pairs)
    if three is not None:
        (t1, cin, t2), cout, t3 = three
        N.append("\n" + _wrap(
            f"**Step 3 — three operations** `(TERM1 COMB TERM2) COMB TERM3`. "
            f"Extending the search to a parenthesized inner pair plus an outer term, "
            f"I find `({_term_str(*t1)} {cin} {_term_str(*t2)}) {cout} {_term_str(*t3)}`. "
            f"Verify on the first example bit by bit:"))
        v1_0 = bm._apply_term(a0, *t1); v2_0 = bm._apply_term(a0, *t2)
        inner0, itable = _vertical_combine(_term_str(*t1), v1_0, _term_str(*t2), v2_0, cin)
        N.append("  Inner combination:")
        N.append(itable)
        v3_0 = bm._apply_term(a0, *t3)
        fin0, ftable = _vertical_combine("inner", inner0, _term_str(*t3), v3_0, cout)
        N.append("  Outer combination:")
        N.append(ftable)
        N.append(f"  That equals the target `{b0:08b}` ✓, and it holds on all {len(pairs)} examples. **Rule found.**")
        # apply to query
        v1 = bm._apply_term(q_input, *t1); v2 = bm._apply_term(q_input, *t2)
        inner_q, iqt = _vertical_combine(_term_str(*t1), v1, _term_str(*t2), v2, cin)
        v3 = bm._apply_term(q_input, *t3)
        fin_q, fqt = _vertical_combine("inner", inner_q, _term_str(*t3), v3, cout)
        N.append(f"\n**Apply to query** `{q_input:08b}`:")
        N.append("  Inner:"); N.append(iqt)
        N.append("  Outer:"); N.append(fqt)
        return Solution(True, format(fin_q, "08b"), "\n".join(N),
                        notes=f"v19:three:({_term_str(*t1)} {cin} {_term_str(*t2)}) {cout} {_term_str(*t3)}")

    return Solution(False, "", "", notes=f"no rule matched all {len(pairs)} pairs")
