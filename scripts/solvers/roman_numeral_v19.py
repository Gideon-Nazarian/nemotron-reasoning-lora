"""v19 procedural reasoning for numeral_system (Roman numerals).

The v18/current trace is pure oracle: "Each example matches standard Roman numerals.
Converting 38: XXXVIII." — it ASSERTS the system is Roman and ASSERTS the answer,
showing none of the work. numeral_system was our THINNEST category (18-word mean).

v19 exposes the solver's actual two-phase logic (critique fixes applied):
  Phase 1 — IDENTIFY the system by VERIFICATION (fix #1, no telepathy): take an example
    number, convert it ourselves via the standard ladder, and check it reproduces the
    example output. Do this for 2-3 examples to *establish* (not assume) it's standard Roman.
  Phase 2 — DERIVE the answer via the subtraction ladder, one symbol at a time (fix #2,
    show the arithmetic vertically/stepwise so the model executes it rather than recalling
    a memorized mapping): 38 -> take XXX (30), remainder 8 -> VIII -> XXXVIII.

Token budget: small category, traces ~150-250 tok. Originality: our solver's own logic
exposed; answer==gold gated.
"""
from __future__ import annotations

import textwrap

from . import roman_numeral as rn
from .types import Solution


def _wrap(text, width=100):
    return "\n".join(textwrap.wrap(text, width=width)) if text else text


def _ladder_steps(n: int):
    """Walk the greedy subtraction ladder, recording each (value, symbol, remainder) step.
    This is exactly what rn.to_roman does, but we capture the trace."""
    steps = []
    rem = n
    for value, sym in rn.ROMAN_PAIRS:
        cnt = 0
        while rem >= value:
            rem -= value
            cnt += 1
        if cnt:
            steps.append((value, sym, cnt, rem))
    return steps


def _format_ladder(n: int) -> (str, str):
    """Return (roman, narration) for converting n via the explicit ladder.

    Fix #2 (critique): when a symbol repeats (e.g. XXX), spell out EACH subtraction
    on its own line — `38 - 10 = 28`, `28 - 10 = 18`, `18 - 10 = 8` — rather than the
    inline `38 - 3*10 = 8`. This keeps every arithmetic step a single subtraction the
    model can execute without combining multiply+subtract in one horizontal jump.
    """
    lines = [f"  Start with {n}. I subtract the largest Roman value that still fits, "
             f"one symbol at a time:"]
    roman = ""
    running = n
    for value, sym in rn.ROMAN_PAIRS:
        while running >= value:
            after = running - value
            lines.append(f"  - {running} >= {value} ({sym}): take `{sym}`, {running} - {value} = {after}.")
            roman += sym
            running = after
    lines.append(f"  Nothing left (remainder 0). Reading the symbols in order: `{roman}`.")
    return roman, "\n".join(lines)


def solve(prompt: str) -> Solution:
    examples = [(int(a), b) for a, b in rn.EXAMPLE_RE.findall(prompt)]
    if not examples:
        return Solution(False, "", "", notes="no examples parsed")
    # The base solver requires every example to match standard Roman; keep that guard.
    for n, expected in examples:
        if rn.to_roman(n) != expected:
            return Solution(False, "", "",
                            notes=f"example {n} -> {expected!r} != standard ({rn.to_roman(n)!r})")
    q_match = rn.QUESTION_RE.search(prompt)
    if not q_match:
        return Solution(False, "", "", notes="no question parsed")
    q_n = int(q_match.group(1))

    # Fix #1 (critique): ground the hypothesis in the characters ACTUALLY present in the
    # example outputs, instead of handing the model the Roman alphabet out of the ether.
    seen_chars = sorted({c for _, out in examples for c in out})
    ex_preview = ", ".join(out for _, out in examples[:3])
    roman_set = set("IVXLCDM")
    are_roman = set(seen_chars) <= roman_set
    if are_roman:
        hypo = (f"Looking at the example outputs ({ex_preview}), the strings use only the "
                f"characters {{{', '.join(seen_chars)}}} — all classic Roman numeral symbols "
                f"(I=1, V=5, X=10, L=50, C=100, D=500, M=1000). So I hypothesize this is the "
                f"standard Roman numeral system, and I VERIFY that by converting the example "
                f"numbers myself before trusting it.")
    else:
        hypo = (f"Looking at the example outputs ({ex_preview}), the strings use the characters "
                f"{{{', '.join(seen_chars)}}}. I check whether these behave as standard Roman "
                f"numeral symbols by converting the example numbers myself and comparing.")
    N = [_wrap(hypo)]
    # Phase 1: verification on up to 3 examples, showing the ladder for the first
    N.append("\n**Verify the rule on the examples:**")
    for i, (n, expected) in enumerate(examples[:3]):
        got, ladder = _format_ladder(n)
        ok = "✓" if got == expected else "✗"
        if i == 0:
            N.append(f"- Example `{n} -> {expected}`. Convert {n} myself:")
            N.append(ladder)
            N.append(f"  Got `{got}` = `{expected}` {ok}.")
        else:
            N.append(f"- Example `{n} -> {expected}`: standard conversion gives `{got}` {ok}.")
    N.append(_wrap("All examples reproduce, so the system is standard Roman numerals."))

    # Phase 2: derive the answer
    answer, ladder_q = _format_ladder(q_n)
    N.append(f"\n**Convert the target {q_n}:**")
    N.append(ladder_q)
    N.append(f"\nThe answer is `{answer}`.")

    return Solution(True, answer, "\n".join(N), notes=f"v19:roman:n_examples={len(examples)}")
