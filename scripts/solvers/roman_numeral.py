"""Solver for `roman_numeral` problems.

Prompt structure:
    In Alice's Wonderland, numbers are secretly converted into a different numeral system.
    Some examples are given below:
    75 -> LXXV
    3 -> III
    ...
    Now, write the number 62 in the Wonderland numeral system.

Despite the framing as a "secret" system, examples reveal it's always the
standard Roman numeral system. We verify this by checking each example matches
our standard converter, then apply the converter to the question's number.
"""

from __future__ import annotations

import re

from .types import Solution

# Match: "<int> -> <ROMAN>"
EXAMPLE_RE = re.compile(r"(\d+)\s*->\s*([IVXLCDM]+)")
QUESTION_RE = re.compile(r"write the number\s+(\d+)\s+in", re.IGNORECASE)

# Standard Roman numeral pairs (descending), supports subtractive notation
ROMAN_PAIRS = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"),
    (1, "I"),
]


def to_roman(n: int) -> str:
    if n <= 0:
        return ""
    parts = []
    for value, sym in ROMAN_PAIRS:
        while n >= value:
            parts.append(sym)
            n -= value
    return "".join(parts)


def solve(prompt: str) -> Solution:
    examples = [(int(a), b) for a, b in EXAMPLE_RE.findall(prompt)]
    if not examples:
        return Solution(False, "", "", notes="no examples parsed")

    # Verify each example matches the standard Roman conversion
    for n, expected in examples:
        if to_roman(n) != expected:
            return Solution(
                False, "", "",
                notes=f"example {n} -> {expected!r} doesn't match standard Roman ({to_roman(n)!r})",
            )

    q_match = QUESTION_RE.search(prompt)
    if not q_match:
        return Solution(False, "", "", notes="no question parsed")
    q_n = int(q_match.group(1))

    answer = to_roman(q_n)

    example_str = "; ".join(f"{n} → {r}" for n, r in examples[:3])
    reasoning = (
        f"Each example matches standard Roman numerals ({example_str}). "
        f"Converting {q_n}: {answer}."
    )
    return Solution(
        success=True, answer=answer, reasoning=reasoning,
        notes=f"n_examples={len(examples)}",
    )
