"""Solver for `unit_conversion` problems — hybrid (long-arithmetic reasoning).

Prompt structure:
    In Alice's Wonderland, a secret unit conversion is applied to measurements.
    For example:
        36.97 m becomes 27.24
        29.73 m becomes 21.91
        ...
    Now, convert the following measurement: 40.7 m

The conversion is always linear: output = ratio * input. The unit ('m', 'kg',
etc.) is decorative.

Algorithm:
  1. Parse pairs (a, b); compute ratio_i = b_i / a_i.
  2. From candidates [round(mean, 2), round(mean, 3), round(mean, 4), mean],
     pick the one that back-tests (rounded to 2dp output) against the most pairs.
  3. Apply: q_out = q_in * best_ratio, round to 2dp.

Empirically: 1328/1594 = 83.3% gold-match coverage; the 266 misses are all
2dp rounding edges (gold answers are 100% 2dp). Median-of-truncated-ratio
doesn't beat back-tested-mean here.

**Upgrade vs v1 (the v11 lever): long-arithmetic reasoning trace** — show the
long-division for each example's ratio computation, then long-multiplication
for the query. Per the 2026-05-17 research findings: this trains arithmetic
capability, not just answer-shape memorization.
"""

from __future__ import annotations

import re
import statistics

from . import _arithmetic as arith
from .types import Solution

EXAMPLE_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*[a-zA-Z]{0,5}\s+becomes\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
QUESTION_RE = re.compile(
    r"convert the following measurement:\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def solve(prompt: str) -> Solution:
    pair_matches = EXAMPLE_RE.findall(prompt)
    if len(pair_matches) < 2:
        return Solution(False, "", "", notes=f"only {len(pair_matches)} pairs parsed")
    pairs = [(a, b, float(a), float(b)) for a, b in pair_matches]

    q_match = QUESTION_RE.search(prompt)
    if not q_match:
        return Solution(False, "", "", notes="no question parsed")
    q_str = q_match.group(1)
    q_input = float(q_str)

    # --- Algorithmic core: back-tested mean ratio ---
    ratios = [b / a for _, _, a, b in pairs if a != 0]
    if not ratios:
        return Solution(False, "", "", notes="all inputs zero")
    ratio_mean = statistics.mean(ratios)
    ratio_stdev = statistics.pstdev(ratios) if len(ratios) > 1 else 0.0
    rel_spread = ratio_stdev / abs(ratio_mean) if ratio_mean != 0 else float("inf")
    if rel_spread > 0.01:
        return Solution(False, "", "", notes=f"ratios spread too large: {rel_spread:.6f}")

    def back_test(r: float) -> int:
        return sum(1 for _, _, a, b in pairs if abs(round(a * r, 2) - b) < 1e-9)

    candidates = [round(ratio_mean, 2), round(ratio_mean, 3), round(ratio_mean, 4), ratio_mean]
    best_score = max(back_test(r) for r in candidates)
    best_ratio = next(r for r in candidates if back_test(r) == best_score)

    q_output = q_input * best_ratio
    answer = f"{q_output:.2f}"

    # --- Long-arithmetic reasoning trace ---
    lines: list[str] = []
    lines.append(
        "Each output is the input times a constant ratio. I derive the ratio "
        "from each example via long division, then apply it to the question."
    )
    lines.append("")

    ratio_strs: list[str] = []
    for a_str, b_str, a, b in pairs:
        if a == 0:
            continue
        a_disp_trunc = arith.truncate_3dp(a_str)
        b_disp_trunc = arith.truncate_3dp(b_str)
        a_disp, b_disp, _, _ = arith.cast_dp_pair(a_disp_trunc, b_disp_trunc)
        lines.append(f"{a_str} -> {b_str}")
        lines.append(
            f"Casting both to the same decimal places: {a_disp} -> {b_disp}"
        )
        lines.append(f"ratio = {b_str} / {a_str} = {b_disp} / {a_disp}")
        div_lines, r_str = arith.long_division_lines(b_disp, a_disp)
        lines.extend(div_lines)
        lines.append(f"ratio ≈ {r_str}")
        lines.append("")
        ratio_strs.append(r_str)

    if ratio_strs:
        lines.append(f"ratio values across examples: {', '.join(ratio_strs)}")
    lines.append(
        f"The ratios cluster tightly; using best back-test fit, ratio ≈ {best_ratio:.4f}."
    )
    lines.append("")

    # --- Apply to query via long multiplication ---
    r_disp = f"{best_ratio:.4f}".rstrip("0").rstrip(".")
    if not r_disp or r_disp == "-":
        r_disp = f"{best_ratio:.4f}"
    lines.append(f"Converting {q_str}: q_out = {q_str} * {r_disp}")
    mult_lines, mult_result = arith.long_multiplication_lines(q_str, r_disp)
    lines.extend(mult_lines)
    if not mult_lines or not mult_lines[-1].endswith(mult_result):
        lines.append(f"= {mult_result}")
    lines.append("")
    lines.append(f"Rounding to 2 decimal places: q_out ≈ {answer}.")

    reasoning = "\n".join(lines)
    return Solution(
        success=True, answer=answer, reasoning=reasoning,
        notes=f"ratio={best_ratio:.6f}, n_pairs={len(pairs)}, spread={rel_spread:.4f}",
    )
