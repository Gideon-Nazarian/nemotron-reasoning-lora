"""Solver for `gravity` problems — hybrid algorithm with long-arithmetic reasoning.

Prompt structure:
    In Alice's Wonderland, the gravitational constant has been secretly changed.
    Here are some example observations:
    For t = 1.04s, distance = 3.93 m
    For t = 2.21s, distance = 17.76 m
    ...
    Now, determine the falling distance for t = 4.19s given d = 0.5*g*t^2.

Algorithm:
  1. Parse (t, d) example pairs.
  2. Compute g_i = 2*d_i / t_i² for each example.
  3. From candidates [round(g_mean, 2), round(g_mean, 3), round(g_mean, 4), g_mean]
     pick whichever back-tests against the most example d values (rounded to 2dp).
  4. Apply d = 0.5 * g_best * t² to the query, round to 2dp.

Output format: 2 decimal places (matches gold's dominant format and the v3
baseline). v4 attempted to preserve per-example precision (e.g. '57.0' instead
of '57.00') and LB regressed -0.02; we hold the format invariant.

**Reasoning text upgrade** (the v3→v11 lever): the old solver emitted ~5 lines
of terse output. This version shows the long-multiplication trace for each t²,
long-division for each k_i = d_i/t_i² (where k = g/2), the sorted k list, and
the final long-multiplication for the query — per Tong's reasoners/gravity.py
style. Hypothesis (per the 2026-05-17 research findings): this format-rich
reasoning is what trains the model's arithmetic capability and is the missing
piece between our 0.62 LB and Tong's 0.85.

The algorithmic choice (back-tested mean) outperforms median-of-3dp-k on our
training corpus (1022 vs 870 / 1597 = 64% vs 54%), so we keep the algorithm
unchanged and only upgrade the trace.
"""

from __future__ import annotations

import re
import statistics

from . import _arithmetic as arith
from .types import Solution

PAIR_RE = re.compile(
    r"For\s+t\s*=\s*(-?\d+(?:\.\d+)?)\s*s\s*,\s*distance\s*=\s*(-?\d+(?:\.\d+)?)\s*m",
    re.IGNORECASE,
)
QUESTION_RE = re.compile(
    r"falling distance for\s+t\s*=\s*(-?\d+(?:\.\d+)?)\s*s",
    re.IGNORECASE,
)


def solve(prompt: str) -> Solution:
    pair_matches = PAIR_RE.findall(prompt)
    if len(pair_matches) < 2:
        return Solution(False, "", "", notes=f"only {len(pair_matches)} pairs parsed")
    pairs = [(t_str, d_str, float(t_str), float(d_str)) for t_str, d_str in pair_matches]

    q_match = QUESTION_RE.search(prompt)
    if not q_match:
        return Solution(False, "", "", notes="no question parsed")
    q_t_str = q_match.group(1)
    q_t = float(q_t_str)

    # --- Algorithmic core (unchanged from v3): back-tested mean g ---
    gs = [2 * d / (t * t) for _, _, t, d in pairs if t != 0]
    if not gs:
        return Solution(False, "", "", notes="no valid (t>0) pairs")

    g_mean = statistics.mean(gs)
    spread = (statistics.pstdev(gs) / abs(g_mean)) if g_mean != 0 and len(gs) > 1 else 0.0
    if spread > 0.02:
        return Solution(False, "", "", notes=f"g spread too large: {spread:.4f}")

    def back_test(g_val: float) -> int:
        return sum(1 for _, _, t, d in pairs if abs(round(0.5 * g_val * t * t, 2) - d) < 1e-9)

    candidates = [round(g_mean, 2), round(g_mean, 3), round(g_mean, 4), g_mean]
    best_score = max(back_test(g) for g in candidates)
    best_g = next(g for g in candidates if back_test(g) == best_score)

    q_t_sq = q_t * q_t
    q_d = 0.5 * best_g * q_t_sq
    answer = f"{q_d:.2f}"

    # --- Long-arithmetic reasoning trace (the new training signal) ---
    lines: list[str] = []
    lines.append(
        "We need to determine the falling distance using d = 0.5 * g * t². "
        "Let me solve for g from each example."
    )
    lines.append("")

    g_strs: list[str] = []
    for t_str, d_str, t, d in pairs:
        if t == 0:
            continue
        # t² via long multiplication
        sq_lines, t_sq_str = arith.long_multiplication_lines(t_str, t_str)
        lines.append(f"t = {t_str}s, d = {d_str}m:")
        lines.append(f"t² = {t_str} * {t_str}:")
        lines.extend(sq_lines)
        if not sq_lines or not sq_lines[-1].endswith(t_sq_str):
            lines.append(f"= {t_sq_str}")

        # 2*d for the numerator of g = 2d/t²
        two_d_lines, two_d_str = arith.long_multiplication_lines(d_str, "2")
        lines.append(f"2 * d = 2 * {d_str}:")
        lines.extend(two_d_lines)
        if not two_d_lines or not two_d_lines[-1].endswith(two_d_str):
            lines.append(f"= {two_d_str}")

        # g_i = 2*d / t² via long division (3dp truncated)
        twod_disp, tsq_disp, _, _ = arith.cast_dp_pair(
            two_d_str, arith.truncate_3dp(t_sq_str)
        )
        lines.append(f"g = 2d / t² = {two_d_str} / {t_sq_str} = {twod_disp} / {tsq_disp}")
        div_lines, g_str = arith.long_division_lines(twod_disp, tsq_disp)
        lines.extend(div_lines)
        lines.append(f"g ≈ {g_str}")
        lines.append("")
        g_strs.append(g_str)

    # Show all g values + the chosen one
    if g_strs:
        lines.append(f"g values across examples: {', '.join(g_strs)}")
    lines.append(f"The g values cluster tightly; I use g = {best_g:.4f} (best back-test fit).")
    lines.append("")

    # --- Query: q_d = 0.5 * g * t² ---
    lines.append(f"For t = {q_t_str}s:")
    lines.append(f"t² = {q_t_str} * {q_t_str}:")
    q_sq_lines, q_t_sq_str = arith.long_multiplication_lines(q_t_str, q_t_str)
    lines.extend(q_sq_lines)
    if not q_sq_lines or not q_sq_lines[-1].endswith(q_t_sq_str):
        lines.append(f"= {q_t_sq_str}")
    lines.append("")

    # g * t² via long multiplication (using g rounded to display nicely)
    g_disp = f"{best_g:.4f}".rstrip("0").rstrip(".")
    if not g_disp or g_disp == "-":
        g_disp = f"{best_g:.4f}"
    lines.append(f"g * t² = {g_disp} * {q_t_sq_str}:")
    gxt_lines, gxt_str = arith.long_multiplication_lines(g_disp, q_t_sq_str)
    lines.extend(gxt_lines)
    if not gxt_lines or not gxt_lines[-1].endswith(gxt_str):
        lines.append(f"= {gxt_str}")
    lines.append("")

    # 0.5 * (g * t²) — this is just halving; we skip the full long-division
    # trace (it iterates over every decimal place of the product, generating
    # ~50 lines of repetitive output). One-line halving keeps the trace tight.
    gxt_float = best_g * q_t_sq
    half_float = gxt_float / 2
    lines.append(f"d = 0.5 * (g * t²) = {gxt_str} / 2 ≈ {half_float:.4f}")
    lines.append(f"Rounding to 2 decimal places: d ≈ {answer}.")

    reasoning = "\n".join(lines)
    return Solution(
        success=True, answer=answer, reasoning=reasoning,
        notes=f"g={best_g:.6f}, spread={spread:.4f}, n_pairs={len(pairs)}",
    )
