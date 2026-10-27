"""Reward function for GRPO training on the Nemotron Reasoning Challenge.

Used as a `reward_funcs` argument to `trl.GRPOTrainer`. The trainer passes us
the model's generated completions plus any custom columns from the dataset
(we use `gold_answer` to know what the correct boxed answer should be).

Reward design:
  - **1.0** if `\\boxed{X}` and X.strip() == gold.strip() (exact match — what
    the grader rewards)
  - **0.3** if `\\boxed{X}` and X numerically equals gold (e.g. "57.00" vs
    "57.0") — partial credit for "right value, format mismatch"
  - **0.1** if gold appears as a substring anywhere in the completion (even
    outside `\\boxed{}`) — minimal credit for "model reached the right answer
    but didn't box it"
  - **0.0** otherwise

The tiered structure gives GRPO a non-binary signal even when the model isn't
perfectly aligned to the boxed-format yet. Without this, early training where
the model rarely produces the exact boxed answer would give all-zero rewards,
leaving GRPO with no gradient.

Crucially we do NOT give partial credit for sub-string overlap of numbers
(e.g. `57.0` ⊂ `57.03` — these are DIFFERENT numbers, not a format gap).
We use numeric equality for that case instead.

Compatible with TRL's GRPOTrainer signature (kwargs allow forward-compat with
new fields the trainer might add).
"""

from __future__ import annotations

import re
from typing import Any

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def extract_boxed(completion: str) -> str:
    """Return the last \\boxed{X} content, or '' if no boxed answer."""
    matches = list(BOXED_RE.finditer(completion))
    return matches[-1].group(1).strip() if matches else ""


def _numerically_equal(a: str, b: str) -> bool:
    """True if both strings parse as floats and have the same value.

    Catches format gaps like "57.0" vs "57.00" without falsely matching
    "57.0" vs "57.03" (substring would match; numeric equality won't).
    """
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return False


def _gold_in_completion(completion: str, gold: str) -> bool:
    """True if gold appears as a substring of completion with digit boundaries.

    Avoids false positives where a shorter numeric gold (e.g. "57.0") matches
    a prefix of a longer number (e.g. "57.03"). Non-numeric golds (ciphers,
    roman numerals) match anywhere as long as adjacent chars aren't digits.
    """
    pattern = r"(?<!\d)" + re.escape(gold) + r"(?!\d)"
    return bool(re.search(pattern, completion))


def score_completion(completion: str, gold: str) -> float:
    """Score a single (completion, gold) pair. See module docstring for tiers."""
    if not gold:
        return 0.0
    gold = gold.strip()
    boxed = extract_boxed(completion)

    # Tier 1: exact string match in \boxed{}
    if boxed and boxed == gold:
        return 1.0

    # Tier 2: \boxed{X} where X is numerically equal to gold (format mismatch only)
    if boxed and _numerically_equal(boxed, gold):
        return 0.3

    # Tier 3: gold appears somewhere in the completion (even outside \boxed{})
    if _gold_in_completion(completion, gold):
        return 0.1

    return 0.0


def boxed_match_reward(
    completions: list[str],
    gold_answer: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Reward function for TRL's GRPOTrainer.

    Args:
        completions: list of generated completions (one per rollout).
        gold_answer: list of gold answers, parallel to completions. Comes from
            the dataset's `gold_answer` column (TRL passes dataset columns as
            kwargs).
        **kwargs: forward-compat for prompts, completion_ids, trainer_state, etc.

    Returns:
        list[float] of rewards in [0.0, 1.0], one per completion.
    """
    if gold_answer is None:
        raise ValueError(
            "Dataset must have a 'gold_answer' column; TRL passes it as a kwarg."
        )
    if len(completions) != len(gold_answer):
        raise ValueError(
            f"Length mismatch: {len(completions)} completions vs "
            f"{len(gold_answer)} gold answers"
        )
    return [score_completion(c, g) for c, g in zip(completions, gold_answer)]
