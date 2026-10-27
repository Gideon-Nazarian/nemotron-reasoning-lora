"""Shared types for solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Solution:
    """The output of a programmatic solver.

    Attributes:
        success: True if the solver was confident in its answer.
        answer: The computed answer string (matches the gold-answer format expected
            for this problem category).
        reasoning: A short chain-of-thought text describing how the answer was
            computed. This is the SFT training signal.
        notes: Optional debugging info (parsed values, intermediate computations).
    """
    success: bool
    answer: str
    reasoning: str
    notes: Optional[str] = None


@dataclass
class Problem:
    """A single problem from train.csv."""
    id: str
    prompt: str
    gold: Optional[str] = None  # may be None for test problems
