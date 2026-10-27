# Coverage gap analysis: v18 (0.84) vs Tong (0.86) vs the "symbol" category (2026-06-02)

> Triggered by the question: "do dgxchen 0.84 / Tong 0.86 actually solve the symbol-transformer
> group, or is it the last unsolved group?" The right instinct — and it corrected a stale-notes error.

## Finding 1 — the SYMBOL group is NOT the unsolved gap (corrects earlier claim)
- `transformation_rules` = 1,555 train.csv problems (916 symbol-answer + 639 numeric).
- **dgxchen (0.84):** 888 symbol-transformation traces, **100% correct** (888/888 == gold).
- **Tong (0.86):** covers **ALL 1,555** transformation_rules incl. **ALL 916 symbol** (his categories
  `cryptarithm_deduce`/`equation_numeric_deduce`/`*_guess`).
- ⇒ Both frontier corpora FULLY + correctly cover symbol-transformation. **Not a blind spot.**
- My earlier "+0.05–0.15 LB / biggest lever" was from STALE notes (the `~0.2%` was old v7/v8 solver-only
  models with ZERO symbol coverage — NOT v18, which inherits dgxchen's 888 correct symbol traces).
  **Lesson: re-verify per-category claims against the CURRENT corpus before investing.**

## Finding 2 — the REAL coverage gap: Tong trains on 100% of train.csv, v18 on 86%
- train.csv = 9,500. **Tong covers 9,500 (100%). v18 covers 8,161 (86%).**
- The **1,339 problems v18 NEVER trained on, Tong covers ALL of them.** By Tong's category:

  | Tong category | count in the v18 gap |
  |---|---|
  | cryptarithm_deduce | 605 |
  | gravity | 206 |
  | cryptarithm_guess | 153 |
  | bit_manipulation | 129 |
  | equation_numeric_guess | 115 |
  | unit_conversion | 90 |
  | equation_numeric_deduce | 41 |

- So **~914 of the 1,339 gap = cryptarithm/equation (transformation-family)** + gravity/bit/unit residue.
- v18 = dgxchen + our solvers, so v18 coverage ≥ dgxchen. The 1,339 are problems NEITHER dgxchen NOR our
  solvers could generate traces for — the **hard residue** Tong's method cracked and we/dgxchen didn't.

## What this means (honest)
- The lever is NOT "crack the symbol rule" (done by everyone). It's **closing the 1,339-problem coverage
  gap** — esp. cryptarithm/transformation. We HAVE gold answers for all 1,339 (train.csv).
- **BUT coverage ≠ score.** Our 5-run finding: v18 = KEEP dgxchen + ADD our solver traces = **0.84 (additive,
  neutral)**. Adding our-voice traces didn't lift. Tong's 0.86 edge likely = his TRACE QUALITY/VOICE on the
  residue, not just coverage. So naively generating our-voice traces for the 1,339 may NOT replicate 0.86.
- The genuinely hard part: the 1,339 are exactly the problems our solvers/dgxchen COULDN'T solve. Generating
  CORRECT traces for them needs either better solvers (cryptarithm/transformation) or a different method.

## The "post-train analysis" — YES, valuable; two halves
- **Offline (coverage) — DONE here.** Tells us TRAINING gaps (the 1,339 + per-category). Already actionable.
- **GPU (per-category ACCURACY) — the complement, not yet done.** Tells us where the MODEL FAILS at test
  time even on covered categories. The held-out eval (`eval_heldout_lib`) is a start but its held-out is the
  skewed 1,339; a fuller per-category eval across ALL categories would pinpoint weak spots to improve.

## Candidate next moves (for discussion)
1. **A1 masking** (queued) — independent, cheap; proceed.
2. **Per-category accuracy eval of v18** — find where the model actually fails (the real "where do we lack").
3. **Close the coverage gap** — generate correct traces for the 1,339 (need better cryptarithm/transformation
   solvers; we have gold answers). Caveat: coverage-alone was neutral before; may need Tong-quality traces.
4. **Bank 0.84** — the gap to 0.86 is small and the residue is hard.

## Finding 3 — Tong PUBLISHED his weakness map, and the weak spot = cryptarithm_deduce
The question was: did the 0.86 author publish coverage/weakness, or can we pick it up to find+fix?
**Answer: YES, extensively published** (we have the repo `reference/tonghuikang-nemotron/`):
- `investigations/<problem_id>.txt` — per-problem CRACKED RULE + examples + predicted answer (1,667 files).
- `investigators/calc_accuracy.py` + `build_generation_index.verify` — his per-category accuracy infra.
- `problems.jsonl` — his id→category taxonomy. (His prediction CSVs v20/v26.csv are NOT in the repo.)
- Every corpus record uses loss masking (`masked_token_count`>0 on all 17,963) → **corroborates A1 masking is in the winning recipe.**

**His published weak spot:** of 1,667 investigations, **97 have no fully-cracked rule — 95 are `cryptarithm_deduce`.**
That is the SAME category dominating the v18 coverage gap (605/1,339). So cryptarithm_deduce is where even
the 0.86 frontier is weak/uncertain.

**What the hard ones look like (e.g. 9aa8dc92):** symbol-operand operator-redefinition (`+`=concat) PLUS an
underdetermined per-symbol substitution — with only 3-4 examples the rule is ambiguous, which is why Tong
could only best-guess. The operator part is the same family I hand-cracked earlier; the symbol-substitution
part is the genuinely hard, possibly-underdetermined piece.

**Find-and-fix path (reference-reading Tong's weak map is fair; our solver stays original):**
- TARGET = cryptarithm_deduce / symbol-operator transformation problems (~659 + the equation/guess siblings).
- We have gold answers (train.csv) for ALL of them. Extend our transformation_rules solver to symbol operands
  + operator-grouped candidate search; for the underdetermined ones, gold-answer-gate (only emit a trace when
  a candidate rule reproduces all examples AND matches gold).
- Caveats (hard-won): (a) HARD — Tong is excellent and still marked 95 unknown; (b) some are underdetermined
  (multiple rules fit the examples) → unsolvable from examples alone; (c) coverage≠score (voice-mismatch).
- Upside: it IS the frontier weak spot → cracking the residue = genuine leapfrog-past-0.86 potential.
