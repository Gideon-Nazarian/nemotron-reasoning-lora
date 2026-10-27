# Nemotron Challenge — v19 Decision Writeup (for outside opinions)

> Self-contained. Written 2026-06-01. Deadline 2026-06-15. ~2 weeks compute left.
> Goal of sharing: get outside opinions on **whether/how to run "v19"** (details below).

---

## 1. The competition (context)

- **NVIDIA Nemotron Reasoning Challenge** (Kaggle). Fine-tune **Nemotron-3-Nano-30B-A3B-BF16**
  (Mamba-Transformer hybrid, 128 MoE experts) with a **LoRA adapter**. Submit the adapter; the
  grader loads it onto the base model and scores hidden reasoning problems.
- **5 problem categories**, roughly even in the test set:
  | category | ~% of test | example |
  |---|---|---|
  | numeric_decimal (gravity + unit conversion) | 33.6% | "g secretly changed; find falling distance" |
  | bit_manipulation | 16.9% | "8-bit rule: 01101100 -> 10111101, find output for X" |
  | cipher_phrase | 16.6% | substitution cipher -> decode phrase |
  | roman_numeral | 16.6% | "secret numeral system" (always standard Roman) |
  | transformation_rules | 16.4% | operator redefinition: `78^42=120` means `^`=add |
- **Public LB ceiling ≈ 0.87.** The top public recipes (Kh0a 0.87, kuangyicheng "0.88") depend on
  **private data files** and do NOT reproduce — we tried (see §3).

## 2. Our approach + the verified floor

We build **solver-generated, 100%-verified chain-of-thought (CoT)** training data: Python solvers
that actually solve each problem, emit a reasoning trace, and we **only keep traces where
answer == gold** (a hard verification gate). This is our originality/methodology edge vs the
private-data teams.

- **Proven floor: 0.84** — reproduced 3 independent ways (dgxchen's public adapter submitted
  directly; our v13 = dgxchen recipe; our v18 = dgxchen + our verified data).

### 2a. Two trace sources we mix-and-match per category

Throughout, every corpus is assembled by choosing, **per category**, between two trace sources:

- **dgxchen** — a public competitor's trained-adapter dataset (`problem_ids_matched.csv`, 7,830
  records). His traces are **LLM-generated** long CoT (mean ~1,156 words), labeled with the **gold**
  answer (so 100% correct *label* by construction; his reasoning is not independently verified).
  His coverage = how many records he sampled per category. He has **no symbolic-transformation and
  no roman/numeral_system** records (gaps we fill).
- **OURS** — our Python solvers (`scripts/solvers/*.py`) that actually solve each problem and emit a
  trace, kept **only when answer == gold** (hard verification gate). 100% verified-correct.

### 2b. Solver benchmarks (our solvers, measured on train.csv, current)

| solver file | category | total | solved | **verified-correct (coverage)** | known weakness |
|---|---|---|---|---|---|
| `roman_numeral.py` | numeral/roman | 1576 | 1576 | **1576 (100%)** | none (always standard Roman) |
| `cipher.py` | cipher | 1576 | 1576 | **1576 (100%)** | relies on Wonderland wordlist for unseen-letter words |
| `bit_manipulation.py` | bit_manip | 1602 | 1447 | **1428 (89%)** | misses choice/majority-function rules (not in grammar) |
| `unit_conversion.py` | unit | 1594 | 1594 | **1328 (83%)** | edge-case unit chains |
| `gravity.py` | gravity | 1597 | 1597 | **1022 (64%)** | g-selection + output rounding (NOT reasoning — fix found→86%, banked) |
| `transformation_rules.py` | transformation | 1555 | 596 | **506 (33%)** | only numeric subset; **53% is SYMBOLIC** (char-perm) = no solver; +9% q_op-not-in-examples (unverifiable) |

Coverage-evolution history (from the git log / ledger; shows where each solver started vs now):
- bit_manipulation: **9.4% (v1/v2/v3 early) → 89.1%** (v3 "full shift composer" grammar)
- cipher: **38.4% → 100%** (added Wonderland-wordlist fallback for unseen letters)
- transformation_rules: **0% → 32.5%** (new numeric-subset solver; symbolic still 0)
- gravity: **64%** (long-arithmetic trace added for quality; coverage limited by rounding-match)
- numeral/roman: **100%** from the start
- unit_conversion: **83%** (long-arithmetic trace added)

**v19 also has _v19 procedural variants** of 3 solvers (`bit_manipulation_v19.py`,
`roman_numeral_v19.py`, `transformation_rules_v19.py`) — same answers, reasoning rewritten to show
the search (see §4). Same coverage as their base solvers (text rewrite, not a coverage change).

### 2c. EXACTLY which source each run used, per category

| category | v13 (0.84) | v17 (0.58) | **v18 (0.84)** | **v19 (proposed)** |
|---|---|---|---|---|
| bit_manipulation | dgxchen | synthetic* | **dgxchen (1754, 100%)** | **OURS-v19 (1428, 89%)** |
| cipher | dgxchen | synthetic* | **dgxchen (1656)** | dgxchen (1656) |
| numeral/roman | dgxchen | synthetic* | **dgxchen (730) + OURS (926)** | **OURS-v19 (1576)** |
| gravity | dgxchen | synthetic* | **dgxchen (1055, 100% label)** | dgxchen (1055) |
| unit_conversion | dgxchen | synthetic* | **dgxchen (1070)** | dgxchen (1070) |
| transformation | none | synthetic* | **OURS (506, 33%)** | **OURS-v19 (506)** |
| cryptarithm/equation | dgxchen | — | **dgxchen (1565)** | dgxchen (1565) |

*v17 used kuangyicheng's reasoning-FREE inline-generated data (16-word boilerplate + bare answers +
1-line synthetic) for ALL categories — NOT our solvers. That's why it scored 0.58 (see §3).

**v18 in detail (the 0.84 corpus, 10,311 records):** dgxchen's full 7,830 base (his bit_manip,
cipher, gravity, unit, numeral, cryptarithm, equation — kept because his were deeper/100%-label) +
our 506 transformation_rules (dgxchen has ZERO) + 1,975 "ours-only" problems (verified traces for
problem_ids dgxchen didn't sample). v18's weakness at the time: **it used dgxchen's "oracle"-style
bit_manip/gravity traces** (state-the-rule), and our additions were coverage-only — no reasoning-
*process* improvement. v18 result confirmed coverage is inert (§3).

## 3. What we've already tested (the key constraint on what's worth running)

| version | what changed | LB | lesson |
|---|---|---|---|
| v13 | dgxchen recipe + boxed-loss | 0.84 | boxed-loss inert |
| v12 | Kh0a's public 0.87 recipe (private data missing) | 0.72 | private data = the gap |
| v17 | kuangyicheng's "0.88" recipe (reasoning-FREE boilerplate data) | **0.58** | hollow data HURTS |
| v14 | dgxchen + format augmenters (mixed 2 trace styles) | **0.68** | **dual-template mixing = -0.16** |
| v15 | dgxchen + lm_head in LoRA targets | **0.52** | lm_head catastrophic |
| **v18** | **dgxchen + our verified additive data (near-MAX coverage)** | **0.84** | **coverage is INERT (+0.00)** |

**Two hard-won conclusions:**
1. **Coverage is not the lever.** v18 maximized verified-data coverage → +0.00. A "maximize
   coverage" corpus now ≈ v18 ≈ almost certainly another 0.84.
2. **Mixing two trace STYLES in one corpus is dangerous** (v14: −0.16). The model learns to emit
   the wrong format and the grader can't extract an answer.

**The diagnostic smoking gun:** v18's final training loss hit **0.50** (very low) yet LB stayed flat
0.84 → the model **MEMORIZES** the training traces but doesn't **GENERALIZE** the reasoning to hidden
test problems.

## 4. The v19 hypothesis (the proposed next run)

**The one SFT lever we have NOT tested: trace STYLE / reasoning quality.**

Our current traces (and dgxchen's) are mostly "oracle" style — they **state the answer rule** then
verify. Example (bit_manipulation, current):
> "The transformation is `SHL(1) XOR NOT SHL(7)`." → verifies on 2 examples → applies.

The model never sees **how** that rule was *found*. The documented bit_manip failure mode (from our
eval logs): *"the model picks a different wrong rotation each time — bit_manip is a coin flip."* It
can't DERIVE rules, only recall stated ones.

**v19 rewrites the traces to show the SEARCH the solver actually does** ("procedural" style):
> "Test SHL(1): 01010001->10100010, ≠ target, reject. Test NOT SHL(1): ... reject. [Step 2] Two-op
> combos: SHL(1) XOR NOT SHL(7), verify bit-by-bit in a vertical column table ✓. Rule found."

This was reviewed by an external model twice and hardened (explicit failed tests; vertical bit-math
that matches dgxchen's 0.84 traces; grounded hypotheses; no style-mixing). **Same problems, same
answers — only the reasoning text changes.**

**v19 covers 3 "weak" categories with new procedural traces:** bit_manipulation, numeral_system,
transformation_rules. Keeps dgxchen's traces for cipher/gravity/unit (those already work).

## 5. Per-category coverage — OURS vs dgxchen (% of category we verify-cover)

(Full solver benchmarks + per-run source history are in §2b/2c. This is the at-a-glance
decision table — note where v19's "who uses" is NOT the higher-coverage source: bit_manip
uses OURS 89% over dgxchen 100%, and unit keeps dgxchen 67% over OURS 83%.)

| category | total | OUR solver | dgxchen | v19 currently uses |
|---|---|---|---|---|
| bit_manipulation | 1602 | **89%** | **100%** | OURS (procedural) |
| cipher | 1576 | 100% | 100% | dgxchen |
| numeral/roman | 1576 | **100%** | 46% | OURS (procedural) |
| gravity | 1597 | 64% | 66% | dgxchen |
| unit_conversion | 1594 | **83%** | 67% | dgxchen |
| transformation* | 1555 | 33% | (different subset) | OURS (procedural) |

*transformation: ours = operator-redefinition numeric subset (506 verified); 53% of the category is
SYMBOLIC (char-permutation, e.g. `@&`) which NEITHER we nor dgxchen solve (an open coverage gap, ~9%
of test, no training signal from anyone — a separate future "symbolic solver" project).

## 6. THE DECISION / TENSION (what we want opinions on)

v19 is **a trace-QUALITY experiment, not a coverage-max corpus.** It deliberately:
- uses OUR bit_manip (89%) instead of dgxchen's (100%) — to get uniform procedural style →
  **trades 11% bit_manip coverage for the procedural-style test.** (This is the main confound.)
- keeps dgxchen's unit (67%) even though ours is higher (83%) — because unit isn't a "weak" category
  we built procedural traces for.

**The core questions:**

**A. Is v19 worth running at all?**
- FOR: it's the ONE untested SFT lever. Coverage is proven inert (v18=0.84). Recipe-replication
  fails (private data). Trace-style is the last unknown. Even a FLAT v19 is informative — it would
  prove trace quality isn't the lever and justify pivoting to RLVR (reinforcement learning with our
  solvers as verifiers) with confidence rather than guessing.
- AGAINST: v18's 0.50-loss-but-flat-LB suggests the model memorizes regardless of trace style, so
  v19 may also be 0.84. And we can't verify the payoff locally — only the LB (~8h GPU run) tells us.

**B. If we run it, how to handle the bit_manip confound?**
1. **Keep ours (89%), accept the small confound.** Cleanest single-variable on the other 2
   categories; if flat, re-examine bit_manip. (Current build.)
2. **Hybrid bit_manip:** our procedural traces for the 1428 we cover + dgxchen's for the ~174 we
   miss → 100% coverage AND mostly procedural. BUT mixes two trace styles within one category =
   the v14 dual-template risk (−0.16) that we KNOW is dangerous.
3. **Don't touch bit_manip** — keep dgxchen's 100%. v19 then tests procedural style only on numeral
   (100% ours) + transformation. Zero coverage loss, but tests the hypothesis on smaller categories
   (bit_manip was the biggest/highest-value target — 16.9% of test + the documented coin-flip
   failure procedural reasoning directly targets).

**C. Or skip v19 entirely and pivot to RLVR now?** The memorization signal (loss 0.50, flat LB) is
the classic motivation for RL: optimize the actual generalization objective (reward correct outputs)
instead of imitating traces. Higher complexity, ~2 weeks, but attacks the root cause. Risk: we'd
skip the cheap experiment (v19, ~8h) that would TELL us if trace quality matters before committing
to the expensive pivot.

## 7. Constraints
- ~2 weeks compute left. Kaggle weekly GPU quota; each real run ~8h on RTX Pro 6000.
- Standing rules: every run must be "a point of progress"; single-variable changes for clean
  attribution; 100%-solver-verified data only (no training on guesses/unverified traces — that's
  what sank v17 and is our methodology edge).
- v19 is fully built + staged (corpus uploaded, notebook validated). Running it = push + ~8h.

## 8. The actual question for an outside reader
Given coverage is proven inert and recipe-replication fails: **is the procedural-trace-style test
(v19) worth one ~8h run, and if so which bit_manip option (B1/B2/B3)? Or pivot straight to RLVR (C)?**
