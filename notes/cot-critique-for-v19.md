# CoT Reasoning-Process Critique — punch-list for v19 (after v18 scores)

> Created 2026-05-31. **Do NOT implement before v18 has a score** (decided: run v18 coverage-fix
> as a clean single variable first, then trace-quality as v19's single variable).
>
> **Originality guardrail** (`feedback_independent_methodology.md`): this is *critique of our own
> solver-authored traces* to guide OUR solver improvements — review, not distillation. We do NOT
> train on another model's generated CoT. Other models may CRITIQUE these traces; their critiques
> inform our solver edits. The traces stay solver-authored + 100% solver-verified (answer==gold).

## Why this matters (measured, not assumed)

v18 fixes COVERAGE gaps (transformation_rules 0%→covered; cipher rare-letter wordlist fallback).
It does NOT fix REASONING-PROCESS gaps — where a trace states a conclusion without showing the
derivation, training the model to skip the exact inference the hidden test rewards. Word-count is
NOT the signal (our 209-word cipher trace is efficient + complete; the 79-word bit-manip is the
problem). The signal is: **does the trace show the model HOW to find the rule, or just assert it?**

## Per-category critique (grounded in real traces + documented eval failures)

### bit_manipulation — HIGHEST PRIORITY (the clearest process gap)
- **Current trace:** "The transformation is `ROT(6)`." → then verifies → applies. **It never shows how
  ROT(6) was discovered.**
- **Documented failure (early diagnostic evals):** "Each version picks a different wrong rotation.
  Bit_manip remains a coin flip" — i.e. the model can't INFER the rule. Our trace trains rule-
  statement, not rule-search.
- **Fix:** solver should emit the search: "Test INV: example 1 out ≠ NOT(in) → reject. Test REV:
  → reject. Test ROT(k) for k=1..7: k=6 reproduces all 3 examples → accept." Show the elimination.
- **Note:** dgxchen's bit-manip traces are 2,038 words (vs our 79) and v18 USES dgxchen's for bit —
  so this fix is for OUR solver if/when we swap bit back to ours, OR to make a hybrid bit trace.
  Lower urgency than cipher/numeral since v18 already uses dgxchen's deep bit traces.

### cipher_phrase — MEDIUM (one shortcut to fix)
- **Current trace:** strong — explicit position alignment, full substitution table, wordlist
  fallback for unseen-letter words. **One shortcut:** "the consistent candidates are: book.
  Choosing 'book'." — when >1 candidate matches, it doesn't show the disambiguation.
- **Fix:** when the wordlist fallback has multiple candidates, show the tiebreak (which mapping
  constraint eliminates the others). When exactly 1, state "only 1 candidate matches pattern+length
  +known-mappings" explicitly so the model learns the *filter*, not the lookup.

### numeral_system — MEDIUM (genuinely thin: 18-word avg)
- **Current:** ~18 words — likely "N in base-X is Y" with no conversion shown. v18 uses dgxchen's
  numeral (390 words) so this is covered for v18, but OUR numeral solver is a v19 improvement target
  if we want full-original numeral coverage.
- **Fix:** show the division-remainder ladder for base conversion, or the subtractive steps for
  Roman numerals (e.g. 48 → XL (40) + VIII (8) → XLVIII), not just the answer.

### gravity / unit_conversion — LOW (already strong)
- 1,346 / 803 words, explicit arithmetic chains. Documented weakness = decimal-precision rounding
  (v4: predicted 57.03 vs gold 57.0). Minor: ensure trace states the gold's decimal-place convention
  explicitly before rounding. Not urgent.

### transformation_rules — LOW for process, but it's NEW in v18
- Our 506 traces show operator-inference ("`^` means addition because 78^42=120") + apply. Process
  is present. Just watch that the operator-elimination is shown when ambiguous.

## v19 plan (single variable on top of v18's score)
1. Improve the bit_manipulation + numeral solvers to emit derivation-process reasoning (above).
2. Regenerate ONLY those category traces (keep 100% solver-verified — answer==gold gate).
3. Swap them into the v18 corpus (single-variable: same records, richer reasoning on 2 categories).
4. Re-smoke (seq-len rises → re-check window), real run, submit, log EXPERIMENTS row.
5. Attribute: if v19 > v18, reasoning-PROCESS depth is a real lever; if flat, coverage was the
   whole story and we stop polishing traces.

## For external-model critique (optional, originality-safe)
Hand a model 3-5 of our traces per weak category + the documented failure mode, ask:
"Does this reasoning show HOW to derive the rule, or just assert it? Where would a model trained on
this fail to generalize?" Their answer informs OUR solver edits. Do NOT ask them to rewrite/generate
the traces.

---

## 2026-06-01 — transformation_rules coverage analysis (reviewer Q answered with data)

External reviewer claimed the coverage gap is "~15-16% q_op-not-in-examples". VERIFIED actual
breakdown of the 1,555 transformation_rules train problems:
- **506 (32.5%) solved + correct** → our corpus (the only ones trained on)
- 90 (5.8%) solved-but-wrong → dropped by answer==gold gate
- **823 (52.9%) SYMBOLIC** (answers like `@&`, `''#` — char permutations) → our numeric solver
  can't parse these. **This is the real coverage wall**, not the reviewer's claim.
- **136 (8.7%) q_op-not-in-examples** → the reviewer's claim; rejected at SEARCH (not parse fail).

**Reviewer's fix (length-prior + reversal-clustering heuristics for the 136) REJECTED** because
those cases are information-theoretically underdetermined — the query operator's rule appears
NOWHERE in the examples, so the answer can only be GUESSED, not solver-verified (reviewer's own
words: "an educated guess"). Training on unverified guesses violates our 100%-verified gate — the
exact failure mode behind v17's 0.58. Only ~1.4% of test even if perfect.

**The REAL opportunity = the 823 symbolic problems** (~9% of test, currently ZERO training signal
from anyone incl. dgxchen). These ARE verifiable (deterministic char-permutation rules) but need a
NEW symbolic solver. Decision: **finish v19 (trace-quality, the 506 verified) first; build the
symbolic transformation_rules solver as the NEXT single-variable run** (Branch B coverage expansion).
