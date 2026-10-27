# Gravity solver: 64% → 85.9% coverage (banked for v20-conditional)

> 2026-06-01. Found while auditing v19 category coverage. **Banked, NOT in v19** —
> v19 uses dgxchen's gravity traces. This matters only if a future corpus swaps OUR
> procedural gravity in (gated on v19's LB proving the procedural-trace thesis).

## The finding

The gravity solver was never a reasoning failure — it solves all 1,597 problems and computes
`g` correctly (spreads ~0.0001). The "36% gap" (575 misses) was entirely **g-selection + output
rounding**:
- pure format (45.00 vs 45.0, same value): 114
- tiny rounding (≤0.02 off): 416
- genuinely wrong (>0.02): 45

## Reverse-engineered gold generator's g-strategy

Backing out `g_implied = 2*gold/qt²` and matching against example-derived g's:
- **~74% raw mean** of all example g's (current solver rounds first → misses these)
- ~8% raw median
- ~8% a single example's g
- small fractions: rounded mean (r2/r3/r4), trimmed mean (drop hi+lo)

## The fix (validated, not yet implemented in solver)

Build candidate list `[mean, median, trimmed_mean, round(mean,2/3/4), round(med,3), *example_gs]`,
score each by how many EXAMPLE outputs it reproduces at the gold's per-example decimal precision,
pick the best back-tester, output at inferred precision (gold uses 1dp for 157, 2dp for 1440).

**Result: 64.0% → 85.9% (+350 recovered).** Remaining ~14% (225) are ±0.01-0.02 final-rounding-order
mismatches needing the gold generator's exact float pipeline — diminishing returns, left alone.

## Decision

BANK it. Don't touch v19's staged corpus (uses dgxchen gravity). If v19 > v18 (procedural-trace
thesis proves out), implement this improved g-selection + procedural gravity traces for v20. If v19
flat, gravity is moot (pivot to RLVR). The validated algorithm is in this note's git history /
reconstructable from the candidate-list + back-test description above.
