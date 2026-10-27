# Logprob scoring result + Plan #1 experiment (2026-06-05)

## Scoring run (kaggle/logprob-score) — VERIFIED clean
Base Nemotron-30B teacher-forced loss over all 10,311 v18-corpus traces. RTX, sanity gate passed,
65 min, no errors. Output: results/base_loss-v18corpus-2026-06-05.csv (per-trace mean_loss + max_loss).

## THE result: sharply BIMODAL with a natural valley at ~0.75
Overall mean_loss: mean 0.555, median 0.446. Histogram trough at 0.7–0.8 (only 78 traces) cleanly
separates two clusters. Per-category (small within-cat std → each category is wholly learnable or not):
- LEARNABLE (≤0.5): gravity 0.22, equation_numeric_deduce 0.24, equation_numeric_guess 0.26,
  unit_conversion 0.29, numeral 0.34, bit_manipulation 0.49, cipher 0.49.
- UNLEARNABLE (≥1.0): cryptarithm_guess 0.99, cryptarithm_deduce 1.02, numeral_system 1.26,
  transformation_rules 1.62 (3× the median).
⇒ CONFIRMS the doc's logprob thesis ON OUR DATA, and explains v19 (0.72) / v21 (0.76): the categories
that sank those runs (transformation_rules, cryptarithm) are exactly the high-base-loss "unlearnable" ones.

## Filter built (0.75 cut)
data/v18-corpus-filtered075.jsonl: 8,090 records (dropped 2,221). Dropped ENTIRELY: numeral_system (926),
cryptarithm_deduce (627), transformation_rules (506), cryptarithm_guess (154) + 8 high-loss tails of
bit/cipher/unit. Kept the 7 learnable categories intact. Uploaded as gideonnazarian/nemotron-v18-corpus-filtered075.
Regen: `python3 - ` filter base_loss.csv mean_loss<0.75 against data/v18-corpus.jsonl line order (idx).

## Queued experiment: kaggle/v22-logprob-filtered (Plan #1, filter-ONLY)
v18's EXACT recipe (Unsloth SFT, same hyperparams) with ONLY the corpus swapped to the filtered set →
the filter is the sole variable. Built + pushed; dataset attached; competition tagged (RTX). SMOKE off (real
~7-8h run). RUN AFTER RESET (does not fit a partial window). Submit → compare to v18=0.84:
- filtered > 0.84 → forgetting-prevention wins (thesis holds); next: recover dropped categories via
  learnable rewrites + add masking.
- filtered < 0.84 → coverage loss dominates for us (v21 confirmed); filter-alone insufficient.
- filtered = 0.84 → neutral; the dropped categories were contributing ~nothing.
HONEST: drops whole test categories (the v21 risk) BUT keeps the full learnable corpus (8,090 vs v21's
3,910) + is a clean single-variable test. ~50/50, but it's THE published lever and never tried.
