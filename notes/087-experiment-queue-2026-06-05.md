# 0.87 experiment queue — built + status (2026-06-05)

All three are v18's IDENTICAL recipe (Unsloth SFT, same hyperparams), varying ONLY the corpus → clean
single-variable comparison. Masking is a phase-2 lever for whichever data wins. Each has a SMOKE variant
(or flip SMOKE_TEST=True in the editor). Run in priority order; let each result redirect the next.

| Plan | Notebook | Corpus (dataset) | The bet | Pushed? |
|---|---|---|---|---|
| #1 | v22-logprob-filtered | nemotron-v18-corpus-filtered075 (8,090) | DROP the 4 high-loss categories | ✅ LIVE |
| #3 | v23-tong | nemotron-tong-corpus (17,963) | Tong's EXACT winning data (his taxonomy) | ✅ LIVE |
| #2 | v24-tahaalam | nemotron-tahaalam-filtered (8,703) | KEEP cryptarithm (loss-capped ~2x), drop transformation | ⏳ built, blocked by 2-GPU-session cap |

Smokes: v22-smoke ✅ LIVE. v23-tong-smoke / v24-tahaalam-smoke built, same cap (or flip SMOKE_TEST in editor).
Datasets all uploaded + ready. Reformat scripts: scripts/build_tong_corpus.py, scripts/build_tahaalam_corpus.py.

## Corpora — verified (agents checked counts/format/samples)
- tong-corpus.jsonl: 17,963 (9,500 reasoning w/ \boxed + 8,463 augmentation NO \boxed). Answer re-extracted
  from inline \boxed per his corpus.py (differs from train.csv on 46%). His taxonomy: matching 4515,
  bit_manip 1602, gravity 1597, etc. NO transformation_rules/numeral_system (his categories differ).
- tahaalam-filtered-corpus.jsonl: 8,703. Fixed a double-</think> bug. KEEPS cryptarithm (1,539, ~2x v18),
  transformation_rules ABSENT. base_loss MATCHES our scoring (independent confirmation our scoring is sound).

## Run plan
1. v22 + v23-tong are LIVE → run on RTX (attach competition → RTX → Run All; ~7-8h each). Submit, compare to 0.84.
2. v24-tahaalam pushes once a GPU slot frees (CLI push auto-runs → needs a free session; cap is 2 concurrent).
3. NONE cover transformation_rules (~16% of test) — consistent with the thesis (highest-loss) but a known coverage risk.
