# Early diagnostic evals (May 2026): aggregate results only

These are small local evals run on Kaggle in the first week, before the leaderboard-driven phase. They used a handful of held-out problems from the competition's `train.csv` (n = 10 for the first, n = 5 for the rest). The per-problem write-ups quoted competition prompts and gold answers, so only the aggregates are kept in this public copy.

| Eval | Adapter | n | Strict `\boxed{}` exact match | Lenient substring match | Avg completion tokens | Notes |
|---|---|---|---|---|---|---|
| v3 local eval (2026-05-09) | untrained (= base model) | 10 | 0/10 | not recorded | 814 (1,024 cap) | 9 of 10 hit the 1,024-token cap without a `\boxed{}` answer. Two stopped at 16 tokens |
| v5 | untrained | 5 | 0/5 | 1/5 | 1,024 (all capped) | none closed their thinking block |
| v6 | SFT v1 (750 records, template-style reasoning) | 5 | 1/5 | 1/5 | 123 | format learned (5/5 closed thinking), but reasoning was template-overfit and values were fabricated |
| v7 | SFT v2 (same records, explicit chain-of-thought) | 5 | 2/5 | 3/5 | 435 | mechanism fixed for cipher, gravity and bit manipulation, with new small arithmetic slips |
| v8 | SFT v3 (full 4,681-record corpus) | 5 | 2/5 | 3/5 | 398 | more data improved multi-digit arithmetic, one bit-manipulation regression |
| v9 | SFT v4 (+114 gravity-format records) | 5 | 2/5 | 3/5 | 397 | flat. The gravity change perturbed other outputs, which pointed to training-data perturbation rather than format confusion |

Leaderboard scores for the corresponding submissions are in `EXPERIMENTS.md`.
