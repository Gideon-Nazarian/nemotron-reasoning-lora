# Nemotron Reasoning Challenge: LoRA fine-tuning for symbolic puzzles

My entry in the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) on Kaggle (May to June 2026, closed 2026-06-16). The task: train a LoRA adapter (rank ≤ 32) for `Nemotron-3-Nano-30B-A3B-BF16`, a 30B mixture-of-experts model, so it solves deterministic symbolic puzzles (ciphers, bit manipulation, unit conversion, gravity, transformation rules and more) by writing out its reasoning and a `\boxed{}` answer within a 7,680-token budget. This repo is the working record: programmatic solvers that generate reasoning traces, training and eval notebooks, an experiment ledger and a post-mortem.

## Results

| | Score |
|---|---|
| Final **public** leaderboard | **0.868**, rank **125 of 4,355** teams (top 2.9%) |
| Best **private**-board score across my submissions | **0.852** |
| My two selected finals, on the private board | about 0.844 and 0.840 |
| Independent from-base ceiling (my own training, ~10 runs) | **0.84** |

The 0.86 submissions came from wrapping a public adapter, not from my own training. The rank is the public-leaderboard rank. The prize-deciding private board had its frontier at 0.88 to 0.92.

## Key findings

- **Bimodal base-loss curation.** I scored all 10,311 training traces by the base model's teacher-forced loss. The distribution is bimodal with a valley near 0.75, and the high-loss categories (transformation rules, numeral system, cryptarithm) are the ones that sank my earlier runs. Dropping them cost 0.01 because they are a large part of the test set.
- **Masking's +0.01 was noise.** Completion-only masking scored 0.85 on the public board and 0.828 on the private board, my worst serious result.
- **The public leaderboard was noisy.** Identical resubmissions varied by about ±0.02 to 0.03, and public vs private scores were uncorrelated (r = −0.057 over 38 submissions). Lesson: build a trustworthy local validation set before optimizing the public metric.
- **`out_proj` LoRA was silently dead under Unsloth.** `out_proj.lora_B` stayed at zero in my warm-start smoke test. A competitor later confirmed it is a gradient-checkpointing bug affecting every from-base run.
- **Correct traces aren't always learnable.** My 100%-solver-verified traces scored 0.72, versus 0.84 for a public adapter's LLM-written traces.

## Repo map

- [`REPORT.md`](./REPORT.md): final report and retrospective (start here)
- [`EXPERIMENTS.md`](./EXPERIMENTS.md): ledger of every submission with config, score and takeaway
- `scripts/`: corpus builders and per-category solvers (`scripts/solvers/`)
- `kaggle/`, `notebooks/`: early SFT, eval and RL-smoke notebooks
- `notes/`, `results/`: analysis notes and submission scores

## Notes

Competition data is not included (get `train.csv` from the competition page, CC BY 4.0, NVIDIA Research), and neither are the notebooks for the later experiments, so reproduction is partial. AI coding assistants wrote and ran much of the code; I set the direction, read the results and made the calls.

Builds on public work from other competitors in the Kaggle community; see `REPORT.md` for details.

Released under the MIT License (see `LICENSE` and `NOTICE`).
