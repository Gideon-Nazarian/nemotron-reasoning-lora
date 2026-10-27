# Nemotron Reasoning Challenge: LoRA fine-tuning for symbolic puzzles

My entry in the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) on Kaggle (May to June 2026, closed 2026-06-16). The task was to train a LoRA adapter (rank ≤ 32) for `Nemotron-3-Nano-30B-A3B-BF16`, a 30B-parameter mixture-of-experts model with a Mamba-Transformer hybrid architecture. The adapter has to make the model solve deterministic symbolic puzzles by writing out its reasoning and then a `\boxed{}` answer, within a 7,680-token budget.

This repo is the working record of the project: training notebooks, programmatic solvers that generate reasoning traces, an experiment ledger, and a post-mortem. I used AI coding assistants heavily to write and run the code. I directed the experiments, read the results, and made the calls.

## Problem

The puzzles are set in "Alice's Wonderland" and come in several families, including bit manipulation, substitution ciphers, numeral systems, cryptarithms, algebraic/transformation rules, unit conversion and gravity. Each gives a few input→output examples of a hidden rule and asks for the output on a new input. Kaggle runs the base model plus your adapter on a hidden test set. The test comes from the same generator as the training set. The training set is 9,500 prompts. The competition used a single 30B MoE model on one RTX Pro 6000, a 12-hour cap per run and a weekly GPU quota, with no internet at training time.

## Results

| | Score |
|---|---|
| Final **public** leaderboard | **0.868**, rank **125 of 4,355** teams (top 2.9%) |
| Best **private**-board score across my submissions | **0.852** |
| The two final submissions I selected, scored on the private board | about 0.844 and 0.840 |
| Independent from-base ceiling (my own training, ~10 runs) | **0.84** |
| Best result overall | 0.86, from wrapping a **public** adapter (Tong's v20, via a public wrapper notebook), not from my own training |

Two caveats:
- The rank is the **public**-leaderboard rank. The competition was decided on the private board, where the frontier scored 0.88 to 0.92. My selected finals scored about 0.84 on the private set.
- Public and private scores were uncorrelated for my submissions (r = −0.057 over 38 submissions with public score ≥ 0.84). A lot of what looked like progress on the public board was noise.

## What I found

- **Learnability beats correctness.** Training on my 100%-solver-verified reasoning traces did worse (0.72) than on a public adapter's fluent LLM-written traces (0.84). A trace being correct isn't the same as the base model being able to learn it.
- **Bimodal log-prob curation.** I scored all 10,311 training traces by the base model's teacher-forced loss. The distribution is bimodal with a valley near 0.75. Easy categories (gravity, equations, cipher, bit manipulation) sit at or below about 0.5. Hard ones (transformation rules about 1.6, numeral system about 1.3, cryptarithm about 1.0) sit above it. Those hard categories are the ones that sank my earlier runs. Dropping them cost 0.01 for me, because they make up a large part of the test set.
- **Completion-only masking looked like +0.01 and wasn't.** It scored 0.85 on the public board, and 0.828 on the private board, my worst serious result. That was noise. The from-base baseline (private 0.848) was within noise of everything I built afterwards.
- **Leaderboard noise.** Identical resubmissions varied by about ±0.02 to 0.03. With r = −0.057 between public and private scores in my band, the public board was not a usable fitness function. The lesson I'd carry forward: build a trustworthy local validation set first, before spending submissions on the noisy metric.
- **A silently dead LoRA layer.** The `out_proj` LoRA adapter does not train under Unsloth (`out_proj.lora_B` stayed at zero in my warm-start smoke test). A competitor later confirmed it is a gradient-checkpointing bug that affected every from-base run. Fixing it was not enough to close the gap.
- **Things that did not work.** Reproducing public "0.87/0.88" recipes from base (0.72 for one, which depends on private data, and 0.58 for another), SVD/merging of public adapters, a warm start that silently zero-initialised, and GRPO/RLVR (blocked by library issues and single-GPU generation throughput).

The full write-up is in [`REPORT.md`](./REPORT.md). The run-by-run ledger is in [`EXPERIMENTS.md`](./EXPERIMENTS.md).

## Repo layout

```
REPORT.md        final report and retrospective (start here)
EXPERIMENTS.md   ledger of every submission: commit, config, public score, takeaway
PLAN.md          strategy log from the competition (internal notes, lightly trimmed)
kaggle/          my early SFT/eval/RL-smoke kernels (notebook + kernel-metadata.json), eval notebooks, adapter-dataset stubs
notebooks/       local eval-harness and early SFT notebooks
scripts/         corpus builders, categorizer and RL reward code
scripts/solvers/ per-category programmatic solvers that generate reasoning traces
notes/           analysis notes (coverage gaps, log-prob filter result, recipe comparisons)
results/         my submission scores and the top of the final private board
```

## Reproducing

This is a research log, not a packaged library, so reproduction is partial.

- Training ran on Kaggle (RTX Pro 6000, offline wheelhouse). The notebooks mount Kaggle datasets and wheel packages that are not in this repo.
- **Competition data is not included.** Download `train.csv` from the competition page (CC BY 4.0, attribution: NVIDIA Research). Rebuild the training corpora with `scripts/categorize_train.py` and the `scripts/build_*_corpus.py` scripts, using the solvers in `scripts/solvers/`.
- Local setup: `uv sync` (see `pyproject.toml`). The training notebooks need a GPU with enough memory for a 30B MoE model.

### What is not in this public copy
Some paths mentioned in the notes, `PLAN.md` and `EXPERIMENTS.md` are intentionally absent:
- The generated training corpora and per-problem outputs (they contain competition data) and the full public-leaderboard snapshot.
- The early per-problem diagnostic write-ups. They are summarised in `notes/early-diagnostic-evals.md`.
- The Kaggle notebooks for the later experiments (v12 and v13 onward, plus the warm-start, repackaging and bank-adapter notebooks). Many are modified copies of other competitors' public notebooks, so I removed them rather than redistribute code under licenses I could not fully verify. The corpus builders (`scripts/build_*_corpus.py`), the solvers, the eval notebooks and the results and analysis for those experiments are kept. `EXPERIMENTS.md` still documents every run.

## Credits

This project built on other competitors' public work. Their code is not redistributed here. These are references and thanks:

- **dgxchen**, ["Training with Unsloth to Achieve 0.85 LB"](https://www.kaggle.com/code/dgxchen/training-with-unsloth-to-achieve-0-85-lb), the training recipe and public CoT dataset my later experiments built on.
- **Tong Huikang**, the 0.85 fine-tuning notebook ([huikang/end-to-end-finetuning-for-lb-0-85](https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85)) and [github.com/tonghuikang/nemotron](https://github.com/tonghuikang/nemotron).
- **asalhi**, [tinker-adapter-to-ready-to-submit-adapter](https://www.kaggle.com/code/asalhi/tinker-adapter-to-ready-to-submit-adapter), the wrapper notebook behind my 0.86 submissions.
- **Kh0a**, [nemotron-unsloth-sft-training-3-30-2](https://www.kaggle.com/code/llkh0a/nemotron-unsloth-sft-training-3-30-2), the boxed-loss recipe I tried to reproduce.
- **kuangyicheng**, [nemotron-087-training](https://www.kaggle.com/code/kuangyicheng/nemotron-087-training), also tried.
- **Ryan Holbrook** and the competition staff, [nvidia-nemotron-submission-demo](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo).
- **mayukh18**, the offline `nemotron-packages` wheel dataset used for training.
- **NVIDIA Research**, for the competition, its data (CC BY 4.0) and the Nemotron-3-Nano model (NVIDIA Open Model License).

## License

The code, notes and write-ups in this repo are my own work, released under the MIT License (see `LICENSE`). The basic model-loading and LoRA setup in the early notebooks follows the competition's public demo notebook (linked under Credits). Competition data and the NVIDIA model are covered by their own licenses (see `NOTICE`).
