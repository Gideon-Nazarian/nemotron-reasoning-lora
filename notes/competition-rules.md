# Competition Rules + Format (extracted)

One-time reference. Sources: official rules text, staff demo notebook, dataset inspection. Last updated: 2026-05-09.

## Critical format insight

**This is a LoRA adapter submission competition, not a predictions submission.** Submissions are trained LoRA adapter weights (zipped). Kaggle runs the base Nemotron model + your adapter on the private test set and scores the outputs.

## Model

- **Base model:** `metric/nemotron-3-nano-30b-a3b-bf16` (loaded via `kagglehub.model_download(...)`)
- **Architecture:** Nemotron 3 Nano — 30B total params, 3B active (MoE), Mamba-Transformer hybrid (uses `mamba_ssm`)
- **Precision:** bfloat16
- **NOT to be retrained / replaced.** The LoRA adapter is the only thing we ship.

## LoRA constraints

- **Max rank:** 32 (hard cap; demo comment says "Can be set to a maximum of 32")
- **Target modules:** `.*\.(in_proj|out_proj|up_proj|down_proj)$` — covers Mamba projections + attention proj layers + MLP up/down proj
- **Bias:** `"none"`
- **Task type:** `TaskType.CAUSAL_LM`
- **Adapter is saved with `model.save_pretrained(OUTPUT_DIR)`** then everything zipped to `submission.zip`

## Dataset

| File | Rows | Columns | Notes |
|---|---|---|---|
| `train.csv` | **9,500** | `id`, `prompt`, `answer` | Available for training. CC BY 4.0. |
| `test.csv` | **3** | `id`, `prompt` | Visible to participants — for sanity-checking only. |

The 3-row visible test is *not* the actual evaluation set. Kaggle injects the real **private test set** when our notebook runs at submission time. Standard Kaggle code-comp pattern.

## Problem types observed in train (need full categorization)

From inspection of first/sample rows:
1. **Bit manipulation** — Given example pairs of 8-bit binary inputs → outputs (rule = some combination of XOR / AND / OR / NOT / shifts / rotations / majority / choice). Predict output for a new input. Answer: 8-bit binary string.
2. **Cipher decryption** — Substitution cipher. Given example plaintext/ciphertext pairs, decrypt new ciphertext. Answer: a phrase.
3. (More categories likely — confirm via category extraction in Day 2.)

These are **few-shot rule-induction reasoning** problems. Pattern: in-context examples → induce rule → apply to new instance.

## Submission workflow

The submission is a Kaggle Notebook that:
1. Loads `metric/nemotron-3-nano-30b-a3b-bf16` from kagglehub
2. (Optionally) fine-tunes a LoRA adapter on `train.csv`
3. Saves adapter to `/kaggle/working/`
4. Zips everything to `submission.zip`

I click **Save Version → Save & Run All** on Kaggle, then **Submit to Competition**. Kaggle then runs the model + adapter on the private test set offline and scores it.

### Cutlass utility-script dependency

The demo notebook references:
```python
cutlass_pkg_path = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
site.addsitedir(cutlass_pkg_path)
```

This is a Kaggle utility script attached to the notebook. We need to make sure our notebook attaches `ryanholbrook/nvidia-utility-script` as a dependency so cutlass is available.

## Submission limits + selections

- **5 submissions per day max**
- **2 final submissions** selected for private LB judging
- Tiebreaker = earliest submission

## Evaluation

- **Public LB** = scored on a representative sample of the test set
- **Private LB** = scored on the full hidden test set; determines final ranking
- (Exact metric — string match? answer-format match? — to confirm from comp Overview tab. Likely some form of accuracy or normalized-edit-distance against the gold answer.)

## Prizes (recap)

- 1st: $25,000 + 5 DGX Sparks
- 2nd: $15,000 + 2 DGX Sparks
- 3rd: $5,000 + 1 DGX Spark
- Open Progress Prize: $5,000 + 1 DGX Spark — already awarded 2026-05-02 to Tong Huikang (LB 0.85)
- **Open Contribution Awards (1 DGX Spark each)** — judged on methodology:
  - Best Data/Synthetic Data Method
  - Best RL Method
  - **Best Fine-tuning Method** ← our realistic target
- Total: $106,388

## Deadline

**2026-06-15 23:59 UTC** (final submission). 37 days from today (2026-05-09).

## Reference

- Tong Huikang's writeup: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915
- Tong Huikang's 0.85 LB notebook: https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85
- Tong Huikang's repo: https://github.com/tonghuikang/nemotron (cloned to `reference/`)
- Staff demo notebook: https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo (downloaded to `notebooks/baseline/`)

## Open questions (resolve in Day 2)

1. Exact evaluation metric — read comp Overview tab
2. Whether Kaggle runs our adapter with vLLM, transformers, or something else at scoring time (affects what inference settings we should optimize for during fine-tuning)
3. Categorization of problem types in `train.csv` — need full breakdown
4. Maximum runtime cap for the submission notebook
5. Whether internet is on during submission run (affects whether we can download anything mid-notebook)
