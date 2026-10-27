# Track A — SFT loss-weighting (the higher-EV, infra-free lever)

> Chosen lead path after reconciling several independent research reviews of the GRPO blocker.
> Rides the PROVEN v18 0.84 SFT pipeline — no Unsloth-GRPO minefield, no s/step gate. Last updated 2026-06-02.

## A1 — token-masking audit: RESULT = v18 does NOT mask the prompt (a real lever is on the table)

**Verified in `kaggle/v18-hybrid/v18.ipynb`:**
- SFT uses TRL `SFTTrainer` (subclassed `StratifiedSFTTrainer`, which only overrides the dataloader for
  type-stratified batching) + `formatting_prompts_func` that returns the FULL chat-templated text
  (`apply_chat_template(conv, tokenize=False, add_generation_prompt=False, enable_thinking=True)`).
- `SFTConfig`: `packing=False`, `max_length=MAX_SEQ_LEN`. **`completion_only_loss` is NOT set** (→ default
  None → language-modeling mode). **ALL masking signals absent:** no `DataCollatorForCompletionOnlyLM`,
  no `response_template`, no `train_on_responses_only`, no `labels=-100`.

**⇒ v18 computes cross-entropy over the ENTIRE sequence — the problem statement AND the assistant's
reasoning+answer.** NVIDIA's own Nano-3 SFT recipe explicitly masks user/system tokens (loss weight 0) so
gradient flows only from assistant responses (R3). v18 omits this. **This is the single most standard SFT
lever, and it's not applied → highest-priority cheap test.**

**Calibration (important — don't assume it helps):** dgxchen's recipe scored 0.84/0.85 WITHOUT masking, and
our 5-run finding (EXPERIMENTS 19–22) showed "more correct" training changes can generalize WORSE
(voice-mismatch). So masking is a PLAUSIBLE win, not a guaranteed one. **Must eval on the local harness vs
the 0.84 baseline before spending a submission.** A flat/worse local score would itself be informative
(tells us dgxchen's no-mask was load-bearing).

## A2 — per-category / hard-example weighting (builds on A1)
- **Plumbing already exists:** every corpus record carries a `type` field (`build_v18_corpus.py`), and the
  trainer already stratifies batches by `type`. So per-category loss weighting is a small `compute_loss`
  override in `StratifiedSFTTrainer`: scale per-token loss by a per-`type` weight.
- **Needs:** a per-category ACCURACY breakdown (which categories v18 gets wrong) → up-weight weak ones,
  down-weight already-solved/easy. Source: the eval harness (`kaggle/eval-harness` / `kaggle/eval-trained`)
  + `scripts/categorize_train.py` (`category_by_answer` / `category_by_prompt`).
- Focal-style hard-example weighting (down-weight low-loss/easy tokens) is the same hook.

## A3 — RAFT / Reinforce-Rej (R1; offline rejection-sampling SFT) — later in Track A
Generate N from v18 on the p≈0.5 band → filter by our solver verifier → SFT the winners. Self-generated =
right voice + verified = correct → directly counters our voice-mismatch. Offline gen = resumable, s/step
defanged. Reinforce-Rej (drop all-correct too) preserves diversity. Build after A1/A2 if they plateau.

## Experiment loop (per variant; matches R3's "eval local before submit")
1. Train the variant on the proven v18 pipeline (1 GPU run, ~8min load + train).
2. Eval the trained adapter on the LOCAL solver harness (held-out) → estimate LB.
3. Only submit if local beats v18's local baseline. Bank v18=0.84 if nothing clears.

## Open design choice for A1 (nail during build)
Masking method on the Unsloth+TRL stack: cleanest is Unsloth `train_on_responses_only(trainer,
instruction_part=..., response_part=...)` — needs Nemotron-H's chat-template assistant-header strings
(inspect `tokenizer.chat_template`). Alt: TRL `completion_only_loss=True` w/ messages (drop formatting_func),
or a custom -100 collator. Decide by reading the actual chat template.

## A1 BUILD — `kaggle/v20-masked-sft/v20.ipynb` (done, validated, NOT yet run)
Clone of v18 with EXACTLY ONE change (verified: only cells 0 + 11 differ): after the trainer is created
and before `trainer.train()`, wrap with Unsloth `train_on_responses_only(instruction_part="<|im_start|>user\n",
response_part="<|im_start|>assistant\n")`. Markers verified from the model's actual chat_template (Nemotron-3-Nano
= ChatML). Same corpus (nemotron-v18-corpus), same hyperparams, same stratified batching, same model load.
- **Safety: a hard `_verify_masking` block runs BEFORE training** — decodes a sample's masked vs unmasked
  spans, asserts mean masked-fraction ∈ (0.03, 0.90) and that the answer/`\boxed`/`</think>` is in the
  UNMASKED region. If `train_on_responses_only` silently no-ops, the run ABORTS at step 0 (cheap), not after
  a full train. Also found: the older `kaggle/training/05-lora-sft-v5-*.ipynb` already did this masking
  (`labels=[-100]*len(prompt)+response`) → v18 was a REGRESSION from it.
- **Run + eval loop:** run on RTX → produces submission.zip → eval the adapter on the local harness
  (`kaggle/eval-harness`) and/or submit for the LB score. Compare vs v18=0.84. Keep only if it clears.
- Honest: masking is the standard lever but NOT guaranteed to help here (dgxchen got 0.84 no-mask; voice-mismatch
  history). The run TELLS us. A flat/worse result = informative (no-mask was load-bearing).


## LOCAL EVAL (built 2026-06-02) — reusable held-out harness
`kaggle/_lib/eval_heldout_lib.py` (CPU logic unit-tested locally): held-out = the **1,339 train.csv
problems whose prompt is NOT in the v18 corpus** (never trained on) = 8,161 trained + 1,339 held-out =
9,500 ✓. Deterministic category-stratified sample (n=160, seed 3407 → 40 each of bit_manipulation /
numeric_int / numeric_decimal / other). Greedy generate → exact-match vs gold → overall + per-category.
- **Inline in `kaggle/v20-masked-sft/` (v2):** trains masked + evals itself → v20's local number.
- **`kaggle/v18-baseline-eval/`:** v18 unmasked + the SAME eval (same 160, seed 3407) → the baseline.
  Run once; reusable baseline for ALL Track-A variants.
- **Caveat:** held-out is the UNTRAINED (category-skewed) subset → a DIRECTIONAL proxy, not the LB.
  Verdict = compare the two local numbers AND/OR submit v20 to the LB vs 0.84.

## Status
- [x] A1 audit — DONE (no masking in v18; lever confirmed; v18 regressed from v5 which DID mask).
- [x] A1 build — `kaggle/v20-masked-sft/` built + validated (1 change, guarded). **Ready to run on RTX.**
- [x] A1 build + local-eval harness (v20 inline + v18-baseline-eval) — pushed, ready to run.
- [ ] A1 run (v20) + baseline run (v18-baseline-eval) → compare local; submit v20 to LB.
- [ ] A2 — per-category weighting (needs per-category accuracy breakdown first).
- [ ] A3 — RAFT (after A1/A2).
