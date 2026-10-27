# Leaderboard intel + published recipes — 2026-05-17 evening

The single most strategically useful document in this folder. Captures the topology
of the leaderboard, the full published recipes of the top public competitors, all
known contradictions between those recipes, and every experiment we and others have
already tried (with outcomes). Read this BEFORE designing any new training run — it
will save 5-10 GPU hours per avoided false-start.

Updated: 2026-05-17 ~20:30 PT (after 3 research agents + v9 diagnostic eval).

---

## TL;DR

1. **The leaderboard is a Tong-shoulders-of-giants exercise.** Of ~3,200 submissions, ~700 tie at 0.86 LB. The 0.86 cluster is people running `asalhi/tinker-adapter-to-ready-to-submit-adapter` (510 votes, CC0, public) which wraps Tong Huikang's adapter **version 20** (Apache-2.0, public). Zero training. Top 4 teams at 0.87 are private, but **Kh0a (rank 4) has a public precursor notebook** that reveals their full recipe.
2. **The 0.86 → 0.87 lift is data + loss-weighting, not architecture.** Per Kh0a's spec: custom CoT traces per category + category-rebalanced sampling + 5× CE-loss upweight on tokens after `\boxed{`.
3. **Two top public recipes (Kh0a 0.87-ish, dgxchen 0.85) have direct contradictions** on `lm_head`, batch size, LR, epochs, and loss function. These aren't independent knobs — they're operating-point packages.
4. **Tong's post-prize improvements (v18 prize → v20 LB 0.86) are NOT pushed to public sources.** His GitHub terminates 5 days BEFORE the prize. The +0.01 from v18→v20 is unknowable from public data.
5. **Our v1-v9 trajectory established that data-shape changes alone cost ~0.02 LB at our v3 stack.** Don't retry. Our v9 breakthrough (LB 0.62) came from multi-epoch on the Unsloth+CCE stack.

---

## PART A — Leaderboard topology (2026-05-17 evening)

```
Score | Count  | What's there
------+--------+-----------------------------------------------------------
0.87  | ~4     | Y|M|F, Researcher 7919, "Lora is all you need", Kh0a
0.86  | ~700   | Tong adapter v20 via asalhi wrapper (the cluster)
0.85  | ~?     | Tong's prize-time adapter (v18) re-runs + dgxchen's notebook
0.5-0.85 | rest | Custom approaches, partial recipes
0.62  | 1      | Us (v9 high-water mark, NOT competitive)
0.50  | many   | Baseline / zero-shot
```

**Why this shape matters:** the cluster at 0.86 means the public ceiling is well-known. Spending compute to get from 0.62 → 0.85 from scratch is wasteful when forking a 0.86 baseline is one click. The real research question is **how to push 0.86 → 0.87+**.

---

## PART B — Published recipes (full specs)

### B.1 Tong Huikang's prize recipe (LB 0.85, 2026-05-02)

Source: `reference/huikang-0.85-lb/end-to-end-finetuning-for-lb-0-85.ipynb` cell 1, plus `reference/tonghuikang-nemotron/` (GitHub mirror frozen at 2026-04-27).

| param | value |
|---|---|
| LoRA rank / alpha / dropout | 32 / 32 / 0 |
| target_modules | q/k/v/o + in/out + up/down + lm_head (9) |
| `train_unembed` | True (lm_head LoRA explicitly trained) |
| max_seq_len | 8192 |
| batch | 32 |
| micro_batch | 4 |
| num_steps | 1000 |
| LR | 2e-4 |
| optimizer | AdamW betas (0.9, 0.95), eps 1e-8, wd 0 |
| schedule | StepLinearDecayLRSchedule → 0, no warmup, no min floor |
| MOE_TIE_WEIGHTS | True |
| `RESET_WEIGHTS` | True |
| `IN_PROJ_ONLY` | False |
| loss | plain `CrossEntropyLossConfig` (but `CrossEntropyWithWeightingLossConfig` is also defined and unused!) |
| data | per-problem investigations + 9 reasoner-generated CoTs + 5 augmenter sets |
| stratified batching | Yes (`_stratified_batches` in `train_sft.py`, by category) |

Data corpus components (verified via local clone):
- 1,667 investigation files (hand-written per-problem rules) in `investigations/`
- 9 reasoners covering all comp categories (cipher 381 LOC, bit_manip 1005 LOC, equation_numeric 603 LOC, cryptarithm 164 LOC, gravity 87 LOC, unit_conversion 75 LOC, numeral 58 LOC, store_types 274 LOC)
- 5 augmenters: spelling 115 LOC, concatenation 83 LOC, splitting 83 LOC, lstrip 79 LOC, matching 273 LOC
- `wonderland.txt` (77 plaintext words for cipher) + `dictionary.txt` (1596 words)

### B.2 Tong adapter v20 (LB 0.86 — the cluster driver)

**Source: NOT PUBLIC.** Tong's GitHub `tonghuikang/nemotron` terminates at commit `82bd1880aa` on 2026-04-27, five days BEFORE the prize. Tong has continued publishing newer adapter weights on Kaggle (`huikang/nemotron-adapter` versions 18 → 20 → 26 at least) without writing them up.

**Best-guess of the v18→v20 delta** (medium-low confidence, per Agent 2):
1. Turned on `CrossEntropyWithWeightingLossConfig` (built but unused at prize time) — one-line code change.
2. Larger/refreshed corpus snapshot — his latest training run timestamp `04-13-13-09` post-dates the prize-frozen `04-08-16-14`.
3. Possibly expanded `wonderland.txt` or `dictionary.txt`.

**These guesses are NOT verifiable from public data.** Anyone reproducing v20 would have to reverse-engineer from the adapter binary diff.

How to bank 0.86 with zero training:
1. Fork `https://www.kaggle.com/code/asalhi/tinker-adapter-to-ready-to-submit-adapter`
2. Save & Run All (the notebook is 4 cells, runs in ~10 min CPU-only)
3. Submit the resulting `submission.zip` to the comp
4. Score: 0.86 (license: Tong's adapter Apache-2.0, wrapper notebook CC0)

### B.3 dgxchen's recipe (LB 0.85 — public training notebook)

Source: `https://www.kaggle.com/code/dgxchen/training-with-unsloth-to-achieve-0-85-lb`. Local copy: `/tmp/dgxchen-recipe/`.

| param | value |
|---|---|
| LoRA rank / alpha / dropout | 32 / 32 / 0 |
| target_modules | q/k/v/o + in/out + up/down — **8 modules, NO lm_head, NO gate_proj** |
| max_seq_len | 8192 |
| per_device_train_batch_size | **1** |
| gradient_accumulation_steps | 32 (effective batch = 32) |
| LR | 2e-4 |
| lr_scheduler_type | linear |
| warmup_steps | 0 |
| optimizer | AdamW betas (0.9, 0.95), eps 1e-8, wd 0 |
| max_grad_norm | 1e9 (effectively no clipping) |
| num_train_epochs | **1** |
| bf16 | True |
| gradient_checkpointing | True (use_reentrant=False) |
| packing | False |
| seed | 42 |
| Unsloth `unsloth_force_compile` | False |
| Cut Cross-Entropy patch | **NO** (he reaches 0.85 without it) |
| `is_fast_path_available` patch | **NO** |
| MoE expert tying | NO explicit logic |
| stratified batching | Yes (custom sampler by `row["type"]`) |
| data | `dgxchen/nemotron-cot-tong/problem_ids_matched.csv` (re-host of Tong's CoT, ~tens-of-thousands of rows) |
| filter | drop rows where `generated_cot` is empty/"nan"/len < 5 |
| prompt suffix | `\nPlease put your final answer inside \\boxed{}...` (same as ours) |
| assistant format | `re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip() + '\n</think>\n\\boxed{{{answer}}}'` |
| chat template | `add_generation_prompt=False, enable_thinking=True` |

**Headline finding from his notebook:** removing `lm_head` from `target_modules` (vs an earlier version that included it) was their +0.1 LB lift. Their notes also explicitly say bumping `per_device_train_batch_size` from 1 to 2 cost ~0.1 LB.

### B.4 Kh0a's recipe (LB 0.87 — public precursor notebook)

Source: `https://www.kaggle.com/code/llkh0a/nemotron-unsloth-sft-training-3-30-2`. Local copy: `/tmp/kh0a-recipe/`, repo copy: `kaggle/v12-kh0a-baseline/`.

| param | value |
|---|---|
| LoRA rank / alpha / dropout | 32 / **16** / **0.1** |
| target_modules | **out/v/q/down/embed_tokens/k/in/up/o/lm_head/gate_proj** — **ALL 11 INCLUDING embed_tokens AND lm_head** |
| max_seq_len | **3500** |
| per_device_train_batch_size | **2** |
| gradient_accumulation_steps | 1 (effective batch = 2!) |
| LR | **1e-4** |
| lr_scheduler_type | **cosine** |
| warmup_ratio | 0.03 |
| optimizer | **adamw_8bit** |
| num_train_epochs | **2** |
| max_grad_norm | (default 1.0) |
| **BOXED_LOSS_WEIGHT** | **5.0** ← 5× CE upweight on every token after the LAST `\boxed{` in the label sequence |
| stratified batching | Yes (custom sampler) |
| category rebalancing | **YES** — `TRAIN_SAMPLES_PER_CATEGORY` skews to LB-dominant categories: 1400 bit_manip + 1300 text_decryption + 600 numeric_equation + 200 each (unit_conversion/numeral_system/gravity_physics) + 10 symbol_transform |
| Unsloth `unsloth_force_compile` | False |
| attn_implementation | eager |
| eval_strategy | `steps`, every 300 |
| save_strategy | `no` (doesn't save checkpoints during training) |
| load_best_model_at_end | False |
| data | `dgxchen/nemotron-cot-tong` (Tong CoT) + private `llkh0a/nvidia-nemotron-distiled-dataset/{splited.csv, bit_manipulation_traces_v4.csv, numeric_equation_traces_new.csv}` |
| inference (validation) | greedy, temp=0, top_p=1, max_new_tokens=7680 |

**The killer trick: `BOXED_LOSS_WEIGHT = 5.0`.** Implementation: custom `compute_loss_func` passed to `SFTTrainer` that scales per-token CE by 5× for every token at-or-after the last occurrence of the `\boxed{` marker tokens in the label sequence. Nobody else's public recipe does this. Likely worth +0.005 to +0.02 LB on its own.

**The known bug in Kh0a's public notebook:** cell 26 references `generate_bit_manipulation_trace(prompt, answer, sample_id)` but the cell containing it accidentally has a duplicate of `generate_text_decryption_trace`. Kh0a's private `bit_manipulation_traces_v4.csv` cached every bit_manip ID they used, so the fallback path never fired in their runs. **Running Kh0a's public notebook as-is would NameError on the first bit_manip example.** Our v12 calibration patches this with our v3 bit_manipulation solver inline (89.1% gold coverage).

Functions Kh0a's notebook DOES define inline (verified):
- `solve_numeral_with_trace`
- `solve_gravity_with_trace`
- `solve_unit_conversion_with_trace`
- `generate_text_decryption_trace`
- `generate_symbol_transform_trace`
- `generate_numeric_equation_trace`
- NOT `generate_bit_manipulation_trace` (the bug)

---

## PART C — Contradictions between dgxchen and Kh0a

Both recipes are published, both work, but they directly disagree on multiple knobs. Resolution: **these aren't independent variables, they're operating-point packages.** Mixing them naively will break.

| knob | dgxchen (0.85) | Kh0a (0.87) | resolution |
|---|---|---|---|
| `lm_head` in targets | NO (says including it cost 0.1 LB) | **YES + embed_tokens** | Kh0a's lower LR (1e-4) + dropout (0.1) + boxed-loss-weight stabilizes lm_head training; dgxchen's hot recipe (2e-4 LR, dropout 0, boxed weight 1.0) overfits when lm_head is trained |
| per_device_batch_size | 1 (says bumping to 2 cost 0.1 LB) | 2 | Different effective batch (32 vs 2) → completely different optimization regime |
| effective batch | 32 | **2** | High gradient variance at batch=2 is part of Kh0a's regularization |
| LR | 2e-4 | 1e-4 | Halved LR matches halved alpha (16 vs 32) → similar gradient magnitude |
| lr_scheduler | linear | cosine | Cosine + 2-epoch + warmup_ratio=0.03 is a different schedule shape |
| epochs | 1 | 2 | Tied to scheduler shape |
| dropout | 0 | 0.1 | Regularization tradeoff |
| optimizer | default (adamw_torch) | adamw_8bit | Memory only — likely no LB effect |
| loss | plain CE | **BOXED_LOSS_WEIGHT=5.0** | Kh0a's key innovation; dgxchen doesn't have it |

**Implication for v12+:** if we want to try anything in dgxchen's column, we have to commit to the WHOLE package (8 target modules, batch 32, LR 2e-4, plain CE). If Kh0a's, we commit to the WHOLE package (all 11 modules incl embed+lm_head, batch 2, LR 1e-4, cosine, BOXED_LOSS_WEIGHT 5.0). DO NOT freelance mix.

---

## PART D — Our experiments and what we already learned

Full LB ledger:

| ver | LB | stack | data | takeaway |
|---|---|---|---|---|
| baseline | 0.500 | untrained adapter | n/a | floor |
| v1 | 0.510 | LoRA on 750 stratified records, Fix-A reasoning, batch 8, 1 epoch | 750 records | template overfit; model learned format perfectly but fabricated values |
| v2 | 0.540 | same, with per-example arithmetic CoT (Fix A) | same 750 records | Fix-A worked at the mechanism level |
| v3 | **0.580** | scaled to 4,681 records | full solver corpus | each iteration produced bigger lifts; new SFT HWM |
| v4 | 0.560 | v3 + gravity 1-dp format fix (114 records) | v3 + 114 | **REGRESSED -0.02.** Diagnosis: data-shape perturbation, not format confusion. Holdout aggregate identical to v3 but per-problem behavior shifted on unrelated problems. **DO NOT REDO format-precision fixes alone.** |
| v5 | 0.560 | v3 + 800 synth transformation_rules records | v3 + 800 | **REGRESSED -0.02.** Same magnitude as v4. **Lesson: ANY data-shape change costs ~0.02 at the v3 stack.** v3 sits at a local optimum for its data shape. **DO NOT ADD records to one category without changing architecture.** |
| v6 | 0.580 | v3 data BYTE-identical + MoE tying + 9 target mods + alpha=r + dropout=0 + AdamW betas | v3 corpus | Flat. Post-run we discovered tying logic had a silent bug (per-expert mean collapsed to rank-1). |
| v7 | 0.580 | v6 with CORRECTED tying + pre-train verification | v3 corpus | **Flat.** Clean negative result: MoE expert weight tying is NOT the missing piece at our data scale. v6/v7 loss curves indistinguishable. **DO NOT REDO MoE tying as a standalone lever.** |
| v8 | 0.560 | full Unsloth + Cut-CE + 9 target mods + manual lm_head LoRA, 1 epoch batch 32 | v3 corpus | REGRESSED. Diagnosis: undertrained (146 steps too few for the bigger batch despite same data exposure). |
| **v9** | **0.620** | v8 stack × **3 epochs** (438 steps) | v3 corpus | **BREAKTHROUGH +0.04** vs v3. First SFT lift in 6 submissions. Lesson: multi-epoch at Unsloth+CCE was the missing lever. |
| v10 | (cancelled) | v9 stack × 5 epochs | v3 corpus | Cancelled after Kh0a recipe identified as a better use of compute |
| v11 | (cancelled) | v9 stack + expanded solver corpus (7,436 records, +59%) | new corpus | Cancelled after LB intel showed from-base training can't catch 0.86 cluster |
| v12 | (training) | Kh0a's full recipe + our v3 bit_manip solver inline (filling Kh0a's missing function) | Tong CoT + our patches | **Calibration run** — does Kh0a's recipe reproduce 0.85+ for us today? Expected ~6-10 hr wall |

### v9 5-problem diagnostic eval (2026-05-17 afternoon)

3/5 strict, 4/5 substring (up from v3 adapter's 2/5, 3/5). Categorical findings:
- bit_manip rotl_2 case: ✓ (solver-derived rule still works post-v9)
- unit_conv (ratio 1.286): ✓ (5-example back-test converged cleanly)
- cipher 5-word (all letters in examples): ✓
- cipher (letter `m` missing from examples): **✗ — predicted `ousee` instead of `mouse`.** Our v11/v12 cipher v2 wordlist fallback fixes exactly this case.
- gravity (gold `57.0` with 1 decimal place): ✗ strict, ✓ substring. The 1-dp gravity-gold issue (~9% of gravity golds use 1-dp format) is unsolved — the model output 2-dp `57.03`. v4 attempted to address this and regressed -0.02, so we've left it.

### Our solver upgrades (this morning, all gold-validated)

| solver | before | after | net change |
|---|---|---|---|
| bit_manipulation | 9.4% / 150 records | **89.1% / 1,428 records** | +1,278, three-tier shift composer ({SINGLE, TWO_TERM, THREE_TERM} × {SHL,SHR,ROT}×{1..7} × {XOR,AND,OR} ± NOT) |
| cipher | 38.4% / 605 records | **100% / 1,576 records** | +971, Wonderland wordlist fallback for unknown-letter words |
| gravity | 64.0% / 1,022 records | 64.0% / 1,022 records | 0 records, but reasoning text 500 chars → 4,897 chars avg (~10× richer arithmetic trace) |
| unit_conversion | 83.3% / 1,328 records | 83.3% / 1,328 records | 0 records, reasoning text ~500 → 3,115 chars (~6× richer) |
| transformation_rules (NEW) | 0% / 0 records | 32.5% / 506 records | +506, numeric subset via operator-grouped candidate enumeration |
| numeral_system | 100% / 1,576 records | 100% / 1,576 records | unchanged |
| **TOTAL** | 4,681 records | **7,436 records (+59%)** | All categories represented |

Stored in `data/training-data.jsonl` (current head) and `kaggle/training-data-dataset/training-data-v11.jsonl` (version-pinned). Uploaded as v6 of the `gideonnazarian/nemotron-solver-training-data` Kaggle dataset.

---

## PART E — Things we KNOW DO NOT WORK / are regressive

These are documented anti-patterns. Each ate compute time without LB lift, OR caused -0.02 LB regression. **Do not re-test as standalone levers.**

1. **Format-precision changes to a single category at v3 stack** (v4 evidence). Adding 114 1-dp gravity records to a 2-dp-trained corpus → -0.02 LB via behavior shifts on UNRELATED problems. Holdout aggregate was identical to v3 but per-problem outputs perturbed.

2. **Adding records to one previously-uncovered category at v3 stack** (v5 evidence). +800 synthetic transformation_rules records → -0.02 LB. Same magnitude as v4. The v3 corpus sat at a local optimum for its shape; ANY shape change paid the regression tax.

3. **MoE expert weight tying as a standalone lever** (v6/v7 evidence). Tested cleanly with and without a silent rank-collapse bug. Neither moved LB above v3's 0.58 at our training scale. Tong has MoE_TIE_WEIGHTS=True too but his lift comes from elsewhere.

4. **Cut Cross-Entropy + 9 target modules + alpha=r + dropout=0 + AdamW betas at 1 epoch / batch 32** (v8 evidence). Undertrained at 146 steps → -0.02 vs v3. The stack only pays off at multi-epoch (v9).

5. **Including `lm_head` in `target_modules` with dgxchen-style hot recipe** (LR 2e-4, dropout 0, plain CE, batch ≥32). dgxchen notes this cost ~0.1 LB. Lm_head training only works combined with the cooler Kh0a-style regime (LR 1e-4, dropout 0.1, BOXED_LOSS_WEIGHT, smaller batch).

6. **Bumping `per_device_train_batch_size` from 1 to 2 in dgxchen-style runs.** dgxchen notes this cost ~0.1 LB. Micro-batch=1 is load-bearing in his recipe.

7. **Training from base model to compete with the public ceiling.** Even with our 7,436-record solver corpus and v9 stack, the floor for from-base SFT is ~0.62-0.65. The public ceiling is 0.86. Compute is dramatically better spent stacking on Tong's adapter or replicating Kh0a's recipe.

8. **GRPO/RLVR at our current setup.** Multiple smoke-test attempts (v1-v9 in `kaggle/training-rl/`) hit Kaggle env issues (torchvision CUDA version mismatch from `ryanholbrook/nvidia-utility-script` update). Smoke tests are scaffolded but blocked on env fix. ~1-2 hr to debug remaining. Worth resuming ONLY if SFT branch is exhausted.

---

## PART F — Open levers worth testing (informed by all the above)

Roughly ranked by expected-value-per-GPU-hour, with crisp reasoning:

### Tier 1 — high confidence, low risk
- **Calibration: Kh0a's recipe on our account** (v12 running NOW). Validates the published 0.85+ recipe reproduces for us. Tells us what floor v13+ deltas build on.
- **Bank 0.86 via asalhi wrapper** (5 min user-side work, near-zero compute). De-risks the entire competition outcome. One of our 2 final submission slots.

### Tier 2 — moderate confidence, higher reward
- **`BOXED_LOSS_WEIGHT=5.0` swap into ANY successful recipe.** Kh0a's specific innovation; not present in Tong v18, dgxchen, or our work. Plausibly +0.005 to +0.02 LB.
- **Category-rebalanced sampling** weighted toward LB-dominant categories (Kh0a's 1400/1300/600 vs 200/200/200/10 split). Tong does category stratification but not skewed sampling. Plausibly +0.005 to +0.015 LB.
- **Add our cipher v2 (Wonderland wordlist fallback) records to whatever data feeds v13.** v9 eval directly evidenced this fixes the `m`-not-in-examples failure mode. The 971 records we gained are gold-validated.
- **Use stratified batching by category** (Tong + dgxchen + Kh0a all do this; we don't). Agent #3's 2026-05-17 morning research flagged this as "likely 1-3 LB points alone."

### Tier 3 — speculative, untested
- **`CrossEntropyWithWeightingLossConfig`** swap in Tong-style training. Tong has this defined but used plain CE at prize time. Possibly what made v20.
- **Expanded `wonderland.txt`** if any pattern of failures suggests known words missing from the 77-word list. Worth checking against our 0.86 baseline error analysis once we have one.
- **Self-consistency-baked-into-training.** Since grader is single-pass vLLM, can't do inference-time SC. But we could train on multiple CoT paths per problem so the single-pass output integrates diversity. Speculative, no published prior.

### Tier 4 — explicitly do NOT do (see Part E)
- More multi-epoch from base at v9 stack (v10-v11)
- Any standalone "tweak one thing in v3 stack" experiment
- Mixing dgxchen and Kh0a knobs naively

---

## PART G — Decision tree for v13+ (after v12 calibration result)

### Important caveat: v12 tests a COMPOSITE hypothesis

v12 is NOT a clean test of "does Kh0a's recipe reproduce." It tests three
things at once, confounded inside one LB number:

1. Does Kh0a's recipe reproduce on our account / today's test set?
2. Is our v3 bit_manip substitute ≥ as good as Kh0a's private cached traces?
3. Does the missing private val split (`splited.csv`) affect early stopping?

If v12 lands ≥ 0.85 the composite is confirmed and we can proceed. If it
lands lower, we **cannot cleanly attribute** the gap to any single
component. The disciplined response to a confounded result is an isolation
experiment, not a guess. Below decision tree reflects this honestly.

### Our pre-run estimate of substitute quality

Our v3 bit_manip solver covers **89.1%** of train.csv bit_manipulation
problems with gold-validated Tong-style traces (1,428/1,602). On Kh0a's
1,400-sample bit_manip training subset (assuming uniform sampling from the
1,602), this means ~1,247 records get a clean trace and ~150 records get
the generic "examine examples and apply" fallback string. Kh0a's private
CSV almost certainly covered all 1,400 with real traces (which is why their
cache existed and the public notebook's fallback path was never exercised).

So our substitute is **materially worse on ~10% of Kh0a's bit_manip records**,
which is ~3.6% of total training. The plausible LB cost of this noise is
small — probably ≤0.02 — but unknown without running the experiment.

### When v12 LB lands

```
v12 LB score
│
├── ≥ 0.85: Composite confirmed. Both recipe and substitute work.
│   We cannot tell if our substitute is BETTER than Kh0a's or just
│   AS GOOD — but at the public ceiling, that distinction doesn't matter.
│   v13 layers more deltas, one at a time:
│   - Replace cipher CoT generator with our cipher v2 (wordlist fallback)
│   - Add our transformation_rules numeric solver records to the corpus
│   - Try CrossEntropyWithWeightingLossConfig instead of BOXED_LOSS_WEIGHT
│   - Try LoRA rank 64 (vs Kh0a's 32) — but pair with halved alpha if overfit
│
├── 0.80 - 0.84: CONFOUNDED. Cannot cleanly attribute the gap. Could be:
│   - Our substitute is worse than Kh0a's (most likely; ~10% of bit_manip
│     records get a fallback trace instead of a real one)
│   - Recipe doesn't reproduce (seed/version drift across our account)
│   - Missing private val split hurts early stopping (minor)
│   - Test set drift since Kh0a's score (timing)
│   ISOLATION EXPERIMENT (mandatory before v13): run v12-baseline =
│   Kh0a recipe with TRAIN_SAMPLES_PER_CATEGORY['bit_manipulation'] = 0
│   (skip the bit_manip category entirely). Compare to v12.
│   - If v12-baseline > v12: our substitute IS hurting; v13 must improve
│     bit_manip data quality (distilled CoTs from a stronger model, or
│     restructure our solver's reasoning text).
│   - If v12-baseline ≈ v12: our substitute is neutral; the gap to 0.85
│     is from elsewhere (recipe drift, val split, test set timing). v13
│     should leave bit_manip alone and look at other levers.
│   - If v12-baseline < v12: our substitute IS helping (even if not as
│     much as Kh0a's). v13 should make it stronger.
│   Cost: another 6-10 hr GPU run, but produces clean attribution.
│
├── 0.70 - 0.79: SAME CONFOUND but more severe. Same isolation experiment
│   plus a v12 diagnostic eval kernel run (like v9's) to see which
│   categories are failing most. Two diagnostic data points before v13.
│
└── < 0.70: Something fundamental broke. Likely culprits:
    - private data dependencies we didn't fully patch (splited.csv path?)
    - tokenizer / chat-template mismatch
    - degenerate bit_manip traces on edge cases (our wrapper has fallbacks
      but they may produce empty or repetitive strings)
    - Unsloth or trl version drift vs Kh0a's notebook
    DEBUG the notebook output before any new training run.
```

### What we explicitly accepted by compressing this into one run

A clean ablation design would have been:
1. **v12-A:** Kh0a recipe, bit_manip dropped → tests recipe alone
2. **v12-B:** Kh0a recipe + our substitute → tests recipe + substitute
3. Δ(B − A) = effect of our substitute, isolated
4. A vs Kh0a's claimed 0.87 = recipe-reproducibility effect, isolated

That's two training runs (~12-20 hr total) instead of one. We chose the
single run to compress to ~6-10 hr because compute is finite and we
wanted ONE actionable data point first. **This is a defensible tradeoff but
NOT a free experiment — the confound is the cost.** If v12 is ambiguous,
we owe ourselves the isolation experiment before any v13.

The disciplined practice for the methodology writeup: name the confound,
explain why we accepted it, document the isolation experiment we'd run if
the result is ambiguous. Don't pretend a confounded result is clean.

---

## PART I — Strategic pivot 2026-05-18: Kh0a → dgxchen foundation

After v12 trained in 3.6 hr and scored LB 0.72 (well below Kh0a's claimed 0.87),
we tested dgxchen's public adapter directly via a CPU-only wrapper kernel. It
scored **0.84** vs his claimed 0.85 — proving the test set hasn't drifted and
dgxchen's recipe IS reproducible.

This empirically rules out test-set drift as the v12 failure cause. The Kh0a
recipe genuinely doesn't reproduce for us without his private CSVs (which we
cannot access).

### Decision: build v13 on dgxchen, park Tong v20 banking

We explicitly chose NOT to bank Tong v20 (LB 0.86) via the asalhi wrapper,
despite it being one click away. Reasoning:

1. **Methodology integrity:** ~700 teams have already submitted the identical
   adapter via asalhi. Banking it puts us in a crowded cluster with no
   contribution to the methodology story. The Methodology Award (our secondary
   target per PLAN.md) rewards original work, not score-banking.

2. **Building requires source, not just weights:** Tong v20 is a black-box
   adapter. Tong's public GitHub terminates 2026-04-27 — five days BEFORE the
   prize. We have the v18-era source (LB 0.85) but the v18→v20 jump is
   undocumented. Fine-tuning ON TOP of v20 without knowing his recipe risks
   catastrophic forgetting. dgxchen's recipe is fully transparent
   (public adapter + public notebook + public CoT data + documented hyperparams)
   so we can safely extend it.

3. **Sufficient floor already banked:** dgxchen 0.84 is in our submissions
   pool. v9 at 0.62 and v12 at 0.72 are also banked. If v13 totally fails,
   our 2 final picks for judging would be dgxchen 0.84 + best-of-our-pool —
   still a respectable competitive submission. The 0.02 LB we "lose" by not
   banking Tong is the cost of competitive integrity.

4. **dgxchen as v13 starting point — REVISED design 2026-05-19:** Original
   plan was to swap our solver CoTs into dgxchen's corpus. After downloading
   his `problem_ids_matched.csv` and inspecting actual CoT content, we found:
   - dgxchen's **bit_manipulation CoTs use bit-serial decomposition** (the
     98.9% accuracy technique flagged by Gemini's research). Likely STRONGER
     than our v3 rule-based solver.
   - dgxchen's cipher CoTs use thorough character-mapping; likely on par
     with our cipher v2 wordlist fallback.
   - dgxchen's gravity/unit_conv use long-arithmetic; comparable to ours.

   **REVISED v13 = dgxchen recipe + BOXED_LOSS_WEIGHT=5.0 only, NO DATA
   CHANGES.** Single-variable test: does Kh0a's key innovation transfer to
   dgxchen's foundation? Cleanest attribution. If yes (≥0.86), v14 can layer
   additional deltas. If no (≤0.84), v14 reconsiders or tries our cipher v2
   swap as a different delta.

   Our solvers are NOT used in v13. They were the foundation for v1-v9
   (peaked LB 0.62). Their primary future role is Phase 2 (RLVR/GRPO)
   verifier rewards — see PLAN.md "Where do OUR solvers fit?" section.

### Reproducibility comparison (the critical strategic distinction)

| Path | LB | Bank-reproducible? | Build-reproducible? |
|---|---|---|---|
| Tong v20 via asalhi | 0.86 | YES (1-click wrapper) | **NO — black box** (no v20 training notebook) |
| dgxchen | 0.84 | YES (verified) | **YES — fully transparent** |
| Kh0a | claimed 0.87 / got 0.72 | N/A (no public adapter) | Partial (notebook public, but private CSV deps hurt) |

### Final-submission strategy (we pick 2 for judging)

```
BEST CASE:    v13 (0.85+) + dgxchen direct (0.84)  ← both reflect our work
GOOD CASE:    v13 (0.84-) + dgxchen direct (0.84)  ← still both ours, mediocre
WORST CASE:   v12 (0.72)  + v9 (0.62)              ← Hail Mary territory
EMERGENCY:    Bank Tong v20 → 0.86 + best of pool   ← only if v13 totally fails
```

The EMERGENCY path remains available — banking Tong v20 takes ~10 min via the
asalhi wrapper. Triggered only if v13 catastrophically regresses below
dgxchen's 0.84, AND we're running out of time before 2026-06-15 deadline.

### Open questions before v13 design

- v12 eval diagnostic (still running on Kaggle as of 2026-05-18) will tell us
  per-category accuracy on the Kh0a-trained adapter. Most useful signal: which
  categories OUR solvers might beat dgxchen's generators. Inform v13 delta
  selection.
- Once v12 eval lands: write up v13 design doc with specific category swaps,
  push training kernel, ~3-4 hr wall on RTX Pro 6000.

---

## PART H — Compute strategy (updated post-Gemini-research, 2026-05-17 night)

A previously-held claim (echoed in user memory: "Colab Pro is useless for comps
requiring A100/H100/RTX Pro 6000") is **partially wrong**. Per Gemini's deep
research citing Kaggle's official blog (level-up-your-compute-more-gpu-hours-
on-kaggle-wit), the temporal-quota bonus from Colab Pro/Pro+ applies 1:1 to
**whatever hardware Kaggle gives that competition** — including the sponsored
RTX Pro 6000 in the Nemotron comp.

Verbatim from Gemini's report: *"An hour of computation on the Blackwell
architecture consumes exactly one hour of the Kaggle temporal quota, avoiding
the accelerated burn rates seen in CCU-based accounting systems."*

| Tier | Base | Bonus | Total weekly | Cost |
|---|---|---|---|---|
| Free | 30 | 0 | 30 hr | $0 |
| Colab Pro | 30 | +15 | 45 hr | $9.99/mo |
| Colab Pro+ | 30 | +30 | **60 hr** | $49.99/mo |

The reconciliation: my memory was correct for **other Kaggle
competitions** (Colab Pro+ just buys T4/P100 hours there). But the RTX Pro 6000
is sponsored-compute specifically gated to the Nemotron dataset; the temporal
bonus applies on top of that hardware. My memory was right in general
but wrong for this specific comp.

**API quirk noted:** the integration is "occasionally unstable" and quotas can
revert to 30 hr without warning. Documented fix: navigate to Kaggle settings,
unlink the Google account, re-authorize via "File → Link to Colab" in the
notebook editor. References: Kaggle discussion threads /663030, /663066,
/682732.

### What this means for our v12 real run

After smoke-testing confirmed the pipeline works, we measured ~14 sec/optimizer-
step. Real run projection at Kh0a's full config:
- 3,910 records × 2 epochs / batch 2 = 3,910 optimizer steps
- × 14-25 sec/step = **~15-22 hr training + ~1.7 hr in-training validation = 17-24 hr total**

Our remaining quota this week is ~12 hours. The real run does NOT fit.

Options on the table (pending user decision):
- **Colab Pro+ ($49.99/mo)** → 60 hr/wk total. Fits the v12 real run + enables 2-3 runs/wk through 2026-06-15 deadline. Cancellable any time.
- **Colab Pro ($9.99/mo)** → 45 hr/wk. Fits v12 real run + maybe one follow-up.
- **Wait for next-Friday reset (~3 day delay)** → 30 hr/wk fresh. Tight fit for one run, no margin for the v12-baseline isolation experiment described in Part G.
- **Cut v12 to fit in 12 hr** → NUM_EPOCHS=1 (cleanest) or halve training data (less clean); adds one more deviation to the v12 composite-hypothesis confound described in Part G.

### Algorithmic ideas surfaced by Gemini's research (worth investigating)

These are NEW ideas (not in our existing playbook); validity unverified, citations are Kaggle discussion threads.

1. **Bit-serial CoT for bit_manipulation (claim: 98.9% accuracy vs our 89.1%)** —
   Cite: discussion/693260. The technique forces the CoT to spell out
   operations one bit at a time (`Bit 0: 1&1=1`, `Bit 1: 1&0=0`) instead of
   describing rules at a higher level. Gemini says this produces
   "hyper-compressed and highly deterministic" gradient signal. Worth a deep
   read before betting compute. If real, +10% bit_manipulation coverage could
   substantially improve v13 corpus quality. Our current v3 traces use
   rule-level descriptions, not bit-level walks.

2. **Muon optimizer ("Momentum Orthogonalized") vs AdamW** — Cite: tascj0
   writeups. Gemini calls it "massive performance improvements" but cites no
   specific gain. Vague; worth investigation only if we have the bandwidth.

3. **Duplicate trace trap (gradient collapse risk)** — Cite: discussion/693260.
   If a CoT generator produces near-identical reasoning text across many
   problems, the model bypasses CoT entirely and accuracy collapses (claim:
   90% → 46.5%). **Action: verify our solvers don't have this.** Our gravity
   and unit_conversion long-arithmetic traces share substantial structure
   ("We need to determine the falling distance...", same opening phrasing
   across 1,022 records). Spot-check on first 100-200 tokens of our v11
   corpus is a fast check, no compute needed.

4. **"Loss illusion" — pre-tokenized prompt masking** — Cite: discussion/694710.
   Default SFTTrainer averages CE loss over BOTH prompt and response tokens,
   so the loss plateaus around 2.5 (the model can't predict random prompt
   text) with chaotic gradients that cause catastrophic forgetting. Solution:
   binary mask 0 for prompt tokens, 1 for response. **Action: verify Kh0a's
   notebook handles this.** Their `apply_chat_template(add_generation_prompt
   =False, enable_thinking=True)` + trl SFTTrainer may or may not auto-mask;
   needs source-level check before v12 real run.

### Gemini's confirmation of things we already did

- Cut Cross-Entropy reduces 11hr → 5h40m → ✓ adopted in v8
- Multi-epoch training needed because 1 epoch undertrains → ✓ v9 finding
- MoE LoRA weight tying mentioned as a fix → ✗ tested cleanly in v6/v7,
  found null at our data scale (loss curves indistinguishable). Possibly
  scale-dependent — Gemini's claim might hold at Tong's 32k corpus scale
  but not our 4.7k v3-corpus scale.
- Programmatic solver-generated CoT is the way → ✓ our entire approach

### Decision: subscribed to Colab Pro ($9.99/mo)

Updated 2026-05-18. User subscribed to Colab Pro (not Pro+). New weekly
quota: 45 hr/wk. With ~12 hr remaining for the week, we now have ~27 hr
usable — fits v12 real run (17-24 hr) with a small margin.

### Pre-flight audit before pushing v12 real (2026-05-18)

Two zero-compute checks performed before flipping `SMOKE_TEST = False`:

#### Loss masking (Gemini's "loss illusion" concern)

Inspected cells 46-47 of Kh0a's notebook directly. Findings:

- Kh0a uses `dataset_text_field="text"` with `apply_chat_template(
  add_generation_prompt=False, enable_thinking=True)`. The whole
  prompt+response text becomes the training target — **no explicit
  prompt-token masking via labels=-100**.
- Their custom `compute_loss_func` mask `(shift_labels != -100)` is
  effectively all-1s in this setup, because SFTTrainer with
  `dataset_text_field` doesn't insert -100 for prompt tokens.
- **The mitigation is the 5× BOXED_LOSS_WEIGHT upweight** on tokens
  after the last `\boxed{` marker. Answer tokens dominate the gradient
  even though prompt tokens still contribute noise.

So Gemini's loss-illusion concern is REAL for default SFTTrainer setups,
but Kh0a handles it differently — via weighting, not masking. Kh0a
reaches 0.87 with this design, so it works. v12 inherits Kh0a's design
exactly. **No fix needed for v12; deferred consideration for v13+.**

#### Trace diversity (Gemini's "duplicate trace trap" concern)

Audited our v11 corpus (`data/training-data.jsonl`, 7,436 records):

| category | first-50-chars distinct | first-100-chars distinct | first-200-chars distinct |
|---|---|---|---|
| bit_manipulation | 33% (475/1428) | **91%** (1301/1428) | 100% |
| cipher | **0.1% (1/1576)** | **0.1% (1/1576)** | 100% |
| gravity | **0.1% (1/1022)** | **0.1% (1/1022)** | 100% |
| unit_conversion | **0.1% (1/1328)** | **0.1% (1/1328)** | 100% |
| transformation_rules | **0.2% (1/506)** | **0.2% (1/506)** | **0.2% (1/506)** ⚠️ |
| numeral_system | 6.3% (100/1576) | 100% | 100% |

**Critical context: only `bit_manipulation` traces from our solver feed
into v12.** Kh0a uses HIS own trace generators (inline in his notebook
cells 18, 20, 22, 24, 28, 30) for the other 5 categories. So our solver's
identical-opener problem doesn't manifest in v12 because those records
aren't being used.

**bit_manipulation is the only category we contribute to v12, and it's
diverse enough:**
- 475 distinct first-50-char openings across 1,428 records
- 1,301 distinct first-100-char openings (91% diverse, max repeat 12×)
- 100% diverse by 200 chars

Why bit_manip is naturally diverse: the trace opens with the discovered
rule (`The transformation is \`SHL(1) XOR NOT SHL(7)\``) — the rule string
varies per problem, so diversity comes free from token 0.

**Verdict for v12: no duplicate-trace-trap risk.** Pushed for real.

#### Action items for v13+

If we ever use our cipher/gravity/unit_conversion/transformation_rules
trace generators alongside or instead of Kh0a's, the 100%-identical
first-100-chars per category IS a risk. Particularly bad:
`transformation_rules` (all 506 records share the same first 200 chars).

Fix before using those records in training:
- Rotate 3-5 opening phrases per solver
- Move problem-specific content (example pairs, query values) earlier
- Or hash-based selection of opener variant by problem_id (deterministic
  but varied)

This is v13-or-later work — not blocking the v12 real run.

---

## Key references (file paths and URLs)

Local:
- `reference/tonghuikang-nemotron/` — Tong's prize-frozen GitHub repo
- `reference/huikang-0.85-lb/end-to-end-finetuning-for-lb-0-85.ipynb` — Tong's prize notebook
- `kaggle/v12-kh0a-baseline/` — our patched copy of Kh0a's notebook
- `/tmp/dgxchen-recipe/` — pulled copy of dgxchen's training notebook
- `scripts/solvers/*.py` — our 6 solvers + shared arithmetic helpers
- `data/training-data.jsonl` — v11 corpus (7,436 records)

Web:
- Comp: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- Tong's adapter: https://www.kaggle.com/models/huikang/nemotron-adapter (v20 == 0.86)
- asalhi wrapper: https://www.kaggle.com/code/asalhi/tinker-adapter-to-ready-to-submit-adapter
- dgxchen recipe: https://www.kaggle.com/code/dgxchen/training-with-unsloth-to-achieve-0-85-lb
- Kh0a precursor: https://www.kaggle.com/code/llkh0a/nemotron-unsloth-sft-training-3-30-2
- Tong's writeup: https://blog.huikang.dev/2026/05/02/nemotron-progress-prize.html

Our LB submissions (all private): https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/submissions
