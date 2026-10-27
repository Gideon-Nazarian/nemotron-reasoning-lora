# Strategy / North Star

> Working strategy log from the competition (May to June 2026). Internal notes, lightly trimmed for publication. The retrospective is in [`REPORT.md`](./REPORT.md) and the run ledger in [`EXPERIMENTS.md`](./EXPERIMENTS.md).

## What we're optimizing for

1. **Learning + portfolio** (primary). Build hands-on experience with modern LLM inference, fine-tuning, eval discipline — the stack DS programs don't cover.
2. **"Best Fine-tuning Method" Open Contribution Award** (realistic secondary target). 1 DGX Spark prize, judged on methodology rather than leaderboard rank. Aligns with our discipline-first approach: clean methodology, reproducibility, error analysis, well-documented experiments. Achievable without competing for top-3 absolute scores.
3. **Leaderboard placement** (stretch). Treated as a real-contender effort from Day 1 so the work transfers cleanly if we push for it.

## Prize structure (for context)

- 1st: $25,000 + 5 DGX Sparks
- 2nd: $15,000 + 2 DGX Sparks
- 3rd: $5,000 + 1 DGX Spark
- Open Progress Prize: $5,000 + 1 DGX Spark (already awarded 2026-05-02 to Tong Huikang)
- **Open Contribution Awards (1 DGX Spark each)** — judged on methodology, not leaderboard:
  - Best Data/Synthetic Data Method
  - Best RL Method
  - **Best Fine-tuning Method** ← our realistic target
- Total prize pool: $106,388

## Constraints

- **Budget:** $0 out of pocket. Compute = Kaggle's free 30 hrs/wk T4×2 + Kaggle's G4 grading infra (free as part of the comp). Modal $30/mo free tier reserved for fine-tuning bursts in Week 3.
- **Deadline:** 2026-06-15 23:59 UTC (final submission). Confirmed via Kaggle CLI on 2026-05-09 → ~37 days runway.
- **Field size:** 2,850 teams (per Kaggle CLI on 2026-05-09).

## Operating disciplines (from rules review)

- **Repo kept private during the competition.** Section 3.6.a prohibits *multi-party* code sharing while the competition runs; single-owner private storage is compliant. No collaborators were added until after the deadline.
- **OSS licenses verified clean** — `transformers` (Apache-2.0), `torch` (BSD), `kaggle` (Apache-2.0), `peft` (Apache-2.0), `accelerate` (Apache-2.0). Nemotron weights under NVIDIA Open Model License (the comp's intended path).
- **Single Kaggle account, max 5 submissions/day, 2 final submissions selected for judging.**

## Discipline rules (non-negotiable)

1. **Every submission notebook is `git commit`-ed first.** Short hash goes in `EXPERIMENTS.md`.
2. **Seeds are set to 42** at top of every notebook (use `scripts/seeds.py` locally; inline it in Kaggle notebooks).
3. **Submissions are the primary iteration loop. Local eval is a diagnostic instrument, not a gate.** (Updated 2026-05-09 after v3 eval revealed local strict-extraction massively underestimates model capability vs. comp's lenient grading. Cost asymmetry: ~14 min/submission vs ~96 min/eval. Eval reserved for failure-mode analysis when LB results are unexpected.)
4. **Error analysis is mandatory** after every submission. Pull 5-10 wrong predictions (or read raw completions saved by eval kernel), categorize failure modes, log them in the takeaway.
5. **No content duplication across working docs.** PLAN/NEXT/EXPERIMENTS each have one job.
6. **Smoke-test ANY new architecture or major notebook fork** before committing real-run compute. Cost: 5-10 min. Saves: potential 1-2+ hr quota burn on env/dependency issues. (Added 2026-05-19 after v13 burned ~30 min of quota silently on a stuck kernel that a smoke test would have caught immediately. v12 had this discipline because we baked SMOKE_TEST=True into Kh0a's notebook; v13 dropped it because dgxchen's notebook "looked simpler" — exactly the wrong calibration. Different docker images, different package versions, different cell structures = each fork is a new architecture from a smoke-test perspective.) Implementation: add a `SMOKE_TEST = True` flag at the top of training config that shrinks dataset to ~20 records and caps `MAX_STEPS = 3-5`. Push with True first; only flip to False after smoke passes end-to-end.

7. **When forking notebooks across environments (Kaggle ↔ Colab), the install cell is environment-specific** — never copy it verbatim, always reference the destination's working sibling notebook. (Added 2026-05-21 after v14-c-train v1 crashed at Cell 6 `import unsloth` because I'd carried over v13-colab.ipynb's Cell 2 — which only installs Mamba wheels because Colab has PyPI for the training stack. Kaggle is offline, so it needs unsloth/trl/peft/transformers/datasets/accelerate/bitsandbytes from mayukh18's `/packages` wheelhouse. v12's Kaggle notebook had the correct pattern; I should have started from it. Fix in v3 mirrored v12's Cell 2 exactly.) **Discipline rule:** before pushing a kernel to a new environment, diff your Cell 2 / install cell against the most recent successful notebook in that environment. Don't trust pattern memory across env boundaries.

8. **Kaggle CLI cannot control GPU type — backend dispatch is best-effort and undocumented** (verified 2026-05-21 by reading kagglesdk source). The public SDK's `ApiSaveKernelRequest` has `enable_gpu: bool` only; no field for GPU type, no `machine_shape`, no `docker_image`. Those fields in `kernel-metadata.json` are silently ignored. Empirically, the backend uses undocumented heuristics — some CLI pushes auto-run on RTX Pro 6000 (likely if the slug previously had RTX Pro 6000 selected in the UI, or if your account has recent RTX Pro 6000 activity), others land on free-tier P100. **Workflow:** (1) `kaggle kernels push`, (2) glance at the auto-run's GPU output — if P100, Cell 1 fail-fast guard catches in ~30 sec then open kernel URL → Accelerator → select RTX Pro 6000 → Save & Run All; if RTX Pro 6000, just monitor logs. **Cell 1 fail-fast guard is mandatory in every training/eval notebook** — costs ~30 sec on a wrong dispatch, saves us from 5+ min of doomed model load. (Source files for future verification: `kaggle/api/kaggle_api_extended.py::kernels_push`, `kagglesdk/kernels/types/kernels_api_service.py::ApiSaveKernelRequest`, `kagglesdk/kernels/services/kernels_api_service.py`. `kaggle.json` vs newer `credentials.json` auth is unrelated to GPU dispatch — both give identical API permissions.)

9. **HARD RULE — account for the FULL `/kaggle/working/` footprint, not just the line-item you're changing. Install Python deps to `/tmp/`, never `/kaggle/working/`.** (Added 2026-05-28 after v15 — the SECOND disk loss — destroyed a fully-completed 6.85-hr training run at the final-checkpoint save.) Kaggle commits `/kaggle/working/` as kernel output with a ~20 GB cap. **Everything we write there for the whole run counts**, including the `pip install --target /kaggle/working/site-packages` from Cell 2 (~10-13 GB once nvidia CUDA transitive deps resolve). The v16 "fix" (`save_only_model=True`, `save_total_limit=1`) only shrank the checkpoint line-item; the ~13 GB site-packages ballast went uncounted and v15 blew the cap at final save (site-packages ~13 + checkpoint-200 not-yet-rotated ~3.5 + checkpoint-245 being written ~3.5 = >20 GB → 0-byte safetensors → total loss). **Three-part rule going forward:**
   - (a) `pip install --target /tmp/nemo-pkgs` (NOT `/kaggle/working/...`) + `sys.path.insert(0, "/tmp/nemo-pkgs")`. `/tmp` is scratch on the container root fs (~57-73 GB total) and is NOT part of the 20 GB output commit. Frees ~13 GB instantly.
   - (b) The SMOKE TEST MUST EXERCISE THE SAVE PATH. v15's smoke used `save_strategy='no'` and skipped `submission.zip` — so it never touched the disk-at-save code that killed the real run. A LoRA adapter after 3 smoke steps is the SAME size as after a full run (param count is fixed), so the smoke can and must do Cell 12 (save adapter) + Cell 13 (build zip) and print `df -h /kaggle/working` after. If the smoke can't save+zip within disk, neither can the 7-hr real run — and we learn it in 15 min, not 7 hr.
   - (c) Print `df -h /kaggle/working /tmp` at kernel start AND after every save, so the headroom is visible in the log. Never run a multi-hour job whose disk math we haven't seen confirmed empirically.

## Weekly roadmap

### Week 1 — Eval discipline + problem understanding
- **Day 1 (today):** scaffold + zero-shot baseline submission
- **Day 2 (priority):** local eval harness — splits public data into hold-out, computes comp metric offline
- Read NVIDIA Nemotron 3 blog, Tong Huikang's progress-prize writeup, top 3 public notebooks
- 2-3 prompt-engineering submissions (CoT, format constraints, role priming) — gated on local eval improvement
- **Recurring weekly:** scan competition discussion forum
- **Goal by EOW1:** 3-5 disciplined entries in EXPERIMENTS.md, dominant failure modes characterized

### Week 2 — Inference-time engineering
- Self-consistency (multiple samples + majority vote)
- Structured output / JSON-schema constrained generation
- Verifier prompts (second-pass grading)
- Decoding hyperparameters (temperature, top-p, n_samples)
- **Goal:** Material LB movement from prompting alone before touching weights

### Week 3 — Synthetic data + LoRA fine-tuning
- Generate synthetic reasoning traces (Nemotron itself or a stronger model)
- Curate / filter the synthetic corpus
- QLoRA fine-tune Nemotron 3 Nano — Modal $30 free credit gets used here (one H100 burst)
- Add Weights & Biases (free personal tier) for run tracking
- **Goal:** Full ML pipeline (data gen → curate → train → eval → submit) end-to-end

### Week 4 — Iterate + finalize
- Best approach from W1-3, push 2-3 tuning iterations
- Stop new directions ~3 days before deadline (avoid last-minute regression)
- Final submission with safety margin
- Polish public artifacts: GitHub README, optional blog post

---

## Trajectory log

Records actual progression vs. the speculative Week 1-4 roadmap above. The roadmap stays as the original plan; this section captures pivots.

**2026-05-09 (Day 1):** Baseline submitted, LB 0.500. Discovered comp uses adapter submission, not predictions, so the pipeline pivoted to LoRA-SFT-on-adapter early. Speculative Week 2 "inference-time engineering" was deprioritized — direct fine-tuning had higher expected value once we saw the adapter-submission format.

**2026-05-10 (Day 2):** Skipped Week 2 entirely. Built per-category programmatic solvers (Path A from `notes/competition-rules.md` — write our own data generators rather than rely on synthetic data tools), generated 4681 clean training examples across 5 of 6 puzzle categories, ran v1 LoRA SFT on 750 stratified records → LB 0.510 (+0.010). Diagnostic eval revealed template overfit: model learned format perfectly but fabricated values inside the template (notes/early-diagnostic-evals.md).

**2026-05-10 (Day 2, continued):** Rewrote solver reasoning text to include explicit per-example arithmetic chain-of-thought ("Fix A"). v2 retrained on same 750 stratified records but with new reasoning → LB 0.540 (+0.030). Confirmed Fix A worked at the mechanism level via v7 diagnostic: cipher now decodes letter-by-letter, gravity computes g per-example, bit_manipulation enumerates operations correctly (notes/early-diagnostic-evals.md).

**2026-05-11 (Day 3):** Scaled v3 to full 4681-record corpus (same Fix A reasoning, same hyperparams) → LB 0.580 (+0.040). Each successive fix has produced a bigger lift than the prior, suggesting we're not yet at diminishing returns.

**2026-05-11 (Day 3, continued):** Triaged Branch A (transformation_rules) and Branch B (bit_manipulation) — neither cleanly tractable as programmatic solvers. Attempted v4 (gravity format fix to preserve gold's decimal precision, +114 training examples) → **LB regressed 0.58 → 0.56 (-0.02)**. v9 diagnostic on v4 showed identical holdout aggregate but per-problem behavior shifts on unrelated problems — diagnosis: training-data perturbation, not format confusion. **Reverted gravity solver. v3 (LB 0.58) confirmed as SFT high-water mark.** Next major lever pivots to RLVR (Branch D) — RL gradient updates target specific failures via reward signal, constraining the ripple-effect risk that hurt v4.

**Key lesson from v4:** more-and-still-correct training data is NOT a monotonic lever. Even "obviously correct" additions can degrade aggregate performance via subtle behavior shifts on unrelated problems. Holdout aggregate is necessary but not sufficient evidence for "is this change net-positive."

**Gap to Tong Huikang's progress prize (0.850): 0.27 remaining.** Path forward: RLVR via NeMo RL framework (NVIDIA-provided guide for Nemotron 3 Nano).

**2026-05-13 to 2026-05-15 (Days 5-7):** GRPO/RLVR attempt blocked by Kaggle env compatibility (torchvision CUDA version mismatch from a `ryanholbrook/nvidia-utility-script` update). Pivoted off the GRPO path to a synthetic-data approach: v5 added 800 synthetic transformation_rules records (the previously 0%-covered category) to v3's solver corpus. **v5 regressed -0.02 to LB 0.56** — the SAME regression magnitude as v4. Revised lesson: it's not just "perturbing covered categories" that hurts — ANY data shape change costs ~0.02. v3 sat at a local optimum for our current data shape.

Re-studied Tong Huikang's 0.85 LB methodology. Identified MoE expert weight tying as the likely critical architectural piece we've been missing.

**2026-05-15 to 2026-05-16 (Days 7-8):** Adopted Tong's architectural pieces incrementally.
- v6 (MoE "tying" + 9 target modules + alpha=r + dropout=0 + AdamW betas): **LB 0.58** (flat). Post-run diagnostic revealed our tying logic had a silent bug (PEFT created per-expert SEPARATE LoRA modules `(rank, dim)` shape; our `mean(dim=0)` downgraded experts to rank-1).
- v7 (corrected per-expert grouped tying + pre-training verification): **LB 0.58** (flat). v6 vs v7 loss curves indistinguishable. **Clean negative result: MoE expert tying is NOT the missing architectural piece.**
- v8 (full Unsloth + Cut Cross-Entropy stack, 1 epoch batch 32): **LB 0.56** (regressed). Diagnosis: undertrained — 146 steps too few for the bigger batch size despite same data exposure.

**2026-05-17 (Day 9): BREAKTHROUGH.** v9 = v8 stack × **3 epochs** (438 steps). **LB 0.62, +0.04 vs v3 baseline.** First SFT lift in 6 submissions. Diagnosis: undertraining was the real bottleneck, not architecture. The Unsloth/CCE/9-module stack only pays off when given enough total gradient updates.

**Key methodological learning:** training loss flatness ≠ training done. v9's loss looked converged by step 30 yet the moving-average across epochs continued drifting down (0.025 → 0.019 → 0.015) and LB jumped substantially. Visual inspection of training loss alone is misleading; LB submissions are the only ground truth.

**2026-05-17 (Day 9, evening): STRATEGIC PIVOT — stand on shoulders.** While solver improvements were landing (bit_manip 9.4%→89.1%, cipher 38.4%→100%, gravity/unit_conv long-arithmetic, new transformation_rules 0%→32.5%, corpus 4,681 → 7,436), a research agent investigated the leaderboard topology and found the picture changed entirely. Of ~3,200 submissions, ~700 cluster at LB **0.86** — that's **Tong Huikang's adapter version 20** (a post-prize iteration he kept publishing, not in any writeup) via the public `asalhi/tinker-adapter-to-ready-to-submit-adapter` wrapper notebook. The top 4 teams at 0.87 are private; only Kh0a (rank 4) has a public precursor notebook that reveals the recipe: category-rebalanced sampling + custom CoT traces + `BOXED_LOSS_WEIGHT=5.0` (5× CE upweight on tokens after `\boxed{`). The 0.86→0.87 gap is data and loss-weighting, not architecture.

**Decision:** cancelled v10 (5-epoch v9 stack) and v11 (expanded solver corpus on v9 stack). Both produce adapters whose plausible ceiling (0.65-0.75) is well below the public 0.86 floor. Pivoted to v12 = fork Kh0a's public training notebook, patch its missing `generate_bit_manipulation_trace` function (their public notebook has a copy-paste bug; their private CSV cached every bit_manip trace so the fallback never fired in their runs) with our v3 bit_manipulation solver inline. The patch is a clean substitution: our solver's 89.1% coverage and Tong-style reasoning matches the shape of Kh0a's other in-notebook trace generators.

**Compliance: explicitly compliant.** PLAN.md operating-discipline forbids multi-party private collaboration; using PUBLIC notebooks/datasets (Tong's adapter is Apache-2.0, asalhi/Kh0a notebooks are CC0) is allowed — same as how we already depend on public `mayukh18/nemotron-packages` and `llkh0a/rtx-wheels`.

**Current state (2026-05-17 evening):** 11 LB submissions (v9 still our adapter HWM at 0.62; never submitted v10/v11). v12 calibration kernel `gideonnazarian/v12-kh0a-baseline-calibration` is training on Kaggle RTX Pro 6000 (~6-10 hr wall). Two final-submission slots for judging: one earmarked for asalhi-via-Tong-v20 (banks 0.86, fork+submit), the other for whatever's best from v12 + future v13.

**See `notes/lb-intel-and-recipes-2026-05-17-evening.md` for the comprehensive synthesis** of all leaderboard intel, full published recipes (Tong prize, dgxchen 0.85, Kh0a 0.87-style), the direct contradictions between them, every experiment we've tried with outcomes, and explicit documentation of which experiments are KNOWN regressive (so future sessions don't waste 10-hour GPU runs re-testing them).

**2026-05-18 (Day 10): v12 → 0.72; dgxchen reproduction → 0.84; strategic pivot to dgxchen foundation.**

v12 (Kh0a recipe + our v3 bit_manip patch) trained in 3.6 hr (vs my projected 17-24) and scored **LB 0.72** — significant lift over v9's 0.62 but well below Kh0a's claimed 0.87. The 0.15 gap is consistent with Kh0a's private CSV dependencies (bit_manipulation_traces_v4.csv, numeric_equation_traces_new.csv, splited.csv) doing much more unmeasured work than our pre-run estimate suggested.

We then submitted `dgxchen/trained-adapter` directly via a CPU-only wrapper kernel — **LB 0.84** (vs his claimed 0.85; -0.01 essentially noise). This empirically proves: (a) the comp's test set has NOT drifted, (b) dgxchen's recipe IS reproducible, (c) the Kh0a-path gap was real recipe issues, not test-set drift.

Reproducibility comparison (the key strategic distinction):

| Path | LB | Bank-reproducible? | Build-reproducible? |
|---|---|---|---|
| Tong v20 via asalhi | 0.86 | YES (just submit) | **NO — black box** (no public training notebook for v20; Tong's GitHub terminates 2026-04-27, 5 days BEFORE prize, source code post-prize is unpublished) |
| dgxchen | 0.84 | YES | **YES — fully transparent** (public adapter + public notebook + public CoT data) |
| Kh0a | claimed 0.87 / got 0.72 | N/A (no public adapter) | Partial (notebook public, private CSV deps hurt reproducibility) |

**Decision 2026-05-18: build v13 on dgxchen's foundation. Park Tong v20 banking as emergency-only fallback.**

Reasoning: (1) banking Tong v20 produces a submission identical to ~700 other teams clustered at 0.86, providing zero methodology contribution; (2) the Methodology Award (our secondary target per PLAN.md) rewards original work, not score-banking; (3) dgxchen's 0.84 is already a sufficient banked floor in our submissions pool; (4) building on dgxchen with our own documented solver deltas is what differentiates our entry. The 0.02 LB we "lose" by not banking Tong is the cost of competitive integrity — small and worth it.

**v13 plan (REVISED 2026-05-19 after inspecting dgxchen's data):** dgxchen's training notebook + **a single non-data change** — Kh0a's `BOXED_LOSS_WEIGHT=5.0` compute_loss_func grafted onto dgxchen's SFTConfig. NOTHING else changes: same data (`dgxchen/nemotron-cot-tong`, 7,830 records), same LoRA config (r=32 α=32 dropout=0, 8 target modules no lm_head), same 1 epoch, batch 32, LR 2e-4, stratified-by-type batching.

**Why we REVISED away from "swap our solver CoTs":** the original plan was to replace dgxchen's CoTs for cipher/bit_manip/transformation_rules with ours. On 2026-05-19 I downloaded dgxchen's `problem_ids_matched.csv` and inspected the actual CoT content. Found:

- His **bit_manipulation CoTs use bit-serial decomposition** (`Output 0: 10111101 \n 0 1 \n 1 0 \n 2 1 ...`) — this matches Gemini's research-flagged "98.9% accuracy" technique. **Likely STRONGER than our v3 rule-based solver.**
- His **cipher CoTs use thorough character-by-character mapping** with explicit Wonderland-style word list iteration — likely comparable to ours.
- His **gravity/unit_conversion CoTs use long-arithmetic** (multi-step multiplication with intermediate values shown) — comparable to ours.

Conclusion: blindly replacing his potentially-stronger CoTs with ours risks regression. v13 is therefore a **single-variable test** — does Kh0a's BOXED_LOSS_WEIGHT innovation transfer to dgxchen's recipe? Cleaner attribution, lower risk.

Target: ≥0.86 (which would beat dgxchen's 0.84 baseline by +0.02, consistent with Kh0a's claimed delta over dgxchen).

### Phase 1 → Phase 2 pivot rule (pre-committed 2026-05-18)

To avoid sunk-cost drift in the dgxchen architecture family, we pre-commit a hard pivot trigger:

```
PHASE 1: dgxchen architecture family (LoRA SFT on Unsloth + Tong CoT data lineage)
  v13: dgxchen + BOXED_LOSS_WEIGHT=5.0 ONLY (single-variable change)     target ≥0.86
  v14: dgxchen + next delta (TBD by v13 result):                         target ≥0.86
       - If v13 lifted: add another Kh0a knob (e.g., lm_head in targets,
         cosine schedule, alpha=16, dropout=0.1)
       - If v13 flat: try our cipher v2 wordlist swap (the one solver we
         have CONCRETE evidence is better than the baseline — v9 eval
         showed it fixes the `mouse`-not-in-examples failure case)
       - If v13 regressed: revert BOXED_LOSS_WEIGHT, try a totally
         different delta

  PIVOT TRIGGER: If BOTH v13 < 0.86 AND v14 < 0.86, declare Phase 1
  architecturally exhausted at our reach. Immediately move to Phase 2.

PHASE 2: Architectural shift — RLVR / GRPO
  v15+: GRPO on best Phase-1 adapter as warm-start, using our existing
        per-category programmatic solvers as the verifiable reward.
        Reference: Dang 2025 (arXiv:2503.16219), our existing scaffolding
        at scripts/rl_reward.py + kaggle/training-rl/10-grpo-smoke-test.ipynb.

PHASE 3: Polish (final 3-5 days before 2026-06-15)
  Stop new experiments. Pick best 2 final submissions. Write up methodology.
```

Why pre-commit: discipline against anchoring. The temptation when v13 lands somewhere between 0.80-0.84 will be to keep iterating dgxchen variants (v14, v15, v16) hoping the next tweak unlocks the ceiling. The pre-committed rule says: if v13 and v14 BOTH fall short of 0.86, the architecture family has been adequately explored at our level of access (no private CSVs, no undocumented data) and we should explore a fundamentally different approach (RLVR) rather than chase diminishing dgxchen-variant returns.

Time budget supports this without rushing: 28 days to deadline × ~3-4 hr per training run × 45 hr/wk Kaggle quota = capacity for 30+ training runs. Phase 1 (2-4 runs) + Phase 2 (3-5 runs) + Phase 3 polish all fit comfortably.

Final-submission pool (we pick 2 for judging):
- dgxchen direct (0.84) ✅ banked
- v12 (0.72) ✅ banked (weaker, backup)
- v9 (0.62) ✅ banked (weakest, backup)
- v13 (TBD; target 0.85+) — what we're building toward

If v13 lands ≥0.85: final picks = best v13 + dgxchen, both reflect our work.
If v13 fails: final picks = dgxchen + v12 (worst case ~0.84). Tong v20 banking remains as a Hail Mary if even that falls through.

### Where do OUR solvers fit? (clarification 2026-05-19)

Quick reference because the trajectory has had several pivots:

| Asset | Status | Used in |
|---|---|---|
| **scripts/solvers/bit_manipulation.py v3** (9.4%→89.1% coverage) | Built ✓ | v12 (Kh0a fork patched our solver in) — scored 0.72. **NOT in v13.** Possibly in v14+ if dgxchen-data swap becomes warranted. Reward signal source for Phase 2 RLVR. |
| **scripts/solvers/cipher.py v2** (38%→100% via Wonderland wordlist) | Built ✓ | v9 (LB 0.62). v9 eval evidenced our wordlist fallback fixes the `mouse`-not-in-examples failure. **NOT in v13.** Strong v14 candidate if v13 doesn't lift via BOXED_LOSS_WEIGHT alone. Reward signal source for Phase 2 RLVR. |
| **scripts/solvers/gravity.py + unit_conversion.py** (long-arithmetic) | Built ✓ | v9. Not directly used in v12 or v13. dgxchen's gravity CoTs are comparable to ours; unclear if swap helps. Reward signal source for Phase 2 RLVR. |
| **scripts/solvers/transformation_rules.py** (0%→32.5%) | Built ✓ | v9. Not in v12 or v13. dgxchen has 4 sub-categories (cryptarithm_deduce/guess + equation_numeric_deduce/guess) with 1,565 records total — likely covers more than our 506. Reward signal source for Phase 2 RLVR. |
| **scripts/solvers/roman_numeral.py** | Built ✓ | v9 corpus. dgxchen also has strong numeral CoTs. Reward signal source for Phase 2 RLVR. |
| **data/training-data.jsonl** (7,436-record solver corpus) | Built ✓ | v9 (LB 0.62). v11 attempted to expand this but cancelled. **NOT used in v12 or v13.** Available for Phase 2 RLVR dataset generation. Worth publishing as a Kaggle Dataset post-deadline as a methodology contribution. |
| **scripts/rl_reward.py + scripts/build_rl_dataset.py** (GRPO scaffolding, 12/12 unit tests pass) | Built ✓ | Phase 2 RLVR (v15+) — uses our solvers as the verifiable reward function. Currently blocked on Kaggle env (torchvision CUDA mismatch from May; revisit after Phase 1 pivot trigger). |

**Summary:** Our solvers are NOT in the active v13 design. They were used heavily in v1-v9 (peaked at 0.62), then we pivoted away to fork stronger public recipes (v12 Kh0a, v13 dgxchen). Their primary future role is **Phase 2 RLVR verifier rewards** — when GRPO trains the model, our solvers programmatically check if outputs are correct, providing the reward signal. This is actually the highest-value reuse: the solvers don't need to BE the training corpus, they need to GRADE the model's outputs during RL.

**2026-05-19 (Day 11): Mamba fast-path saga + Colab Pro G4 migration.** Kaggle quota exhausted (~38:23 / 45 hr used) after v13 v1-v5 attempts burned time on stuck kernels and compilation pathologies. Migrated v13 to Colab Pro G4 (RTX Pro 6000 Blackwell 96 GB, 8.71 CCU/hr). Took 7+ hours of debugging to land the smoke. Major bugs uncovered:

1. **PyPI doesn't ship `causal_conv1d`/`mamba_ssm` wheels for torch 2.10** — pip source-build fails on Colab. Fix: download mayukh18's pre-built wheels from Kaggle dataset `mayukh18/nemotron-packages` via Kaggle CLI from inside Colab. Required separate cell handling kaggle.json upload + wheels install. **Also note:** mayukh18/nemotron-packages includes a `/packages` offline wheelhouse with unsloth/trl/peft/transformers/datasets/accelerate/bitsandbytes — required on Kaggle (no internet) but NOT needed on Colab (PyPI accessible). Canonical Kaggle Cell 2 pattern is in `kaggle/v12-kh0a-baseline/nemotron-unsloth-sft-training-3-30-2.ipynb`. **Forking discipline:** when porting a notebook between Colab and Kaggle, the install cell is environment-specific — never copy it verbatim across environments. Always check against the destination's reference notebook.

2. **Kaggle credential confusion** — Kaggle's new "credentials.json" (OAuth) is not the legacy `kaggle.json` the CLI expects. I had to click "Create Legacy API Key" in account settings. Added defensive content validation in the upload cell.

3. **Mamba fast-path `@lru_cache` sticky-False trap** — `transformers.utils.import_utils.is_causal_conv1d_available()` is `@lru_cache`'d. If ANY code path calls it once before our wheels are visible to Python, it caches `False` forever for the session. Nemotron's `modeling_nemotron_h.py` (loaded via `trust_remote_code`) then skips its `from causal_conv1d import ...` line and binds `causal_conv1d_fn = None`, locking every Mamba block into naive `torch_forward` at `__init__`. **Naive Mamba OOMs at batch=1 seq=8192 on a 96 GB GPU** (allocates ~13 GB chunked outer product). Late module-global monkey-patch failed because each layer cached `use_fast=False` at init. Fix required: restart runtime → ensure wheels installed BEFORE any transformers import touches the helper → verify gate returns True before model load. Now baked into `v13-colab.ipynb` v3 as Cell 4.5 (gate force-patch) + Cell 7.5 (post-load fast-path verify, hard-fails if names are None).

4. **Smoke ≠ real validation lesson** (added Rule #6 to operating disciplines on 2026-05-19): a 3-record smoke validates code paths but not wall-time behavior — real-record compilation pathology only shows up at scale. Also: smoke-test ANY new architecture or notebook fork before committing real-run compute. v13 burned ~30 min of Kaggle quota silently on a stuck kernel that a smoke test would have caught immediately.

End of day 2026-05-19: v13 smoke validated on Colab Pro G4 — Mamba fast path engaged, 64 sec/step steady state, BOXED_LOSS_WEIGHT compute_loss_func accepted by trl 0.24, loss descending 11.26 → 10.01 → 8.92. Cleared for real run.

**2026-05-20 (Day 12): v13 real run → LB 0.84 (flat vs dgxchen).** Real training landed in 6 hr 56 min (per-step 101 sec — slower than smoke's 64 sec because real records average longer than the smoke's 20-record subset). Total Colab cost ~60 CCU (out of ~85 starting budget, leaving ~29 CCU for follow-up smokes). Adapter packaged and uploaded to `gideonnazarian/v13-dgxchen-boxedloss-adapter` (3.02 GB). Submitted via Kaggle CLI directly from the Mac (downloaded the adapter dataset locally, then `kaggle competitions submit`).

**LB grade: 0.840 — identical to dgxchen's direct submission (LB 0.84).** Zero delta. BOXED_LOSS_WEIGHT=5.0 is inert in dgxchen's stack. Two interpretations distinguishable only by diagnostic eval: (a) upweight had literally no effect because boxed-token loss is already near-zero in dgxchen's category-stratified batches; (b) upweight shifted categorical wins/losses but they cancelled in aggregate. Kh0a's claimed +0.03 from this lever was presumably contingent on the rest of his stack (rebalanced sampling, small batch, cosine LR, all-modules targets).

Phase 1 → Phase 2 pivot trigger HALF-TRIPPED. The pre-committed rule: if v14 also < 0.86, declare dgxchen architecture family exhausted at our access level and pivot to Phase 2 (RLVR/GRPO using our solvers as verifier rewards).

Side artifacts captured in commits `22d75c0` and `d0d8fb7`:
- `v13-colab.ipynb` v3 — the final validated notebook with both fast-path defenses
- `v13-pipeline-test.ipynb` — CPU-only ~3 min validator that catches Kaggle-upload bugs before real-run compute is committed. Uses a fake 50 MB adapter and a timestamped Kaggle slug. Saved us from learning about upload-pipeline bugs the hard way after 4.4 hr of training.

**v14 design discussion (deferred pending v13 diagnostic eval):**
- Option α — Cipher v2 wordlist swap (pre-committed in PLAN.md trigger): concrete evidence from v9 eval (fixes `mouse`-not-in-examples failure). Risk: PLAN.md's 2026-05-19 inspection note says dgxchen's cipher CoTs may already cover this; never empirically verified.
- Option β — Tong's format augmenters (~3,450 pattern-teaching records). What Tong's actual 0.85 entry used. Risk: data perturbation cost (v4 lesson).
- Option γ — Run v13 diagnostic eval (~25 min, free on Kaggle) FIRST. If v13 vs dgxchen prediction breakdowns are identical → BOXED_LOSS_WEIGHT really did nothing → drop loss-weighting entirely for v14. If they DIFFER but cancel in aggregate → maybe a smaller weight (3.0) or different gate (`</think>` instead of `\boxed{`) would help.

Compute reality: Kaggle reset Saturday 2026-05-23 UTC (3 days out). Remaining now = ~29 CCU Colab + 5.5 hr Kaggle. Neither fits a full v14 real run (~7 hr). Best use of this window: diagnostic eval (γ) + v14 smoke design, then real v14 run on fresh Kaggle quota Saturday.

**2026-05-20 (Day 12, evening): v13 diagnostic eval cancelled. v14-C (stacked: augmenters + cipher swap) locked.**

Attempted v13 diagnostic eval on Kaggle. First push got dispatched to free-tier P100 16 GB (CLI ignores `machine_shape` metadata) — fail-fast guard caught it. Second push with manual RTX Pro 6000 selection from Kaggle UI got the right hardware but problem #1 took 2,661 sec for 4,096 tokens and produced `boxed=''` (hit max_new_tokens without finishing). Extrapolation: 22 hr for full 30-problem run, exceeds 12 hr Kaggle kernel limit. Cancelled.

The one data point + comparison to v12/dgxchen evals (which ran ~24-60 sec/problem on the same harness) tells us: **v13 generates 10-15× longer outputs than dgxchen.** LB scores both at 0.84, so the LB grader gives v13 enough budget to finish (or the final `\boxed{}` answers are the same). **BOXED_LOSS_WEIGHT=5.0 is a verbosity dial, not a quality dial** — upweighting tokens after `\boxed{` gave them stronger gradient signal but the model learned "spend more tokens" not "be more correct."

Decision: **drop BOXED_LOSS_WEIGHT for v14. Lock v14-C = stack two single-variable changes on dgxchen baseline.**

```
v14-C = dgxchen recipe + (a) Tong's format augmenters + (b) cipher v2 swap
  (a) Add Tong's ~8,463 augmenter records (matching/concatenation/splitting/spelling/lstrip)
      Pure pattern-teaching; what Tong's public 0.85 entry actually used.
  (b) Replace dgxchen's 1,656 cipher CoTs with our cipher v2 1,576 records
      (Wonderland wordlist fallback — concrete evidence from v9 eval mouse-failure case)
```

Stacking violates strict single-variable discipline but the pivot trigger is binary (v14 < 0.86 → Phase 2 RLVR). Stacking maximizes the chance v14 lifts. If lifts → future iterations attribute via ablation. If flatlines → Phase 1 family exhausted, pivot decisively.

Combined corpus ≈ 16,213 records (6,174 dgxchen non-cipher + 1,576 our cipher v2 + 8,463 Tong augmenters). At v13's measured 101 sec/step × ~507 steps = ~14.2 hr — overflows Kaggle's 12-hr kernel limit. Options: (i) subsample augmenters to ~3,400 records (fits Kaggle 10 hr), (ii) run on Colab (no kernel timeout, browser-tab-coupled), (iii) split across two Kaggle runs. Decision deferred until smoke test informs per-step time on the new corpus.

**2026-05-21: v14-C smoke PASSED on Kaggle.** Steady-state **60 sec/step** (vs v13's 101 sec/step — faster because augmenter records are shorter on average, pulling the corpus-level mean down). Real-run projection: 506 steps × 60 sec = **~8.4 hr**, fits Kaggle's 12-hr kernel limit comfortably with buffer for model load + adapter save + Kaggle upload. **No subsampling needed.** Smoke wall: 15.5 min. Quota burned: ~0.26 hr.

Smoke validation checklist (all passed):
- GPU dispatch: RTX Pro 6000 102 GB (slug stickiness held across pushes)
- Mamba fast path engaged (Cell 6 verified all 3 globals are real functions)
- Dual-template data loader works: 20 records split 10 reasoner + 10 augmenter
- Model load + LoRA wrap: 883M trainable params (2.7228%) — matches v13
- No OOM; VRAM peak 84.4 GB / 102 GB
- 3 training steps completed, optimizer step time stable

One signal worth a deeper look: `train_loss` aggregate over 3 smoke steps was **19.34**, higher than v13's smoke (~10 average descending from 11.26). Most likely the augmenter records start at higher loss because the model never saw their symbol-table format in pretraining; loss should descend faster than v13's once the model picks up the format. Longer 10-step smoke planned to verify the trajectory before committing to real run.

**Longer smoke (10 steps × 64 records) 2026-05-21 v13:** confirms descent is healthy.
- Trajectory: 28.5 → 20.2 over 10 steps (0.92 points/step avg descent — comparable to v13's 0.78/step)
- Starting loss (28.5) is high because of 50/50 reasoner/augmenter mix: reasoner records use Nemotron's familiar `<think>...</think>\boxed{}` pattern (~10-15 starting loss); augmenter records use unfamiliar multi-line symbol-table format (~40-50 starting loss). Mix averages to ~25-30. Matches observation.
- Real-run projection: at observed descent rate, loss should reach single-digits by step 100 and converge to ~1-2 by step 500. Normal SFT trajectory.
- **Per-step time: 96 sec on 64-record sample** (vs 60 sec on 20-record sample). Larger sample includes longer augmenter records, raising the average. If 96s holds at real-run scale, 506 steps × 96 = **13.5 hr — overflows Kaggle's 12-hr kernel limit by ~90 min**.
- Mitigations if real run hits the timeout (in priority order): (1) subsample matching augmenter from 4,515 → 2,000 records, drops corpus to ~13K → ~10 hr; (2) increase batch size to 64 if VRAM permits; (3) run on Colab Pro G4 (no kernel timeout but ~120 CCU burn at observed rate). Decision deferred until real-run wall time is measured early.
- Open question: dual-template handling — augmenter records may train the model to produce raw output INSIDE Nemotron's `<think>` framework (because `enable_thinking=True` in the chat template). This *works* (loss descends) but might not match Tong's exact 0.85 training. Whether this matters for LB will be answered by the v14 LB score itself.

**Status: cleared for Saturday 2026-05-23 real run on fresh Kaggle quota.** Optional mitigation #1 (subsample matching) can be applied pre-emptively if we want a safer 10-hr fit; otherwise launch full corpus, monitor first 30-40 HEARTBEATs, and apply mitigation if average per-step exceeds 75 sec.

**2026-05-24 (Day 16): v14 LB 0.68 — Phase 1 pivot trigger fires.**

After two failed real runs (v15 timed out at step 442/507 with no save_strategy; v16 disk-crashed at step ~150 with save_strategy='steps' + save_total_limit=2 saving full optimizer state), the v17/v2-slug attempt with `save_only_model=True` + `save_total_limit=1` (~3.5 GB per checkpoint) and matching downsampled 4515 → 1500 completed full 413 steps in 11.0 hr. Submission graded at **LB 0.680**.

Significant regression: -0.16 from dgxchen baseline (0.84). Root cause hypothesis:

- **Dual-template SFT mixing collapsed reasoner-mode behavior at inference.** We trained the model on TWO formats simultaneously: reasoner records (`prompt + CoT + \boxed{answer}`, dgxchen native) and augmenter records (`prompt + multi-line symbol table`, Tong augmenter style). Both wrapped in Nemotron's chat template with `enable_thinking=True`.
- At inference, when given a reasoner-style problem and entering thinking mode, the model sometimes emits augmenter-format output INSIDE the `<think>` block instead of a proper reasoning trace. No closing `</think>`, no `\boxed{}` answer — the grader extracts nothing and scores 0.
- Math sanity check: ~20% of reasoner problems collapsing into augmenter-format output → 0.84 × 0.81 ≈ 0.68. Matches observed LB.

**Methodology lesson:** Tong's 0.85 LB with augmenters worked because his FULL pipeline (different chat template handling, different masking, different training objective) handled the dual-format mixing cleanly. Copying just the augmenter data into dgxchen's stack does NOT replicate his result. This is a publishable negative result for the Open Contribution Award writeup ("we tested format-augmenter integration in a non-Tong SFT recipe and observed -0.16 LB regression; the augmenter approach is not portable across SFT pipelines without matching template-handling logic").

**Pivot trigger DEFINITIVELY FIRES.** Pre-committed rule (from 2026-05-18 entry): "If v13 < 0.86 AND v14 < 0.86, declare Phase 1 architecturally exhausted at our reach. Immediately move to Phase 2."
- v13 = 0.84 < 0.86 ✅
- v14 = 0.68 < 0.86 ✅

Phase 1 (dgxchen SFT recipe family with our deltas) is exhausted. Moving to Phase 2: **RLVR / GRPO with our solvers as verifier rewards.**

Phase 2 design (initial sketch — to be refined after research agents return):
- Base policy: dgxchen 0.84 adapter as warm-start (best Phase-1 result)
- Reward: 6 binary verifier solvers from `scripts/rl_reward.py` (one per category, 12/12 unit tests passing)
- Trainer: TRL `GRPOTrainer` integrated with Unsloth FastLanguageModel (same Kaggle stack as v13/v14)
- Prompt dataset: train.csv problems (~9,466 total), no completions (GRPO generates and grades)
- Reference: Dang et al. 2503.16219 — reports +0.05 to +0.15 LB lift from GRPO on similar setups
- Expected outcome: 0.84 + 0.05 to 0.15 = 0.89 to 0.99 (would meet/exceed the private 0.87 ceiling)

**Pre-launch research (running 2026-05-24):**
- Leaderboard intel refresh — has the public/private LB topology shifted in the week since 2026-05-17? Any new published recipes? Any GRPO data points from the community?
- GRPO feasibility check — TRL+Unsloth GRPO status, compute cost estimates for 30B MoE+Mamba model with K-sample generation, known failure modes (reward hacking, format collapse), realistic effort estimate for the 22 days remaining to deadline

Banked submissions after v14: v9 (0.62), v12 (0.72), dgxchen direct (0.84), v13 (0.84), v14 (0.68). Best 2 for final picks: dgxchen + v13 OR v13 + v15 (if v15 ≥ 0.84). v14 doesn't help the pool, but IS valuable as a methodology negative result for award writeup.

**Compute state going into Phase 2:**
- Kaggle: ~3 hr remaining this week (heavily burned by v15/v16/v17 attempts); resets 2026-05-30 to fresh 45 hr
- Colab Pro G4: ~29 CCU (~3.3 hr at 8.71/hr); resets monthly (date unknown, check)
- Total budget through deadline: ~45 hr/wk × 3 weeks = ~135 hr Kaggle + ~3 hr Colab = comfortable runway for 1-2 GRPO real runs + iteration

Time-to-deadline budget: 22 days. Phase 2 effort estimate (off-GPU coding + smoke + real run + LB + iteration) ≈ 10-15 days of focused work. Buffer ~7-10 days for unexpected issues or v15+ iterations.

Cell 2 install pattern (CORRECTED 2026-05-28 after the v15 disk loss — `TARGET` MUST be `/tmp`, NOT `/kaggle/working`):
```python
TARGET = "/tmp/nemo-pkgs"   # /tmp is scratch, NOT counted against the 20 GB /kaggle/working output cap.
                            # Using /kaggle/working/site-packages here is what blew v15's disk at final save.
# Training stack (offline wheelhouse has all transitive deps):
pip install -q --no-index --find-links {MAYUKH}/packages --target {TARGET} \
    unsloth trl peft transformers datasets accelerate bitsandbytes
# Mamba wheels (--no-deps because their torch dep is blocked by enable_internet:false):
pip install -q --target {TARGET} --no-deps {MAYUKH}/causal_conv1d-*.whl
pip install -q --target {TARGET} --no-deps {MAYUKH}/mamba_ssm-*.whl
sys.path.insert(0, TARGET)
```
This is now the canonical Kaggle training install. The eval/inference notebooks have a different pattern (site.addsitedir on ryanholbrook/nvidia-utility-script for the patched transformers); the two MUST NOT be cross-pollinated. See Operating Discipline #9 for the full disk-budget rule.

Augmenter records need a DIFFERENT training template than reasoner records: reasoners use `prompt + CoT + \boxed{answer}` (dgxchen's template), augmenters use raw `prompt + completion` (no CoT, no `\boxed{}` — answers are multi-line tables). v14 notebook must handle both templates conditionally on `type` field.

**Methodology Award angle:** the solver work is still a valid contribution even if it doesn't appear in our winning adapter — it's the verifier infrastructure that enables Phase 2 and the dataset is independently valuable. Worth highlighting in the writeup.

**2026-05-28 (Day 20): v15 LOST to disk crash at final save — second disk loss; hard rule #9 added.**

v15 = dgxchen recipe + lm_head only (single-variable). Smoke passed cleanly (888M trainable params confirmed the lm_head LoRA was added, ~61s/step, loss descending). Real run trained the FULL 245 steps (train_loss 2.92) over 6.85 hr — then died at the final-checkpoint save: "Your notebook tried to use more disk space than is available." Zero usable output; `checkpoint-245/adapter_model.safetensors` is 0 bytes at source, no checkpoint-200 fallback (rotated out by save_total_limit=1). Not recoverable.

Root cause: I never counted the ~10-13 GB `site-packages` that Cell 2's `pip install --target /kaggle/working/site-packages` puts in the output-committed directory for the whole run. The v16 "disk fix" only shrank the checkpoint line-item. At step-245 save, /kaggle/working held site-packages (~13) + checkpoint-200 not-yet-rotated (~3.5) + checkpoint-245 being written (~3.5) > 20 GB cap. v14-c-train-v2 (0.68) had squeaked under the same cap; v15 crossed it.

This is the THIRD lost real run (v15-timeout earlier, v16-disk, now v15-lm-head-disk) and the SECOND specifically to disk. The cost: ~7 hr of a ~12.6 hr weekly quota, gone, for a result we already knew how to prevent. Owning it: I okayed a 7-hr run, knowing compute was scarce and that we'd been bitten by disk once, without doing the full /kaggle/working arithmetic. Hard Operating Discipline #9 added to prevent recurrence (install to /tmp; smoke must exercise the save path; print df -h).

**lm_head hypothesis is still UNTESTED** — training completed fine; only the save failed. v15-retry next week (after Saturday 2026-05-30 Kaggle reset) with the /tmp fix should finally give us the lm_head data point.

**Hard reset on cadence for the remaining weeks:** every real run from here must (1) install deps to /tmp, (2) pass a smoke that ALSO saves+zips and prints df -h, (3) fit the 12-hr cap with the disk math shown empirically before launch. No more "looks like it should fit" runs.

**2026-05-29 (Day 21): v15 recovered + graded 0.52 — lm_head catastrophic. Recipe-tweaking declared exhausted. Pivot to SYNTHETIC DATA.**

Correction to the 2026-05-28 entry: v15 was NOT unrecoverable. My CLI `kernels output` recoverability check was wrong — it downloads small files first and I killed it before the 4.26 GB safetensors streamed, then misread the un-streamed 0-byte placeholder as "empty at source." I checked the Kaggle UI Output tab and saw the full files. The `v15_adapter/adapter_model.safetensors` was intact (4.26 GB); the disk crash had only truncated the auto-built `submission.zip` (2.96 GB vs the full 3.62 GB), which is why the first submit returned "Evaluation metric raised an unexpected error." A CPU `v15-repackage` kernel (`kaggle/v15-repackage/`) re-zipped the intact adapter (tensor-byte-range integrity verified, CRC verified) → graded at **0.52**.

**0.52 ≈ untrained baseline (0.50). lm_head addition is CATASTROPHIC (−0.32), not inert.** Mechanism (best hypothesis): Nemotron has TIED embeddings (lm_head ↔ embed_tokens). A LoRA on lm_head ALONE trains fine in-memory (loss descended to 2.92) but does not round-trip through PEFT save→load on the tied base — the grader reconstructs a corrupted output projection → near-random logits. dgxchen's exclusion of lm_head was a deliberate, load-bearing choice. (Kh0a used lm_head WITH embed_tokens, handling the tie symmetrically; we broke it.)

**The decisive pattern — every modification of dgxchen's recipe goes flat-or-down:**

| vs dgxchen 0.84 | change | LB | Δ |
|---|---|---|---|
| v13 | + BOXED_LOSS_WEIGHT=5.0 | 0.84 | 0.00 |
| v14 | + augmenters + cipher swap | 0.68 | −0.16 |
| v15 | + lm_head | 0.52 | −0.32 |

Three axes tested (loss-reweighting, data-augmentation, architecture/target-modules) — all neutral-or-catastrophic. This mirrors v4/v5/v6/v7 (every perturbation of our own v3 regressed ~0.02). **dgxchen's recipe is a sharp local optimum. We will not beat 0.84 by tweaking the recipe — proven three ways. Recipe-perturbation is DONE.**

**New direction (chosen 2026-05-29): SYNTHETIC DATA.** The 0.84→0.87 gap is widely believed to be DATA (Kh0a's private `*_traces_*.csv` feeds we never had). Rather than perturb the recipe, reconstruct that hidden-data advantage: generate MORE high-quality, dgxchen-format CoT traces (verified-correct via our solvers, ideally natural-language via a strong-LLM generator since templated solver-CoTs underperformed — v9 from-base was only 0.62). Format-matching discipline is mandatory (v14 proved different-shaped data regresses −0.16). Two research agents dispatched 2026-05-29: (1) LB topology refresh + comp rules on synthetic/external-LLM data, (2) synthetic-data methods deep-dive (STaR/RFT/distillation, natural-vs-templated CoT, per-category targeting, generation cost). v16 design pending their return.

**Final-pick floor is SAFE regardless:** dgxchen direct (0.84) + v13 (0.84), both reproducible/understood, vs a 0.86 public ceiling. Synthetic data is now an upside swing on a solid floor, not a desperation move. If it doesn't beat 0.84, the negative-results catalogue (BOXED_LOSS_WEIGHT inert, augmenters −0.16, lm_head −0.32 via tied-embedding load failure) is itself strong Open-Contribution-Award material.
