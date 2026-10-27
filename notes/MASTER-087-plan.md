# MASTER 0.87 PLAN & STATUS BOARD — single source of truth
_Last updated: 2026-06-16 — **PROJECT CLOSED** (see bottom of file for the post-close final-outcome audit). Supersedes the scattered priority lists._

## Where we stand
- **Banked floor: 0.86** (asalhi one-click wrap of Tong v20). ⚠️ **KEEP THIS SELECTED as the final submission** — it's the guaranteed floor; everything below is upside.
- **Our from-base ceiling: 0.84** (every from-base data change hurt or was neutral across ~8 runs).
- **Goal: 0.87** (~15 teams there).
- **Confirmed key insight:** the published 0.86→0.87 lever is **logprob-aware data curation** — drop the training traces the *base model* finds "unlearnable" (high teacher-forced loss) + mask prompt tokens. Our own scoring (2026-06-05) **CONFIRMED this on our data**: loss is sharply bimodal (valley @ 0.75); the high-loss categories (transformation_rules 1.62, numeral_system 1.26, cryptarithm ~1.0) are exactly the ones that sank v19/v21.

---

## PRIORITY-ORDERED OPTIONS (the chart)

### TIER 1 — run NOW (both LIVE; the 2-GPU-session cap lets them run in parallel)
| # | Plan | Notebook | What it tests | Status | Next action |
|---|---|---|---|---|---|
| **1** | **#1 Logprob-filter OUR corpus** (drop 4 high-loss categories) | `v22-logprob-filtered` | Does dropping unlearnable traces beat 0.84? Cleanest test of the confirmed lever; rides our proven pipeline; original. | ✅ LIVE (scoring done, corpus built, pushed) | Run on RTX → submit → compare to 0.84 |
| **2** | **#3 Tong's EXACT corpus** | `v23-tong` | Most faithful winner reproduction (his actual 17,963-record data, his taxonomy). | ✅ LIVE (corpus agent-verified, pushed) | Run on RTX (parallel w/ v22) → submit |

### TIER 2 — run after Tier 1 (or once a GPU slot frees)
| # | Plan | Notebook | What it tests | Status | Next action |
|---|---|---|---|---|---|
| **3** | **#2 Pre-filtered public set** (tahaalam — KEEP cryptarithm loss-capped) | `v24-tahaalam` | Alternative to #1: keep hard categories capped instead of dropping. | ⏳ built + dataset uploaded; blocked by 2-GPU-session cap | Push when a slot frees → run → submit |

### TIER 3 — PHASE-2 levers (apply to whichever Tier-1/2 corpus scores best)
| # | Lever | Asset | Status | Notes |
|---|---|---|---|---|
| 4 | **Masking** (completion-only loss) | `v20-masked-sft` (built, never run) | ready | Tong masks; pairs with the filter. Add to the winning corpus. |
| 5 | **Warm-start a strong adapter** + winning corpus | `warmstart-smoke` proved the mechanism | ✅ validated | Warm-start asalhi-v20 (0.86, separate-expert → binds) then continue-train on the winner. NOT a standalone path — a modifier. |
| 6 | **Hyperparam tweaks** (accum 64 vs our 32, 3× hard-cap, lower LR, anchor blend) | — | free riders | Fold into the winning run. ⚠️ do NOT add lm_head/train_unembed — we got 0.52 from base. |
| 7 | **Category recovery** | needs new solver work | idea | IF filtering helps: rewrite the dropped categories (transformation_rules/cryptarithm) in a *low-loss/learnable* form (e.g. spatial formatting) to win coverage back without the forgetting. |

### TIER 4 — lower priority / only if a result points here
| # | Plan | Asset | Status | Notes |
|---|---|---|---|---|
| 8 | **#5 repaired_sft** (solver-verified trace repair) | `wethepeople918/repairv4` (9,488 rows, public) | not built | Different lever (repair wrong traces). Only worth it logprob-gated. |
| 9 | **#6 Spatial bit-manip formatting** (Tong's x/y per-bit grids) | his `reasoners/bit_manipulation.py` | SKIP unless pointed here | ~1000-line solver rewrite for a category we don't even score (bit_manip is already learnable for us @ 0.49). Low EV. |

---

## DECISION LOGIC (so we don't jump around)
1. Run **v22 + v23-tong** (parallel). Submit both, compare to 0.84.
2. **If v22 > 0.84** (drop-filter helps) → the thesis holds → add **masking (#4)** to v22's corpus, then **category-recovery (#7)**.
3. **If v22 ≤ 0.84 but v23-tong > 0.84** → Tong's *data quality* is the lever, not just the filter → build on his corpus + masking.
4. **If both ≤ 0.84** → coverage loss dominates → run **v24-tahaalam (#3, keep-but-cap)**; if that also fails, the gap is infra (Tinker vs our Unsloth) and **0.86 is our genuine ceiling**.
5. **Common caveat:** none of v22/v23/v24 cover `transformation_rules` (~16% of test) — consistent with the thesis (it's the highest-loss category) but the known coverage risk. Category-recovery (#7) is the fix if filtering otherwise works.

---

## RULED OUT — with proof (DO NOT REVISIT)
| Path | Result | Why dead |
|---|---|---|
| Tong **v27** (all routes: direct, SVD, warm-start) | 0.51 (SVD-convert) + v16 warm-start 0.54 | v27's batched format breaks every v20-tuned tool; zero evidence it's better than v20. |
| **SVD** of Tong v20 (the public "0.87" notebooks) | 0.84–0.85 (all 5 candidates) | Does not reproduce 0.87 for us; SVD is lossy vs asalhi's 0.86 naive wrap. |
| **Kh0a 0.87** recipe | unreachable | Needs private data (`llkh0a/...distiled`); our recipe-minus-data = 0.72. |
| **kuangyicheng "0.88"** warm-start-from-v27 | = our v16 (0.54) | Same batched/separate mismatch; warm-start silently zero-inits on our stack. |
| **From-base data-engineering** (more/better solver traces) | 0.84 ceiling (v13–v19; v19 procedural 0.72, v21 rebalance 0.76) | Adding/replacing solver traces hurts or is neutral — because they're the high-loss "unlearnable" ones (now measured). |
| **RLVR / GRPO** | blocked | unsloth#5121 (logits-as-hidden-states crash) + single-GPU gen throughput wall. Not in our offline wheelhouse. |

## DEFINITIVELY-PROVED status of public ≥0.87 methods (strict: faithful-execution only)
- **SVD (hammadfarooq)** → PROVED does NOT reach 0.87 (got 0.84–0.85, ran their own candidates).
- **asalhi/Tong-v20** → PROVED 0.86 (banked).
- **dgxchen** → PROVED 0.84.
- **kuangyicheng warm-start / Kh0a** → BLANK (never faithfully testable — load bug / private data).
- ⇒ Every faithfully-tested public method maxed at **0.86**. Logprob-curation (Tier 1) is the first genuinely-untested lever, and it's now in flight.

---

## Detailed notes index
- `notes/logprob-filter-result-2026-06-05.md` — the scoring result (bimodal, valley@0.75) + filter design.
- `notes/087-experiment-queue-2026-06-05.md` — the 3 corpora + datasets + notebook details.
- `notes/087-leads-ranked-2026-06-05.md` — the 4-subagent research synthesis (logprob/masking/Tong-repo).
- `notes/public-087-methods-proof-status.md` — the strict proof table.
- `notes/recipe-diff-vs-frontier.md` — the running recipe-diff + ruled-out log (v27, SVD, warm-start correction).
- `EXPERIMENTS.md` — the full run-by-run ledger.

---
## LIVE STATUS UPDATE — 2026-06-05 (later)
- **v23-tong runtime bug caught + fixed.** Tong's 17,963-record corpus = **562 steps ≈ 15h** at batch1×accum32 — exceeds Kaggle's ~12h commit cap, and v23 had `save_strategy='no'` → the currently-running v23 will be **killed with no output.** Patched to `max_steps=380` (~10h, still 12,160 recs > v18's 10,311) + checkpointing. **The running v23 is doomed — cancel it; re-push the fixed version when a slot frees.**
- **Lesson banked:** before a real run, compute `total_steps × s_per_step` vs the 12h cap. v22 (253 steps ≈7h) + v24 (272 steps ≈7h) are fine; only Tong's big corpus overran.
- **Net stage:** v22 smoke-validated + ready; v23 FIXED (needs re-push); v24 built (needs push). All three fit <12h now.

### Push status — 2026-06-05 (all 3 real notebooks LIVE)
- **v22** → `gideonnazarian/v22-logprob-filtered` (smoke-validated)
- **v23** → `gideonnazarian/v23-tong` v3 (fixed: max_steps=380 ~10h + checkpoint)
- **v24** → `gideonnazarian/v24-tahaalam-run` (NOTE renamed slug — old `v24-tahaalam` got corrupted on Kaggle by session-cap-failed pushes; status 500s, abandoned)
All fit <12h. After reset: run at window-start → submit each → compare to 0.84 → board decision logic.

### SMOKE STATUS — ALL THREE VALIDATED ✅ (2026-06-05)
- **v22** smoke ✅ passed (filtered corpus, 7 learnable cats, start-loss 16.69).
- **v23** ✅ corpus+train proven by the cancelled real run (step 22 on full corpus).
- **v24** smoke ✅ passed via `v24-tahaalam-smk` (tahaalam corpus, cryptarithm PRESENT + no transformation_rules, start-loss 22.67).
- Consistency check: step-1 loss orders correctly with high-loss content — v22 16.69 < v24 22.67 < v18 26.5. Corpora confirmed.
- **FULLY GREEN.** After reset, run all 3 reals (v22-logprob-filtered / v23-tong / v24-tahaalam-run) at window-start → submit → compare to 0.84.

### ECOSYSTEM SCAN — 2026-06-07 (no reproducible shortcut found)
Leaderboard structure (pulled CSV): 0.89×1 (NullSira, no public method) · 0.88×0 · **0.87×19 (hard wall)** · 0.86×1,421 (plateau=our floor) · 0.85×506 · 0.84×243. **Every PUBLIC-notebook author is ≤0.86; the 19 at 0.87 publish nothing reproducible.**
- New high-vote notebooks (lossless/finding-nemo/best-0.86/best-score) = one-click wraps or SVD of the same Tong/Kien 0.86 adapter = our floor or our ruled-out SVD. cfxyai tong-adapter = same 0.86 family.
- RL wave (grpo/dpo/dapo) = same RLVR blocked for us. New trace datasets = re-exports of Tong corpus (already in v23) or self-gen CoT (our 0.84 ceiling).
- **kuangyicheng-087 UPDATED**: now warm-starts from Kien Tinker **v14 (0.86, not v27)** + an EXPLICIT verified 100%-coverage weight-conversion that defeats the zero-init that broke our v16/v17 + trains in_proj only 32 steps @ LR 2e-6, keeps 50% delta. The conversion routine is the engineering fix we lacked — BUT author is at 0.86, labeled aspirational, no outputs → NOT a 0.87 path. Banked as the correct warm-start pattern IF revisited.
- Marginal-new content: `leevvin` donald_solvers = terser bit_manip CoT for same problems (paired-trace logprob-pick test possible, single-task, low value).
⇒ **VERDICT: stay the course.** Our logprob-curation runs target the only structurally-open lever; no public artifact crosses 0.87. Re-scan ~2026-06-12.

### RESULTS IN — 2026-06-07: v22=0.83, v23=0.78 (decision tree → branch 4)
- **v22 (drop-filter) = 0.83** (−0.01 vs 0.84). Dropping the 4 high-loss categories HURT — coverage loss (transformation_rules ~16% of test) dominated forgetting-prevention. **The logprob-DROP filter alone is NOT our lever.**
- **v23 (Tong raw, from-base, no mask) = 0.78** (−0.06). His data needs HIS pipeline (masking); raw-from-base is worse than our own data.
- **Decision logic → branch 4 (both ≤0.84):** coverage-loss/no-mask dominates. The from-base 0.84 ceiling holds; **0.86 (banked wrap) remains our real ceiling.**
- **The ONE untested doc lever left = MASKING** (`v20-masked-sft`, built, never run). The doc's 0.87 = filter + MASKING; we only ever tested filter. v23's raw-data collapse (0.78) is itself evidence masking matters. → NEXT: run v20-masked (masking on the proven full v18 corpus = clean masking test). Honest odds: modest (dgxchen hit 0.85 w/o masking) — likely 0.84–0.85, not 0.87.
- **v24 (keep-but-cap)** now lower-priority (still no-mask, still drops transformation_rules → likely ~0.84). Run only if curious.

---
## HARD REFRESH — 2026-06-09 (discussion layer cracked via r.jina.ai; 2-agent sweep)
**LB now: 0.89×1 · 0.88×3 (Kh0a 168 subs, Domdolus 93, vli only 10) · 0.87×~20. Still ZERO public artifacts from any ≥0.87 team.**

### Finding 1 — LB NOISE ±0.02–0.03 on IDENTICAL resubmits (D/704491, ≥4 independent confirms; host CPMP: fluctuation ≤~3%)
MoE-router/Mamba non-determinism: same adapter, temp=0 → scores swing ±2–3 points. Same-seed training also non-deterministic.
⇒ (a) Many 0.87/0.88 reads are likely LUCKY ROLLS of 0.86-mean adapters (Kh0a 168 entries, JK-Piece 216 — grinding the lottery; vli's 10-entry 0.88 is the only "efficient" outlier suggesting real method).
⇒ (b) RETROACTIVE CAVEAT on OUR single reads: masking "0.85 vs 0.84" and v22 "0.83 hurt" are WITHIN noise — directional, not confirmed. (0.78/0.51 remain outside noise = real.)
⇒ (c) ZERO-GPU ACTION: resubmit the banked 0.86 zip + the masking 0.85 zip repeatedly (≈5 slots/day, ~30 left before 06-15) to (i) estimate true means for final selection, (ii) roll for a public 0.87+ read. Highest certainty-per-cost move available.

### Finding 2 — Merge/soup/TIES/DARE = CONFIRMED DEAD END (D/704473 full negative result: soup/SLERP/TIES/DARE/Optuna-soup all failed, −0.33; math: 3×r32 = r96 → SVD back to 32 destroys features). Overrides the artifact-side "lopure merge-suite" lead (code published but author 0.86, no outputs). DO NOT spend quota on merging.

### Finding 3 — The one DOCUMENTED 0.86→0.87 mechanism: TRUNCATION (D/703864)
Thesis: the 0.85–0.86 plateau is format/truncation — long CoT hits the 7,680-token budget before emitting \boxed{}. Fix = short concise-format SFT pass (~kuangyicheng: warm-start Tong + ~240 steps of ~30-token boxed answers). Caveats: its practitioners read 0.86 (mechanism plausible, gain unproven beyond noise). Enablers now in hand: tkm2261's `insert_default_adapter_segment` key-transform (FIXES our v16 warm-start zero-init — documented, 11,960/12,008 keys matched); vLLM 0.18 PROVEN working on Kaggle for this 30B (krishnagupta DAPO log) → fast local eval finally feasible → can MEASURE truncation% before spending a run. andy279 HF dataset (49,290 verified traces, Apache-2.0) = free raw material for short-format targets.
WARNING SIGNS: tkm (warm-start + 5-step continuation + logprob gating) sits at 0.84; biohack44 confirms finisher-deltas-on-Tong neutral at 0.86. Light-SFT-on-Tong has hurt as often as helped.

### Finding 4 — The 0.88→0.89 gap = cryptarithm (D/704595, D/703479; Tong: +13–14% there ≈ top). Every public attack stalled (kemshim 27.8% solve, 0.85) or nuked other cats (cryptarithm-only → 0.02). NOT for us in 1–2 runs. Symbol-transform = the other plateau cat (D/704943); no teacher solves the hard ones (andy279: 399 unsolvable-by-any-teacher).

### Other: host confirms train/test SAME GENERATOR (D/705227) → generator-matched local val is trustworthy → with vLLM, a real local eval beats noisy LB reads for selection. RLVR technically unblocked (TRL 1.4 GRPO + vLLM stack ran 11.6h on Kaggle) but 0.86-team, no proven gain, eats the whole final window — skip. Tong GitHub/site STALE (HEAD = our snapshot). Climbers' external footprint: zero.

## FINAL-WEEK PLAN (reset ~06-13 → deadline 06-15; ~30h RTX; ~30 sub slots)
| Pri | Action | Cost | Odds/payoff |
|---|---|---|---|
| **A (NOW)** | Resubmission protocol: banked-0.86 zip ×2–3 + masking-0.85 zip ×1–2; track distribution; final selection by MEAN not lucky read | 0 GPU, sub slots | ~high a 0.87 public read appears across ~10+ rolls if mean 0.86; protects final selection |
| **B (post-reset #1)** | vLLM local-eval notebook: per-category acc + TRUNCATION % of misses on ~200–950 train holdout (generator-matched per host) | ~2–4h | informs/aborts C; reusable selection tool |
| **C (post-reset #2)** | IF truncation% material (>2–3%): concise-format pass — masked v18 recipe + short-format slice (andy279-compressed), OR warm-start Tong-v20 via tkm key-transform + tiny concise SFT | ~8–12h | ~30–40% mechanism real; the only documented 0.87 path |
| skip | v24, merging, RLVR, cryptarithm assault, tkm-style continuation, refine-amplify (unless spare sub slot: ~5–10% lottery) | — | ruled out / capped / too risky |

---
## SWEEP 2026-06-11 — answer: NO public ≥0.87 exists. Resubmission protocol = confirmed top play.
- LB: NullSira **0.90** · 0.88×3 · **0.87×37** · cliff: 1,504 teams at 0.86.
- Cross-referenced ALL 103 members of the 41 teams ≥0.87 vs public kernels/datasets: ZERO live artifacts (only stale March/May ≤0.83 stuff + empty forks). Dipam (new 0.87): nothing public.
- The vote-spiked notebooks are all ≤0.86-evidenced: refine (new "routing-vs-payload" subspace split, claim-only, author 0.86 after 48 subs); kuangyicheng walked BACK to "0.86 Replay Exact Public Reproduction" (explicit disclaimer); nemotronloraforge (100v) = AI-generated vote-bait ("0.91 projection", mock fallbacks, regexes that can't match) — author's 0.86 is a wrap.
- NOISE now multiply-confirmed w/ numbers: same adapter 0.86→0.84→0.83 (screenshot, temp=0); identical seeds 0.84-0.86 vs 0.80-0.81; host CPMP: test set "rather small". Top-of-0.87-band sub counts: JK-Piece 222, Surya 242, Kh0a 168, Y|M|F 169 ⇒ much of the 0.87 band = 0.86-mean adapters grinding upward variance.
- HONEST caveat: documented swings skew DOWNWARD from 0.86 — upside rolls plausible, not proven. Still the highest-EV free action.
- Taha datum: cryptarithm 8%→71% locally yet stuck 0.86 ⇒ single-category fixes don't move the headline metric. De-prioritizes a cryptarithm/concise-format training gamble further.
⇒ FINAL-WEEK: grind the resubmission protocol (0.86 zip ~3/day + 0.85 zip occasionally), select 2 finals by mean. Training quota (post-reset) only if a clearly-better idea emerges; otherwise optional.

### Checkup 2026-06-12 — no new ≥0.87; one finding that EXPLAINS our out_proj mystery
- LB static (0.90×1 / 0.88×3 / 0.87-band same). New kernels = vote-drift on known ≤0.86 wraps; new datasets = adapter repacks (incl. a literal "replay-data-0-86"). JK-Piece's new thread = Kaggle-entry-UI question, no method.
- **D/707802 (Mark Susol): `out_proj` LoRA is silently DEAD under Unsloth** — `UnslothCheckpointFunction` backward guard → out_proj.lora_B exactly zero across all 23 layers, lora_A frozen at init. **This exactly explains our warmstart-smoke observation (`out_proj 0/23 non-zero`)** — not "v3 chose not to train it"; Unsloth never trains it. Implication: EVERY public+our from-base run (incl. dgxchen's 0.84-0.85) trained with dead out_proj — so fixing it is an untested variable, but zero evidence it moves scores (finder is low-ranked), fix requires patching Unsloth internals, and a from-base +0.01 would only TIE the banked wrap. NOTED, NOT ACTIONED.
- Plan unchanged: grind resubmits of the 0.86 wrap (labeled), select two 0.86-reads as finals.

### Resubmission tally — through 2026-06-12
Day 4 (all-wrap protocol): 4×0.86, 1×0.85. Campaign total ≈20 rolls: ~12×0.86 · ~8×0.85 (some = masked zip) · 1×0.84 · **0×0.87**.
Read: wrap's true mean ≈0.856-0.86, excursions DOWN only so far; empirical p(0.87/roll) shrinking (0/20). Keep rolling (free, nothing better) but the decisive remaining task = FINAL SELECTION: tick TWO 0.86-reading submissions of the wrap as finals (can re-tick later; do it now so it can't be forgotten). Expected final (private = its own roll): 0.85-0.86 → the top-cluster plateau.

### FINAL SWEEP 2026-06-14 (deadline 06-15) — answer: NO public >=0.87; the decisive lever is now TIE-BREAK
- Exhaustive dig (full LB CSV + every high-vote deadline-day artifact + datasets + discussions): NOTHING public exceeds 0.86. All high-vote items map to <=0.86 authors (mirza 121v=0.86 repack; nikita 85v=0.86 SVD-merge; refine 104v=0.86; amit "0.85"=0.85; gdataranger out_proj-FIX rebuild=only 0.64-0.67 — confirms fixing the dead-out_proj bug does NOT recover the wrap's level). The 0.88/0.90 frontier (59 teams) published nothing — private distilled data + private warm-start chain.
- CORROBORATION: **asalhi (author of OUR 0.86 wrap) publicly GAVE UP** (D/708105): "all my experiments didn't lead to anything better than 0.86... selected my final subs." D/708212 (today): public 0.88 scripts depend on private data; forks roll 0.81-0.84.
- **NEW DECISIVE FACT (D/708212): 0.86 cluster = 1,528 teams; bronze cutline falls INSIDE the tie; ties broken by EARLIEST submission timestamp.** ⇒ our submission TIMESTAMP is the only remaining medal lever.
- Our earliest 0.86 = **2026-06-03 21:25** (asalhi wrap) — EARLY vs the final-week pile-in → genuinely good tie-break position (plausible bronze).
- **FINAL ACTION (lock NOW): select two finals = the 06-03 21:25 0.86 + the 06-09 22:12 0.86 (both early-timestamp wrap). Resubmission lottery is dry (0/0.87 in ~20 rolls) — keep banked; only swap in a 0.87 if one ever lands.** Do not let a late-timestamp sub become a final.

---
## CLOSED 2026-06-16 — private LB revealed. Project complete.
- **Final: public 0.868 → rank 125 / 4,355 (top 2.9%); best private 0.852.** Frontier private 0.88–0.92 (NullSira 0.920 #1) — clear of the whole plateau ⇒ the private-moat call was correct, not noise.
- **The number that closes the project: r(public, private) = −0.057 over our 38 serious subs (public≥0.84).** Public LB carried ZERO signal about private. Everything collapsed to private 0.828–0.852 (mean 0.842) regardless of public score.
- **Baseline was the quiet winner:** dgxchen from-base (public 0.844) → private **0.848**, ≥ every wrap/tuned adapter built after it. The masking "+0.01" was public noise (private 0.828, below baseline) — corrected in REPORT §9.
- The tie-break plan (earliest-0.86) selected subs scoring 0.844/0.840 private; best-available private (0.852) sat on a public-0.844 sub no public-based heuristic would pick — because public ⟂ private here.
- **The standing lesson (REPORT §9):** on a small noisy test, build a trustworthy local held-out eval BEFORE spending submissions optimizing the public LB. We never did; with r≈0 that was the one instrument that separated signal from noise. Full audit: `REPORT.md` §9 + `EXPERIMENTS.md` tracking table + `results/final-*-2026-06-16.*`. **No open tasks.**
