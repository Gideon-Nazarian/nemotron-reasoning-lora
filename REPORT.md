# NVIDIA Nemotron Model Reasoning Challenge — Final Report & Retrospective
*Project: LoRA fine-tuning of Nemotron-3-Nano-30B-A3B for symbolic reasoning. Window: 2026-05 → 2026-06-15.*

---

## TL;DR

- **Final result (competition closed 2026-06-16): public 0.868 → rank 125 / 4,355 on the public leaderboard (top 2.9%); best private-board score 0.852, with my two selected finals scoring about 0.844 and 0.840.** The private re-roll resolved the way the noise analysis predicted — see **§9** for the full post-close audit. The headline post-mortem finding: in our serious band, **public and private scores were statistically uncorrelated (r = −0.057)**, and the from-base **baseline (private 0.848) was within noise of everything we built after it.** The prize frontier (private 0.88–0.92) sat clearly above the whole plateau — the "private moat" was real, not variance.
- **The 0.87+ frontier (~59 teams) was effectively private.** Across **five dated public sweeps** (06-09 → 06-14) we proved no public artifact exceeded 0.86. The strongest confirmation: *the author of the very 0.86 wrap we submitted* (asalhi) publicly posted that he gave up at 0.86 too.
- **Our independent ceiling was 0.84** from-base, robust across ~10 training runs. The **0.86 came only from wrapping a public adapter** (no public method we could run reproduced more).
- **Masking looked like a lever but wasn't.** Completion-only masking scored +0.01 (0.85) on the *public* leaderboard, the first from-base change in ~10 runs to beat 0.84. It was later shown to be **public-leaderboard noise**: the same run scored 0.828 on the private board, below the baseline (see **§9**). The honest from-base story is a flat ~0.84–0.85 with no real lever found.
- **The real deliverable isn't the score — it's the map:** we know exactly what the 0.87 path requires, what's a dead end, and why. We got there by *measuring*, not guessing.

---

## 1. The problem

Fine-tune a **LoRA adapter** for `Nemotron-3-Nano-30B-A3B-BF16` to solve deterministic symbolic puzzles (bit manipulation, ciphers, cryptarithms, algebraic/transformation rules, unit conversion, gravity). The model must emit a Chain-of-Thought then a `\boxed{}` answer, within a **7,680-token budget**. Output is a rank-≤32 LoRA adapter, graded on a held-out set drawn from the *same generator* as train (confirmed by the host).

Constraints that shaped everything: a 30B MoE model (128 experts) on a single RTX Pro 6000; a 12h Kaggle commit cap; weekly GPU quota; and an **offline wheelhouse** (no internet at train time).

---

## 2. Our trajectory

| Phase | What we tried | Best score | Verdict |
|---|---|---|---|
| Baseline reproduction | dgxchen public adapter (direct submit) | **0.84** | reproduces; our from-base anchor |
| | asalhi naive wrap of Tong's adapter | **0.86** | **banked floor** — our best, all project |
| From-base SFT (×~8) | boxed-loss, augmenters, lm_head, MoE-tie, procedural traces, rebalancing | 0.52–0.84 | **hard 0.84 ceiling**; every data change hurt or was neutral |
| Public-recipe replication | Kh0a (minus private data) 0.72; kuangyicheng from-base 0.58; SVD-of-Tong 0.84–0.85; Tong v27 0.51 | ≤0.85 | none reproduce their claim for us |
| Warm-start | from Tong v27 (v16) | 0.54 | silent zero-init (batched/separate key mismatch) — never a real test |
| RLVR / GRPO | smoke | blocked | unsloth#5121 + single-GPU gen throughput |
| **Logprob curation** (the confirmed-public lever) | scored all 10,311 traces by base loss → **bimodal, valley@0.75** | — | the diagnostic that reframed everything |
| → v22 drop-4-hard-categories | | **0.83** | coverage loss > forgetting-prevention → **hurt** |
| → v23 Tong's exact corpus (raw, no mask) | | **0.78** | his data needs his pipeline (masking) |
| → **v20 masking** (completion-only) | | **0.85** public | +0.01 public, **later shown to be noise (private 0.828, see §9)** |
| Endgame | resubmission lottery (~20 rolls) + tie-break | 0.85–0.86, **0× 0.87** | eval noise ±0.02–0.03, skews down; lock early 0.86 |

---

## 3. What we proved (original findings)

These are *our* measured results, not borrowed claims:

1. **A hard from-base ceiling at 0.84.** Across ~10 independent training runs varying data, loss weighting, trace style, and category balance, nothing from-base beat 0.84. The 0.86 required wrapping someone else's trained adapter.

2. **"Verified-correct ≠ generalizes."** Replacing dgxchen's fluent LLM-voice traces with our *100%-solver-verified* traces consistently **hurt** (0.72 vs 0.84). Correctness of a trace is not the same as the base model finding it learnable.

3. **We then *measured* why** — the logprob scoring pass. Base-model teacher-forced loss over our corpus is **sharply bimodal** with a clean valley at 0.75: learnable categories (gravity/equation/cipher/bit-manip ≤0.5) vs "unlearnable" ones (transformation_rules **1.62**, numeral_system 1.26, cryptarithm ~1.0). Those high-loss categories are *exactly* the ones that sank our earlier runs. This turned a hunch ("voice mismatch") into a quantity.

4. **The drop-filter doesn't transfer for us.** Dropping the unlearnable categories (the published 0.87 lever) **cost 0.01** — coverage loss on real test categories (transformation_rules is ~16% of the test) outweighed the forgetting-prevention benefit. The published recipe needs its *companion* levers (masking + the winner's data quality), not the filter alone.

5. **Masking looked real but mild (+0.01 public).** *Later shown to be public-leaderboard noise: the masked run scored 0.828 on the private board, below the baseline (see §9).*

6. **The `out_proj` LoRA is silently dead under Unsloth.** Our warm-start smoke flagged `out_proj.lora_B = 0/23 non-zero`; we initially read it as benign, but a competitor later confirmed it's an Unsloth gradient-checkpointing bug affecting *every* from-base run in the public band. (Fixing it only got that competitor to 0.67 — so it's not the missing point.)

7. **The leaderboard is noisy: ±0.02–0.03 on identical resubmits** (host-confirmed small test set). Much of the 0.87 *band* is 0.86-mean adapters catching upward variance through submission volume (top teams have 168–242 submissions). The decisive endgame lever became the **tie-break (earliest timestamp)**, not the model.

---

## 4. What we ruled out — with proof

So future-us doesn't re-litigate dead ends:

- **Tong v27 (direct / SVD / warm-start):** 0.51 / broken. Its batched expert format breaks every v20-tuned tool; no evidence it's better than v20.
- **SVD of Tong v20** (the public "0.87" notebooks): 0.84–0.85 for us — lossy vs the naive wrap's 0.86.
- **Merging / souping / TIES / DARE:** community-confirmed dead end (rank-96 → SVD-32 destroys features) + our own SVD experience.
- **Kh0a 0.87:** needs private distilled data — unreachable.
- **kuangyicheng warm-start:** silent zero-init on our Unsloth stack.
- **RLVR / GRPO:** library + single-GPU throughput blocked (though a competitor later showed the TRL+vLLM stack *can* run on Kaggle — it just didn't beat 0.86 for anyone public).

---

## 5. The honest gap to 0.87

We did not crack it, and here's exactly why — measured, not hand-waved:

- **The frontier is private.** All 59 teams ≥0.87 published nothing reproducible. The public 0.88 scripts depend on a **private distilled dataset + a private warm-start adapter chain**; forks roll 0.81–0.84.
- **The 0.86→0.87 mechanism is most likely *truncation/format*, not reasoning** (per the community): long CoT hits the 7,680-token budget before emitting `\boxed{}`. The fix is a *concise-format* SFT pass — which we identified but didn't run (out of quota/time, and its public practitioners also sit at 0.86).
- **The 0.88→0.89+ gap is cryptarithm** (Tong: +13–14% there ≈ top). It's ~NP-hard search that doesn't fit one forward pass; every public attempt either stalled or catastrophically degraded other categories.
- **We could not reproduce the winner's full pipeline** (Tinker cloud training + logprob-filter-done-right + masking + spatial bit-formatting + multi-epoch branch-weighted loss) on an offline single-GPU Unsloth stack. Each piece in isolation underperformed; the package is the moat.

---

## 6. Lessons learned

**Technical**
- For trace-style SFT, *learnability beats correctness* — score candidate data by base-model loss before training.
- **Masking** (completion-only loss) is cheap to try, but its +0.01 public gain turned out to be noise (see §9); judge it on a trustworthy local validation set.
- MoE LoRA on this model has real gotchas: batched-vs-separate expert key formats, silently-dead `out_proj`, fused-expert conversion for submission.
- **Always size the run:** Tong's 17,963-record corpus = ~15h > the 12h cap. Compute `steps × s/step` before launching.
- **Leaderboard ≠ truth on a small noisy test.** A single read is a sample; the tie-break can matter more than the model in a dense plateau.

**Process** (the part that actually transfers to real ML work)
- **Measure, don't guess.** The turning points were always a measurement (the logprob scoring, the noise quantification, the author→LB mapping), never a clever idea.
- **Smoke before the real run.** Caught the v23 runtime overrun, the filtered-corpus attach, the warm-start zero-init — each before it wasted 7–10h.
- **A single source-of-truth board + a run ledger** kept ~25 experiments and 5 research sweeps coherent across weeks. Pre-writing the decision tree ("if v22>0.84 then…") stopped us thrashing.
- **Adversarial honesty.** Repeatedly correcting our own over-claims (the "masking is the smoking gun" walkback, the "warm-start was never actually tested" catch) was worth more than any single experiment.
- **Parallel research sweeps** (subagents reading public notebooks + the discussion forum) turned "is there a shortcut?" from anxiety into a dated, evidenced *no*.

---

## 7. Future take — the concrete path to 0.87+

If revisiting with more time / a Tinker budget, in priority order:

1. **Build the vLLM local eval first** (we proved the stack runs on Kaggle; we never finished ours). Host confirmed same-generator train/test → a 950-sample local val is trustworthy. **Measure the truncation rate of our misses** — this single number decides #2.
2. **Concise-format SFT pass** — the one documented 0.86→0.87 mechanism. Compress CoT to end in `\boxed{}` well under budget; fix the dead `out_proj`; warm-start a strong adapter using the now-documented key-transform.
3. **Reproduce the winner's *full* package, not pieces:** Tong's exact corpus + completion masking + his branch-weighted loss (his per-token logprobs are public) + original training order. I had his corpus locally (not included in this repo); I only ever ran fragments.
4. **Then, and only then, cryptarithm** — the 0.88→0.89 category — with a multiplication-anchored / token-remapped approach, *gated by local val so it doesn't nuke other categories.*
5. **Infra:** move heavy generation/scoring off the single GPU (the constraint that blocked RLVR and full-corpus epochs).

---

## 8. Reflection

We finished on the **top plateau with a strong tie-break position** — the best *reproducible* result that was actually on the table, since every public path provably caps at 0.86 and the real frontier is private. More importantly, we ran the project the way good applied ML *should* be run: hypotheses pre-registered, every claim measured, dead ends documented with proof, over-claims corrected out loud, and the search exhausted with evidence rather than abandoned on a hunch. The 0.87 we didn't get is a private-data moat; the rigor we built is portable to the next problem.

---

## 9. Final outcome — competition closed 2026-06-16 (post-close audit)

The private leaderboard was released. We pulled every submission's private score and the official boards (snapshots in `results/final-*-2026-06-16.*`). This section records what actually happened and corrects two pre-close claims.

### Where we finished
- **Public LB: 0.868 → rank 125 / 4,355 teams (top 2.9%).**
- **Private LB (the prize-deciding board): our best submission scored 0.852.** Of our two documented tie-break finals (earliest-0.86-public), the 06-03 21:25 sub scored **0.844** private and the 06-09 22:12 sub **0.840** — so the *counted* private was ~0.844 if that selected pair stood.
- **Prize frontier: private 0.88–0.92** (NullSira 0.920 #1), sitting clear above the entire ~0.85 plateau.

### The decisive post-mortem: public score was uncorrelated with private
Across our 38 serious submissions (public ≥ 0.84):

| | mean | sd | range |
|---|---|---|---|
| Public | 0.8562 | 0.0056 | 0.844–0.868 |
| Private | 0.8418 | 0.0065 | 0.828–0.852 |

**Pearson r(public, private) = −0.057 — statistically zero.** Within our operating band, the public LB carried *no information* about the real test. Everything we did collapsed onto a 0.828–0.852 private cluster (mean 0.842) regardless of public score, with a systematic ~−0.014 public→private shrinkage. Full per-approach table in `EXPERIMENTS.md` → "Public-vs-private leaderboard tracking."

### Two honest corrections to the pre-close report
1. **Masking was not a real lever (corrects §3 finding #5 / §6).** The "+0.01 → 0.85 *public*" masking win scored **0.828 private — our *worst* serious result, below the from-base baseline.** It was public-LB noise, not signal. There was, in the end, **no from-base lever that beat the baseline on the real test.**
2. **The from-base baseline was the quiet winner all along.** The very first thing we submitted — the dgxchen adapter, public 0.844 — scored **private 0.848**, a *positive* gap and within noise of (or above) every wrap and tuned adapter we built over the following month. The best private in our whole pool (0.852) came off a *public-0.844* submission we would never have selected by public score.

### What this vindicates, and the one lesson that changes
- **Vindicated:** the noise thesis (REPORT §3 #7) and the private-moat thesis (§5) — both confirmed by the real numbers. The frontier teams were genuinely, reproducibly ahead; the 0.86 plateau really was a wall of 0.84-mean adapters catching public variance.
- **The lesson that changes:** *on a small, noisy test, the public LB is not a fitness function — it's a single noisy sample.* We knew this directionally (we even ran the noise analysis), but we still let public reads drive ~25 submissions of selection. The thing we kept saying we'd build and never did — a **trustworthy local held-out eval** (REPORT §7 #1) — was not a "nice-to-have for next time"; it was the *only* instrument that could have told signal from noise here. With r≈0 between public and private, **a local val was the difference between measuring and guessing**, and we guessed. That's the portable correction: when the public metric is noisy, build your own measurement *first*, before you spend a single submission optimizing the noisy one.

*Final standing and per-submission private scores: `results/final-submissions-2026-06-16.csv` and `results/final-private-lb-top_2026-06-16.txt` (the full public leaderboard snapshot is not included in this copy).*

---

*Full run-by-run detail: `EXPERIMENTS.md`. Priority board + ruled-out log: `notes/MASTER-087-plan.md`. Method notes: `notes/`.*
