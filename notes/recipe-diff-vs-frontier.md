# Recipe diff: our 0.84 vs the reproducible frontier (2026-06-02, local/free)

> Goal (per user): stop guessing — line-diff our recipe against the published frontier recipes to find
> the SYSTEMATIC ingredients we're missing. Sources: notes/lb-intel B.1–B.4 + our notebook configs + EXPERIMENTS.md.

## The table (what each recipe actually does)
| Ingredient | **Us v18 (0.84)** | dgxchen (0.85) | **Tong (0.86)** | Kh0a (0.87) |
|---|---|---|---|---|
| Loss MASKING (prompt tokens) | **NO** | **NO** | **YES (100%)** | ? (likely yes) |
| Category REBALANCING | NO | NO | ? | **YES** (skews to bit_manip/decrypt; symbol_transform→10) |
| BOXED_LOSS_WEIGHT 5× | NO | NO | ? | **YES** |
| lm_head in LoRA targets | NO | NO | ? | YES (+embed_tokens) |
| epochs | 1 | 1 | ? | 2 |
| LR / sched / optim | 2e-4 / linear / adamw | 2e-4 / linear / adamw | ? | 1e-4 / cosine / adamw_8bit |
| **private distilled data** | NO | NO | ? | **YES (unreachable)** |

## What we ALREADY TESTED (so it's not a guess)
- **BOXED_LOSS_WEIGHT=5** (v16): **0.84 — NEUTRAL.** Kh0a's "killer trick" did nothing for us.
- **lm_head in targets** (v18-old): **0.52 — CATASTROPHIC.** (dgxchen also found removing lm_head was +0.1.)
- **More epochs / from-scratch corpus**: capped at 0.62.
- ⇒ Every individual knob we changed on the 0.84 base **hurt or did nothing.**

## CORRECTION to my last message (I over-claimed masking — being honest)
Last message I called masking "the smoking gun." **The diff tempers that: dgxchen reaches 0.85 WITHOUT
masking** (verified — v13-dgxchen masks=False). So masking is NOT necessary for 0.85, and it is NOT a
proven differentiator. It correlates with the 0.85→0.86 step (Tong masks, dgxchen doesn't) but is
confounded with everything else Tong did. **Masking is ONE untested candidate, not the answer.** I keep
over-weighting single levers — flagging this explicitly.

## What's genuinely UNTESTED by us (the only honest candidates)
1. **Loss masking** — Tong does it, we/dgxchen don't. A1 (queued) tests it. Expect 0 to +0.02, not a miracle.
2. **Category rebalancing** — Kh0a skews training counts toward LB-dominant categories. We've never tried it.
   Cheap to test (reweight the corpus sampling). Arguably more principled than masking.

## What is UNREACHABLE
- **Kh0a's 0.87 uses PRIVATE data** (`llkh0a/nvidia-nemotron-distiled-dataset`: splited.csv,
  bit_manipulation_traces_v4.csv, numeric_equation_traces_new.csv). We cannot get it. So **0.87 via Kh0a's
  recipe is not reproducible for us** — which is exactly why our v12 reproduction got 0.72, not 0.87.

## HONEST bottom line (the sobering truth)
- Our v18 (0.84) ≈ dgxchen (0.85). **We have already essentially matched the dgxchen recipe.**
- The reproducible public ceiling is **0.86 (Tong, one-click fork).** Top teams at 0.87 are private and/or
  use private data.
- The only untested levers that might move us 0.84→0.85/0.86 are **masking + category-rebalancing** (both cheap).
- **0.87+ is very likely out of reach for us** (private data + a 0/15 record of beating 0.84). I won't pretend otherwise.
- Realistic best outcomes, ranked: (a) bank Tong 0.86 (floor, one-click); (b) get OUR recipe to ~0.86 via
  masking+rebalancing (original 0.86, better story); (c) 0.87 = unlikely.

## RESULT: v21 category-rebalancing = LB 0.76 (HURT −0.08). Lever RULED OUT.
Tested 2026-06-03. v18 (no rebalance) 0.84 → v21 (Kh0a-style rebalance) **0.76**.
**Why it hurt (honest):** we near-eliminated transformation/cryptarithm (1,287→10), but the TEST set
contains those (~16% of train.csv is transformation_rules). So we cut coverage on a real test category +
shrank the corpus (10,311→3,910). **Kh0a downweighted the same categories BUT replaced them with his PRIVATE
high-quality bit_manip/numeric traces** — we replicated the *cut* without the *replacement*, so we just lost
signal. This CONFIRMS the recipe-diff thesis: Kh0a's recipe is an operating-point package tied to private data;
copying one piece without the rest hurts.
**Pattern now overwhelming:** every change to the 0.84 base → hurts or neutral (v17 0.68, v18-lmhead 0.52,
v19 0.72, v21 0.76; boxed-loss 0.84-neutral). v20 masking is the LAST untested lever (small ~10% footprint).

## RESULT: the public "0.87 SVD" does NOT reproduce for us (2026-06-03)
Forked the public 0.87 method (hammadfarooq470/agi-for-medal-0-87 = SVD-compress Tong/Huikang v20 → rank-32).
Tested ALL 5 candidate strategies → **0.84–0.85**, NOT 0.87. Key facts:
- asalhi's 0.86 wrapper AND the SVD notebook use the **SAME adapter version** (`huikang/nemotron-adapter/.../20`)
  → the model version is NOT the variable.
- For us, the SVD conversion (0.85) is **WORSE** than asalhi's naive conversion (0.86). So SVD hurts, doesn't help.
- ⇒ the public 0.87 SVD notebooks don't reproduce 0.87 for us; the claim is inflated or depends on an
  unidentifiable detail. **0.86 (asalhi naive fork) is our reproducible ceiling.**

## FINAL reproducible ladder (banked = 0.86)
| LB | source | reproducible by us? |
|---|---|---|
| 0.84 | our own training (every modification hurt) | yes — our self-built ceiling |
| 0.85 | SVD candidates of Tong v20 | yes, but worse than naive |
| **0.86** | **asalhi tinker wrapper of Tong v20** | **yes — BANKED, top cluster** |
| 0.87 | public SVD notebooks (claim) / private top teams | NO — does not reproduce for us |

## INVESTIGATED + CORRECTED (2026-06-04): the kuangyicheng "0.88" warm-start is a KNOWN FAILURE, not a new avenue
Re-examined `kuangyicheng/nemotron-087-training` ("Nemotron 0.88 Final"). The current version DOES have an active
warm-start (`PeftModel.from_pretrained(model, huikang/.../27, is_trainable=True)`) — newer than the version we
analyzed on 2026-05-30. I initially flagged this as the missing piece behind our v17=0.58. **That was WRONG, and
our own verified record proves it:**
- **We already DID the warm-start-from-v27 run — it was `v16-warmstart` → 0.54/0.56 (≈untrained).** Root-caused by
  WEIGHT INSPECTION (EXPERIMENTS row 19): Tong v27 is the **BATCHED** expert layout (418 tensors, `model.layers…
  mixer.experts.w1/w2/w3`); our Unsloth+wheelhouse load (identical to kuangyicheng's `FastLanguageModel.from_pretrained`)
  produces the **SEPARATE** layout (12,008 tensors, `backbone.layers…experts.0..127`). PEFT can't key-match → silently
  zero-inits the LoRA → trains noise. **kuangyicheng's warm-start no-ops the SAME way on this stack.**
- The training data is also thin: cell 6 gives every real problem a single canned boilerplate trace + the gold answer
  pasted in (`'I identify one rule… Final answer: \boxed{X}.'`); synthetic = templated one-liners. Reasoning-FREE
  supervision — the v14/v15/v17 collapse signature. v17 (faithful from-base replication) already proved it → 0.58.
- **CONCLUSION: "Copy&Edit kuangyicheng + run ~7h" would repeat v16/v17. DO NOT RUN IT.** Training (from-base OR
  warm-start) is exhausted: verified failures 0.52–0.72 across v6/v12/v14/v15/v16/v17.

## THE one genuinely-untested CHEAP avenue: submit Tong v27 DIRECTLY (no training)
Everyone (asalhi 0.86, all SVD 0.87 notebooks, us) wraps Tong **v20**. NOBODY has submitted **v27** (7 versions newer)
directly. v27 raw is BATCHED → not directly submittable; needs the same batched→submittable conversion asalhi/SVD do.
- **Probe built: `kaggle/bank-tong-v27/`** = our proven `bank-svd-087` converter (SVD-v20 graded 0.85) re-pointed at v27.
  Cheap (free GPU, minutes), CONTROLLED. Prints a v27 format diagnostic (tensor count / batched-vs-separate / rank).
  - If SVD-v27 **> 0.85** → v27 carries more signal than v20 → escalate to the asalhi converter on v27 (0.86→maybe 0.87).
  - If SVD-v27 **= 0.85** → v27 ≈ v20; v27-direct won't help.
- **Best-shot manual variant (higher ceiling than SVD):** Copy&Edit asalhi's 0.86 notebook, swap the adapter input
  v20→v27 (edit `/default/20`→`/default/27` or swap the attached model version), Run All, submit.
- **HONEST odds:** likely 0.86 again. The active 0.87-chasing community would probably have found v27 if it were a
  clear one-click improvement, yet v20 remains the meta 7 versions later → weak evidence v27 isn't obviously better.
  Worth ONE cheap submission because I want every available stone turned; not a high-probability 0.87.

## HONEST ceiling statement (2026-06-04)
For US, the ONLY thing that has ever beaten our own 0.84 training is DIRECTLY submitting someone's strong adapter
(dgxchen→0.84, asalhi/Tong-v20→0.86). Every TRAIN attempt toward 0.87/0.88 failed. The public SVD "0.87" gives us 0.85.
⇒ **0.86 (asalhi/Tong-v20) is very likely our genuine reproducible ceiling**; the ~15 teams at 0.87 are almost
certainly private-data/private-method. v27-direct is the last cheap stone. **Most important action regardless: ensure
the 0.86 submission is SELECTED as final.**

## CORRECTION (2026-06-04, user-prompted): the WARM-START was NEVER verified-loaded — it is UNTESTED, not exhausted
The right question to ask was: "when we tried the warm-start, did we verify it WORKED, or did it default to
nothing because it didn't attach?" Checked the actual artifacts + the retained Kaggle log:
- **v16 Run A (graded 0.54)**: separate-expert env → PEFT couldn't match Tong v27's BATCHED keys → silent
  FRESH ZERO-INIT LoRA → trained NOISE 3h. Caught only post-hoc (lora_B exactly 0). LOAD FAILED.
- **v16 all-linear variant (0.56)**: still separate-expert → still failed load. The +0.02 from a config change
  is what PROVED it was the weights-not-loading, not the config.
- **v16-v2 (rebuilt WITH a verification gate, cell 7.5, that hard-asserts batched repr + nonzero lora_B before
  training)**: the only RETAINED Kaggle run DIED at 14.5s at the VRAM guard (dispatched to a 16GB T4, not RTX)
  — NEVER reached the warm-start load or the gate. NO record of the gate EVER printing "WARM-START VERIFIED."
- Then pivoted to v17 (from-base) on the (then-true) belief kuangyicheng didn't warm-start.
⇒ **A correctly-loaded warm-start + train was NEVER actually run.** Every graded "warm-start" result was a
failed-attach zero-init. So "training is exhausted" was TOO STRONG: FROM-BASE SFT is exhausted (0.84 ceiling);
**WARM-START + train is genuinely UNTESTED.** Fresh wrinkle: the CURRENT kuangyicheng "0.88 Final" DOES warm-start
now (cell 10), unlike the version we judged on 05-30.
**Honest odds caveat:** training v27 (~0.86) on our data risks REGRESSING it toward our 0.84 ceiling (catastrophic
forgetting). Only the LIGHT-ADDITIVE design (warm-start v27 + short pass over the v18 verified-additive traces that
were provably neutral-from-base) has real upside without collapse. Decision gate: the bank-tong-v27 probe's grade —
if v27 standalone ≈0.86, warm-start mostly only regresses; if >0.86, light-additive warm-start becomes worth a run.
Practical: MUST ensure RTX dispatch (UI select) + the cell-7.5 gate must PASS before any training.

## RESULT: SVD-v27 = 0.51 (BROKEN) — v27's format breaks the v20-tuned converter (2026-06-04)
Submitted bank-tong-v27 (SVD converter re-pointed v20->v27). **Graded 0.51 ≈ untrained base** (vs SVD-v20 = 0.85
through the IDENTICAL converter). 0.51 = the adapter loaded but had ~ZERO effect → its converted tensor keys/values
don't bind to the grader's model. **This is a CONVERSION bug specific to v27's format, NOT evidence v27 is a bad
adapter** (user's instinct correct). Same root cause as the v16 warm-start zero-init: v27's internal key/shape export
differs from v20's, and BOTH our converter and the PEFT warm-start loader are tuned to v20's keys. Likely the
expert-expansion shape logic (`if lora_A.shape[0]==1: expand`) doesn't match how v27 stores its expert LoRA →
mangled per-expert weights.
**Strategic read (honest):** this is the 2nd–3rd time v27's non-standard format has broken our tools, and we have
ZERO evidence v27 is actually BETTER than v20 (only kuangyicheng points at v27, and that notebook likely mis-loads
it too). The whole public field wraps v20 because v20 is the clean ready checkpoint; v27 is an awkward intermediate.
⇒ "Fix the v27 converter" = real engineering on an unproven hunch; best case = another 0.86. **v27 looks like a dead
end. STOP chasing it.** 0.86 (asalhi-v20) remains the robust banked ceiling — SELECT as final.
**One valid untested stone that SIDESTEPS v27 entirely:** warm-start from the KNOWN-GOOD 0.86 adapter (asalhi-v20
output, separate-expert = matches our load → binds cleanly, no key mismatch) + light-additive train on the v18
verified-additive data. Modest odds (our data regresses strong adapters), ~7h RTX, but VALID + buildable.
