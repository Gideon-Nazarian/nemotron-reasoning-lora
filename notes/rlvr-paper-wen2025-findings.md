# RLVR paper (Wen et al. 2025, arXiv 2506.14245v2) — findings vs OUR case

> "RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs" (Microsoft Research Asia / PKU).
> Read 2026-06-01. This is the theory+evidence backbone for our Phase-2 RLVR pivot.

## The big claim (and why it matters for us)
RLVR (via GRPO) **genuinely extends the reasoning boundary** — not just sampling efficiency. They
prove (Theorem 1) that **even when the base model GUESSES the right answer with a wrong CoT, GRPO's
gradient increases the probability of CORRECT reasoning over time** — purely from answer-correctness
rewards, no CoT supervision needed. This is EXACTLY our problem: our SFT model gets answers but the
reasoning doesn't generalize (loss 0.50, flat 0.84). RLVR is the theoretically-correct fix.

## Mechanism (Theorem 1, the math we can use)
- Reward is BINARY: R(y)=1 if answer correct, 0 else. (Our `rl_reward.py` is richer — tiered — but
  binary is the proven-minimal baseline.)
- GRPO advantage Â(y) = (R(y) − μ)/σ over a GROUP of G sampled responses per prompt.
- **"Logic Prior" assumption (α>β):** correct CoTs are more likely to yield correct answers than
  incorrect CoTs. If this holds, expected advantage for correct-CoT > 0, incorrect-CoT < 0 → the
  policy monotonically increases P(correct CoT). **Holds when the base model has decent priors** —
  which our v18 0.84 SFT adapter does. This is the green light.

## ⚠️ THE CRITICAL WARNING FOR US (this is the key finding)
**"R1-Zero (pure RLVR from base) did NOT work for the 32B DENSE model; distillation can be very
effective. Key factors (p_c, α, β) for the 32B base are in an even worse state, causing pure RLVR
to converge to suboptimal local solutions."**

And: **"in practice — the 'unideal case' — it is often necessary to first FINE-TUNE the base LLM to
align its output with a proper reasoning distribution BEFORE applying RLVR."**

→ **DO NOT run RLVR from the raw base model.** Start from a SFT-aligned checkpoint. **We already have
the right starting point: v18's 0.84 adapter (SFT-aligned, decent priors).** This directly confirms
the design-doc decision to start from v18, and warns that starting from base would likely fail.

## Answers to our open questions (§7 of design doc)
- **Q (algorithm):** GRPO confirmed (matches NVIDIA doc + Tong's toolkit). Binary reward suffices in
  theory; our tiered reward is a refinement.
- **Q (group size G):** they use the GROUP-relative baseline (advantage vs group mean). Need G large
  enough that σ_Y > 0 (some correct + some incorrect in the group). For our SHORT-answer tasks, a
  modest G (8-16) should give signal. They don't run tiny-G; we'll smoke-test G=8 vs 16.
- **Q (when does it help):** **early** — "RLVR begins to incentivize correct reasoning from the very
  beginning... generalizes to unseen questions within the FIRST 20 training steps" (each step = 16
  gradient updates in their DAPO). → We should see signal FAST. A short run can tell us if it's working.
- **Q (failure mode):** Logic-Prior can fail → "improper model biases / fatal knowledge errors in
  CoTs that finally yield the correct answer get reinforced" (reward-hacking via lucky-wrong-CoT).
  Mitigation in their related work: their CoT-Pass@K metric / multi-verification — but for US, our
  SOLVER verifiers are exact, so we can reward CoT-correctness too, not just answer (see below).

## OUR ADVANTAGE the paper highlights (and we should exploit)
The paper's whole difficulty is that **verifying CoT correctness is hard** — they need an 8B LLM-judge
with 3-vote majority (Fig 7) and admit it's "not infallible." **WE DON'T HAVE THIS PROBLEM.** Our
solvers deterministically verify BOTH the answer AND can check intermediate steps. So we can:
- Reward answer-correctness (their binary R) — baseline.
- OPTIONALLY reward CoT-correctness too (our solvers can verify intermediate values) — which the
  paper says is the real driver (P(CC|CA), "correct CoT given correct answer") but which they can't
  cleanly measure. This could be our edge over generic RLVR.

## Practical numbers from the paper (calibrate our smoke)
- Their reproduction: 32 AMD MI300X GPUs, VERL framework, **over two weeks**, reached ~44% Pass@1 on
  AIME (hard). → RLVR at scale is EXPENSIVE. We have ONE GPU + ~2 weeks. **We must run a SMALL RLVR**
  (few hundred prompts, modest G, short runs), not their scale. Realistic goal: a measurable lift on
  our task, not SOTA.
- "Correct reasoning generalizes after only a FEW gradient updates" → favors short, cheap runs. Good
  for our constraints.
- KL: the DAPO/GRPO recipes here don't emphasize KL; consistent with NVIDIA's KL=0. But at our tiny
  batch, watch for collapse (their R1-Zero-on-32B-dense suboptimal-convergence warning).

## Net takeaways for the build
1. **Start RLVR from v18's 0.84 SFT adapter** (NOT base — paper explicitly warns 32B-from-base fails).
2. **GRPO, group G=8-16, binary answer reward first** (add tiered/CoT-reward as refinement).
3. **Expect fast signal (first ~20 steps)** → smoke can validate cheaply before committing.
4. **Our exact solver verifiers are a genuine edge** — we can reward CoT-correctness, which the paper
   identifies as the true driver but can't cleanly measure.
5. **Scale is our enemy** — they used 32 GPUs × 2 weeks. We run a deliberately SMALL RLVR; the goal is
   a measurable lift off 0.84, validated fast, not matching their absolute numbers.
