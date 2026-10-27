# RLVR Design Doc — Nemotron-3-Nano-30B-A3B (v20+ pivot)

> 2026-06-01. Phase-2 pivot after v17/v18/v19 confirmed SFT data-engineering is exhausted at 0.84.
> Goal: push past 0.84 via reinforcement on verifiable rewards. Single RTX Pro 6000 (95 GB).
> Status: DESIGN — no GPU yet. The **OPEN QUESTIONS (§7)** are the targeted deep-research prompts.

## 1. Why RLVR (corroborated, not a guess)
- Our 5-run SFT pattern: KEEP dgxchen traces=0.84, ADD ours=0.84, REPLACE=0.72. Better imitation
  data can't beat 0.84 (`EXPERIMENTS.md` rows 19-22).
- Tong shipped a ppo/cispo/dro/importance-sampling toolkit + advantage-weighting loss; the 0.86 tier
  was an RL/loss-weighting move, not more data (`notes/dgxchen-vs-tong-v20-gap-analysis.md`).
- NVIDIA's own Nemotron RLVR doc: **GRPO** is the recommended algorithm for this model family.
- The diagnostic: v18 loss 0.50 but flat 0.84 LB = memorize-not-generalize. RL optimizes the actual
  generalization objective (reward on correct outputs), not next-token imitation.

## 2. What transfers from NVIDIA's RLVR doc (and what does NOT)
**Transfers (scale-invariant ideas):**
- Algorithm: **GRPO** (Group Relative Policy Optimization). Removes PPO's value+reward models →
  single-GPU feasible (confirmed by Unsloth: GRPO+LoRA runs on one GPU; Qwen3-30B-A3B supported 1.8x).
- **Generations per prompt = 16** (the GRPO group size; advantage = reward vs group mean).
- **Difficulty curriculum: filter out prompts the SFT model already solves** — don't waste rollouts
  on solved problems. Cheap + high-value for us (we can pre-score train.csv with the v18 adapter).
- **Hybrid reward: correctness + token-count** (efficiency) — relevant to reward shaping.
- **KL penalty = 0** in NVIDIA's standard config (simplifies; but see open Q on small-scale stability).
- **LR = 3e-6** — ~30-65x lower than our SFT 2e-4. RL needs tiny steps. Key stability anchor.

**Does NOT transfer (their 1000-GPU cluster machinery):**
- Batch 4,096, 21 nodes, TP=4/CP=8/EP=8, async decoupled inference workers, seq 49K-65K.
- We run SYNCHRONOUS single-GPU GRPO with small batch/group. Their async/in-flight-weights design
  is irrelevant at our scale.

## 3. Proposed architecture (single RTX Pro 6000)
- **Base policy**: v18's KEPT-dgxchen 0.84 LoRA adapter (NOT v19's 0.72). Load + continue-train LoRA.
- **Stack**: TRL `GRPOTrainer` + Unsloth FastLanguageModel (our proven stack; Unsloth has explicit
  GRPO + MoE-A3B support). vLLM for fast rollout generation if it fits.
- **Reward**: `scripts/rl_reward.py` (12/12 unit tests) — tiered 1.0 exact / 0.3 numeric-format-gap /
  0.1 substring / 0.0 wrong, per-category via our solvers as verifiers. THIS is our edge: 5 exact
  verifiers, no reward-model needed.
- **Prompts**: train.csv (9,500), curriculum-filtered to drop ones v18 already solves (focus rollouts
  on the hard ~middle band — too-easy wastes compute, too-hard gives no signal).
- **Memory plan**: base bf16 ~60GB + LoRA + rollout KV cache. Group=16 generations × seq is the
  memory driver. May need group=8 or 4 and/or short max_new_tokens (our answers are short!).

## 4. Why our SHORT answers make this tractable
Unlike NVIDIA's 49K-token reasoning, our outputs are short (mean ~75-1200 tokens). Rollout generation
(the RL bottleneck) is cheap. Group=16 × short-seq is far more affordable than the doc's worst case.

## 5. Reward shaping (from rl_reward.py + NVIDIA hybrid idea)
- Base: solver-verified correctness (1.0/0.3/0.1/0.0). Already built + tested.
- Consider: small token-count penalty (NVIDIA "efficient reasoning") to avoid reward-hacking via
  rambling. Open question whether to add now or after a baseline run.

## 6. Smoke-first discipline (unchanged)
CPU pipeline test → tiny GPU smoke (few prompts, few steps, group=4, confirm reward fires + memory) →
measure per-step → project window → real run. NEVER a blind multi-hour RL run.

## 7. OPEN QUESTIONS → these are the targeted deep-research prompts
1. **TRL GRPOTrainer + Unsloth + MoE (Nemotron_H / A3B) on ONE GPU**: known-working config? Does the
   Mamba-hybrid + 128-expert MoE break GRPO rollout/logprob computation? (We hit MoE-specific issues
   in SFT — gate_up_proj/down_proj expert targeting.)
2. **Group size vs memory** on a 95GB GPU for a 30B-A3B bf16 + LoRA: is group=16 feasible, or do we
   need 8/4? What's the rollout KV-cache cost at our short seq lengths?
3. **KL control at small scale**: NVIDIA uses KL=0 at 1000-GPU batch 4096. At our tiny batch, is a KL
   penalty (or ref-model clamp) needed to prevent policy collapse? What β?
4. **LR + steps**: NVIDIA LR=3e-6. For LoRA-GRPO on a single GPU, typical LR / warmup / how many
   optimizer steps to see lift? (We have ~2 weeks; need to fit in ≤12h Kaggle runs.)
5. **Reward hacking / collapse failure modes** specific to verifiable-reward GRPO on reasoning tasks:
   what are the documented pitfalls (length hacking, format gaming, entropy collapse) and mitigations?
6. **vLLM vs HF-generate for rollouts** with Unsloth on Kaggle: which works offline, and the memory
   tradeoff (vLLM is faster but holds its own weights copy).
7. **Curriculum filtering**: best practice for difficulty-sorting verifiable-reward prompts; does
   dropping solved prompts actually help or does it shift the distribution badly?

## 8. Decision before building the full loop
Once §7 is researched: pick group size, KL setting, LR, rollout backend → THEN build the notebook.
Do NOT build the full loop before resolving Q1 (does GRPO even work with this MoE on our stack) —
that's the make-or-break feasibility gate.
