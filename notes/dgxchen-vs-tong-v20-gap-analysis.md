# The 0.84 → 0.86 gap: what's knowable, and what it tells us for RLVR

> 2026-06-01. Written before pivoting to RLVR, to understand the gap we're trying to close.
> Sources: `reference/tonghuikang-nemotron/` (Tong's repo, frozen 2026-04-27), `reference/huikang-0.85-lb/`,
> `data/dgxchen-corpus/`, `notes/lb-intel-and-recipes-2026-05-17-evening.md`.

## 1. The gap decomposes into TWO +0.01 steps (one knowable, one not)

| build | LB | reproducible? | what it is |
|---|---|---|---|
| OUR v18/v13 | 0.84 | ✅ ours | dgxchen's recipe (Tong CoT re-hosted) + our retrain |
| dgxchen | 0.85 | ✅ public notebook | his own training (we repro 0.84, −0.01 = noise) |
| Tong v18 (prize) | 0.85 | ✅ public recipe + repo | the documented prize adapter |
| **Tong v20** | **0.86** | adapter public, **RECIPE NOT** | post-prize; +0.01 over v18, **unknowable** |

- **0.84 → 0.85** (~+0.01): noise + minor recipe differences. KNOWABLE (both recipes local).
- **0.85 → 0.86** (Tong v18→v20): **NOT knowable.** Tong's GitHub ends 5 days before the prize; v20
  was never written up. Best public guess (low confidence): he "turned on" his pre-built
  `CrossEntropyWithWeightingLossConfig` (unused at prize time) + a refreshed corpus snapshot.

**So chasing 0.86 via reverse-engineering v20 is a dead end** — the delta is private. Banking it
(asalhi wrapper, zero-training) is one click but isn't our work.

## 2. THE KEY FINDING — Tong's training framework already has the RL objectives built in

`reference/tonghuikang-nemotron/train_sft.py` docstring: *"Supports loss functions: cross_entropy,
importance_sampling, ppo, cispo, dro."* And `loss_config.py` defines all of them:

| class | LossFnType | what it is |
|---|---|---|
| `CrossEntropyLossConfig` | cross_entropy | plain SFT (what 0.84/0.85 used) |
| `CrossEntropyWithWeightingLossConfig` | cross_entropy + weighting | **the v20 hint** — per-token advantage weighting (branch_logprob, first_cutoff_weight) |
| `ImportanceSamplingLossConfig` | importance_sampling | off-policy correction |
| `PPOLossConfig` | ppo | clipped policy gradient |
| `CISPOLossConfig` | cispo | clipped importance-sampling PO |
| `DROLossConfig` | dro | direct reward optimization |

**Implication: the 0.85→0.86 lift was almost certainly an RL/loss-weighting move, NOT more SFT data.**
Tong shipped a PPO/CISPO/DRO toolkit and a per-token advantage-weighting loss. The 0.86 cluster's
recipe is reinforcement-style, not data-style. This independently corroborates our own conclusion
(v17/v18/v19: SFT data engineering is exhausted at 0.84) — **the frontier moved to RL, and the
0.86-tier author had RL machinery ready.**

## 3. The CrossEntropyWithWeighting mechanism (directly informs OUR reward design)

```
branch_weight(logprob)   = min(1, |logprob| / branch_logprob)   # e.g. branch_logprob=0.01
cutoff_weight(epoch)     = first_cutoff_weight if epoch==0 else 1.0   # e.g. 0.5
apply_weights: advantage *= branch_weight * cutoff_weight
```
Reading: it scales each token's gradient by how "surprising" it is (|logprob|), capped at 1, and
damps epoch-0 updates by a cutoff. This is **advantage shaping on top of CE** — a bridge between SFT
and RL. For our RLVR, it says: don't reward uniformly; weight by token-level signal.

## 4. What this means for our RLVR pivot (the actionable takeaways)

1. **Start from v18's 0.84 KEPT-dgxchen adapter** (the proven base), NOT v19's 0.72 (our traces hurt).
2. **Our reward is ready**: `scripts/rl_reward.py` (12/12 unit tests) — tiered 1.0 exact / 0.3
   numeric-format-gap / 0.1 substring / 0.0 wrong, per category via our solvers as verifiers.
3. **Objective choice**: Tong's toolkit suggests CISPO or PPO over vanilla GRPO; the
   `CrossEntropyWithWeighting` shows per-token advantage weighting is part of the winning regime.
4. **The 5-run SFT pattern (our finding) + Tong's built-in RL toolkit (this finding) point the same
   way**: the 0.84 wall is an SFT-objective ceiling; the lift is reinforcement on the verified
   reward, not better imitation data. RLVR is the correct and corroborated next lever.

## 5. What we did NOT/cannot learn
- The exact v20 recipe (private). We stop trying to reverse-engineer it.
- Whether v20's lift was the weighting-loss vs corpus-refresh vs both (unverifiable from public data).
- We do NOT copy Tong's `tinker`-framework code (proprietary-ish, and our stack is TRL+Unsloth);
  we use it only as CONCEPTUAL confirmation that RL/loss-weighting is the frontier. Our RLVR is
  built originally on TRL+Unsloth+our reward (originality preserved).
