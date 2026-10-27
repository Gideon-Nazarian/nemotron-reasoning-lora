# Research Findings Synthesis — 2026-05-17

Three research agents dispatched in parallel to scout the next experiments after v9 broke the ceiling at LB 0.62. **All three independently converge on the same conclusion: the bottleneck is DATA, not architecture or training mechanics.** Below is the consolidated playbook.

---

## Agent #1 — Mined Tong's `investigations/*.txt` (1,667 problems)

**Key finding: Tong's bit_manipulation rules fit a NARROW template we mostly don't cover.**

- 1,441 of his 1,667 investigations are `bit_manipulation` (86.4%)
- Rules consist of `{SHL, SHR, ROT}` × `{1..7}` operands combined with `{XOR, AND, OR}` and optional `NOT` — that's 21 atomic ops × 9,408 two-term combinations
- Rule family distribution:
  - F1: `OP1(n) XOR OP2(m)` — 488 problems
  - F2: `OP1(n) {AND,OR,XOR} NOT OP2(m)` — 376
  - F3: `OP1(n) {AND,OR} OP2(m)` — 250
  - F4: Three-term parenthesized `(A op B) op C` — 124
  - F5-F8: Single ops (SHL/SHR/ROT) — 188 (we cover these)
- **Our current bit_manipulation solver covers ~10% (150/1428). A two-term shift-composer with NOT prefix would unlock ~1,283 more problems (95% coverage).**

**Also:** Tong handles 9 categories total; we have 5. **We're missing: cryptarithm_deduce (99), cryptarithm_guess (29), equation_numeric_deduce (10), equation_numeric_guess (41), equation_numeric (6).** ~185 problems we can't currently generate training data for.

**Recommended actions (in order of leverage):**
1. P0: bit_manipulation v2 — two-term shift composer with optional NOT (~2-3 hr, unlocks 1,157 problems)
2. P1: bit_manipulation v3 — 3-term parenthesized composer (~2 hr, unlocks 124 more)
3. P2: cryptarithm_* solver — operator semantics over {+, -, *, concat, abs_diff, reverse_*} with digit-symbol constraint propagation (~4-6 hr, unlocks 128 problems)
4. P3: equation_numeric_* solver — same operator library as P2 but with literal digits (~2-3 hr, unlocks 57 problems)
5. P4: bit_manipulation v4 — choice/majority functions (`ch(a,b,c) = (a&b)|(~a&c)`, `maj(a,b,c) = (a&b)|(a&c)|(b&c)`) (~1 hr, unlocks ~10-20)

**Total reachable training corpus expansion: ~1,500 new solver-verified records (vs our current 4,681).**

---

## Agent #2 — Recent reasoning-SFT papers (8 papers reviewed)

**Key finding: GRPO/RLVR is the highest-EV move for our exact 0.62→0.85 gap.**

Direct quote from agent: *"your 0.62 to 0.85 gap is exactly the shape RLVR closes (sampling efficiency / exploitation of latent capacity)"*

Top 3 papers to deeply read:
1. **Dang et al. 2025 — RL for Reasoning in Small LLMs ([arXiv:2503.16219](https://arxiv.org/abs/2503.16219))** — GRPO on 1.5B distilled reasoner, 7k math examples, verifiable reward. Reports AMC23 63→80, AIME24 to 46.7. Used 4×A40 48GB — fits within our single 96GB GPU. **Direct fit to our setup.**
2. **Wang et al. 2025 — One-Shot RLVR ([arXiv:2504.20571](https://arxiv.org/abs/2504.20571))** — Single carefully-chosen example + entropy bonus → 17.6%→35.7% on 6 reasoning benchmarks. Trivial compute. Drop-in ablation control.
3. **Zhang et al. 2025 — ADCL ([arXiv:2505.08364](https://arxiv.org/abs/2505.08364))** — Adaptive Difficulty Curriculum Learning. Re-estimates per-example difficulty as model improves, lets teacher rewrite problems student keeps failing. Reports +10% AIME24, +16.6% AIME25.

**Suggested 4-week plan (if we commit to RLVR path):**
- Week 1: bolt verifiers into GRPO loop on top of v9 checkpoint (paper #1)
- Week 2: layer ADCL difficulty re-estimation + Self-Evolving Curriculum sampler
- Week 3: verifier-reranked self-consistency at inference
- Week 4: if plateaued, swap flat LoRA for DR-LoRA expert-saliency ranks

**Skip:** TT-LoRA MoE, MixLoRA (we already have MoE), long-context RL papers (irrelevant), heavy multi-stage distillation.

---

## Agent #3 — Tong's actual source repo deep-dive (8 untapped ideas)

Kaggle discussion forum was JS-rendered and unscrapable, but the agent pivoted to read Tong's public source at github.com/tonghuikang/nemotron. **Found 8 specific techniques he uses that we don't.**

### Tong's actual config (eye-opening)
- **LoRA rank=32, batch_size=64, max_length=8192, 1 epoch typical** (NOT multi-epoch like we just discovered helps us)
- AdamW betas (0.9, 0.95), wd=0
- `train_unembed=True` (LoRA on lm_head explicitly trained)
- `lr_schedule.py::StepLinearDecayLRSchedule` — linear decay all the way to ZERO, no warmup, no min floor
- Per-category stratified batching (`_stratified_batches` in `train_sft.py`)

### 8 specific untapped ideas (ranked by agent)

1. **Build cryptarithm + equation_numeric solvers** — Tong's source files (`reasoners/cryptarithm.py`, `reasoners/equation_numeric.py`) are <500 LOC each and directly portable. He uses operator-semantic enumeration + constraint propagation. Same family as Agent #1's P2 and P3.

2. **Tokenizer-derived augmentation corpus** (HUGE)
   - `augmenters/spelling.py` — ~150 problems spelling out tokens char-by-char (`–s–e–x–`)
   - `augmenters/concatenation.py` — 1,500 problems on bracket merging (`【]】【}】 → 【]}】`)
   - `augmenters/splitting.py` — 1,500 problems on token splitting
   - `augmenters/lstrip.py` — 300 problems stripping whitespace inside brackets
   - `augmenters/matching.py` — bit-column pattern matching from bit_manip reasoning
   - **These have NO `\boxed{}`** — pure pattern-conditioning corpus that teaches token-level mechanical skills
   - **This is what gets Tong precise verbatim transformation tables in output format on long-tail problems**

3. **Branch-weighted CE loss** — down-weights tokens model already predicts confidently, concentrates gradient on branching tokens. `loss_config.py::CrossEntropyWithWeightingLossConfig` uses `min(1, |logprob| / branch_logprob)`. We use flat CE.

4. **Verify `lm_head` LoRA actually trains** — Tong sets `train_unembed=True`. Our target_modules includes `lm_head` but Unsloth may skip it for MoE models (we saw the warning earlier).

5. **Step-linear LR decay to zero + 1 epoch** — Tong's actual setup is NOT multi-epoch. He spends his full gradient budget in one careful pass at batch_size=64. Our v9 at 3 epochs is a different strategy. Worth A/B testing.

6. **Per-category stratified batching** — avoids "this batch is 100% numeral, next batch is 100% cipher" oscillations. Likely 1-3 LB points alone.

7. **Per-epoch hard-example mining** — `investigators/get_examples.py` logs per-token logprob each epoch, surfaces problems with `min_logprob < -0.69` (≈p=0.5) as "still hard." These get re-investigated and the hypothesis texts become more training data. Iterative.

8. **Grader spec exploit:**
   - Binary strings: exact match (case-insensitive)
   - Floats: `math.isclose(rel_tol=1e-2, abs_tol=1e-5)` — **LOOSE** (1% tolerance, so 3 sig figs is enough)
   - Else: case-insensitive string match
   - Boxed regex: `\\boxed\{([^}]*)(?:\}|$)` — **accepts unterminated boxes!**
   - Implication: round gravity/unit_conv outputs to 3 sig figs; auditing our exact prompt/completion format vs Tong's is free upside

### Tong's exact completion format (audit this for compatibility)
```
PROMPT: {original problem}
Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`

COMPLETION: {reasoning}\n</think>\n\boxed{answer}<|im_end|>
```

---

## Combined synthesis — three concrete paths forward

### Path 1 — Expand training corpus (HIGHEST CONFIDENCE)
**Build the 4 missing solvers** (cryptarithm × 2 + equation_numeric × 2) plus extend bit_manipulation. Estimated ~10-15 hr of solver work, off-GPU. Tong's source is directly portable. Could plausibly double our training corpus and unlock several full categories we currently can't generate data for.

**Expected lift:** +0.05 to +0.15 (large; Tong needed exactly this data to hit 0.85)

### Path 2 — Tokenizer-derived augmentation corpus (CHEAP + HIGH-VALUE)
**Port the 5 augmenter modules** (spelling, concatenation, splitting, lstrip, matching) — each is 80-150 LOC. These produce a ~3,450-record corpus of pure format-conditioning examples with no `\boxed{}` — train alongside our solver records to teach mechanical token-level skills.

**Expected lift:** +0.02 to +0.05 (per Agent #3, this is what makes long-tail formatting work)

### Path 3 — GRPO/RLVR using our existing verifiers
**Build a GRPO training loop on top of v9 checkpoint** using TRL + Unsloth (we proved this stack works). Reward function = our existing solver verifiers (binary 1/0 per category). Start from v9 (LB 0.62) and let RL push higher.

**Expected lift:** the literature suggests +0.10-0.20 for this exact pattern (small reasoner + verifiable rewards). But it's RL — variance is high, instability is real.

### Recommended ORDER
1. **First (this week):** Path 1 + Path 2 in parallel. Both are off-GPU coding work that can run while v10 finishes and v9 diagnostic runs.
2. **Then:** Re-train v11 with the expanded corpus (probably ~7,500 records vs current 4,681). Use v9 stack at maybe 2 epochs instead of 3 (Tong's 1-epoch-at-batch-64 schedule may also be worth A/B-ing).
3. **Then:** If LB lifts past 0.70, Path 3 (GRPO) becomes the swing for the remaining gap to 0.85.

---

## Update to forward branches in NEXT.md

The earlier "branches A-F" framework is superseded. New priority list:

**Tier 1 (this week, off-GPU coding):**
- Build bit_manipulation v2 (two-term shift composer)
- Build cryptarithm_* solver (Tong's reasoners/cryptarithm.py is portable)
- Build equation_numeric_* solver (Tong's reasoners/equation_numeric.py is portable)
- Port 5 augmenter modules (spelling, concat, splitting, lstrip, matching)

**Tier 2 (after Tier 1, ~3-5 hr training run):**
- v11: v9 stack + expanded training corpus from Tier 1
- Audit prompt/completion format vs Tong's exact spec

**Tier 3 (after v11 lands, contingent on its result):**
- Path 3 GRPO/RLVR if v11 lifts but we still have headroom to 0.85
- Branch-weighted CE loss if simpler experiments plateau
- Step-linear decay + 1 epoch A/B vs multi-epoch

**Tier 4 (speculative, last-mile):**
- ADCL / Self-Evolving Curriculum
- Per-epoch hard-example mining
- 3 sig fig rounding for numeric answer outputs
