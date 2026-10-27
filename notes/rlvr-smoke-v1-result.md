# RLVR Phase-A smoke — v1 run result (2026-06-02)

> Durable record of the FIRST GRPO-on-Nemotron attempt. Read alongside
> `notes/rlvr-design-doc.md` and `kaggle/rlvr-smoke/rlvr_smoke.py`.

## TL;DR
**v1 did NOT pass, but it cleared every HARD feasibility gate and crashed on a
known, one-line-fixable torch.compile bug.** This is the best possible "failure":
it proves RLVR is mechanically viable on this model + this single GPU. The only
unanswered question (s/step) is exactly what the v2 retry will measure.

## What the v1 run PROVED (gates cleared — durable wins)
| Gate | Result | Evidence |
|---|---|---|
| Blackwell sm_120 torch build | ✅ | `torch=2.10.0+cu128`, `bf16 matmul OK` |
| Offline wheelhouse (no internet) | ✅ | triton 3.6.0 + causal-conv1d 1.6.1 + mamba-ssm 2.3.1 installed from `mayukh18/nemotron-packages` + `llkh0a/rtx-wheels` |
| GRPOTrainer present in offline trl | ✅ | `trl=0.24.0 transformers=4.57.6 peft=0.18.1` |
| 30B bf16 load on one card | ✅ | loaded in ~8 min, `Max memory: 94.97 GB` |
| LoRA on MoE (128 experts) | ✅ | Unsloth auto-targeted `mlp.experts.gate_up_proj`/`down_proj`; "found no matching parameters" = EXPECTED (matches our SFT precedent) |
| **RoPE bug #4612 (the #1 feared crash)** | ✅ **DID NOT FIRE** | `attn_implementation="eager"` carried it through rollout + into compute_loss with zero RoPE error — confirms prometheus04's ground-truth |
| Explicit 8-module LoRA list (no hang) | ✅ | `[lora] explicit targets set` printed; no get_peft_model hang on experts |
| Reached the GRPO training loop | ✅ | crash was INSIDE `trainer.train()` → `compute_loss`, i.e. past rollout |

## What crashed (the ONE remaining issue)
`torch.compile` (Dynamo) symbolic-shape failure inside Unsloth's **compiled**
chunked log-softmax — NOT our code, NOT a real shape bug:

```
TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
  matmul([.., s12] X [2688, 131072]) — a and b must have same reduction dim
  @ UnslothGRPOTrainer.py:136  chunk_hidden_states.to(lm_head.dtype) @ lm_head.t()
```
- Triggered by the path selector `if (index.shape[1] <= 1024 and batch_size <= 8) or batch_size==1:`
  → our short completions + group=8 take the chunked path.
- Dynamo can't prove the hidden dim `s12 == 2688` (the vocab-projection contraction
  dim) during fake-tensor tracing. **Eager runs this fine** — it's a compile-tracing
  artifact, not a runtime error.

## The fix (applied in v2)
`torch._dynamo.config.suppress_errors = True` — set right after `import torch`
AND re-asserted just before `trainer.train()` (Unsloth can reset dynamo cfg during
init). Dynamo falls back to eager **for the failing region only**; everything that
compiles cleanly stays fast.

**Why this doesn't spoil the s/step measurement:** s/step is dominated by NATIVE
generation (the autoregressive rollout of 8×≤512 tokens). The eager logsoftmax adds
only a small slice to the loss/backward. So a GREEN under suppress_errors is *doubly*
safe (real compiled loss would be marginally faster).

## Escalation ladder if v2 STILL crashes at the same spot
1. (v2) `suppress_errors=True` — surgical eager fallback. ← current
2. `torch._dynamo.config.disable = True` — full eager, guaranteed no compile bug,
   more conservative s/step.
3. Avoid the chunked path entirely: `GROUP=16` (batch_size>8 skips the `chunked_*`
   branch) — but doubles rollout cost and the else-branch is unproven.

## Cost / timing facts (for budgeting the 17h week)
- One smoke iteration ≈ **13 min** wall (≈8 min of it is the fixed 30B load).
- v1 burned ~13 min. Plenty of headroom for v2 + the real curriculum pre-filter.

## Open question v2 answers
**seconds/step at group=8 native gen** → projects whether ~300 GRPO steps fit an
11h compute window. GREEN (≥300) ⇒ build the real run for next week's 45h. RED ⇒
pivot to Unsloth-embedded vLLM or a multi-session checkpointed run.

---

## v2 run result (2026-06-02, ~19:09 UTC) — suppress_errors did NOT work
Re-ran with `torch._dynamo.config.suppress_errors = True` (set after import + re-asserted
before train()). **Crashed at the EXACT same spot** — `chunked_hidden_states_selective_log_softmax`
line 136, identical `s12 X [2688, 131072]` matmul error. All the same gates cleared again
(RoPE still didn't fire; load ~9 min). 

**Why suppress_errors failed:** the error is raised during fake-tensor propagation
(`get_fake_value` → `run_node` → `meta_mm`) which does a HARD `torch._check(M1==M2)` and
re-raises as `TorchRuntimeError` — this is not the kind of backend-compiler failure that
`suppress_errors` intercepts. It propagates straight out.

## v3 fix (applied) — full-eager via TORCHDYNAMO_DISABLE
- `os.environ["TORCHDYNAMO_DISABLE"]="1"` set **before** `import torch` (so Unsloth
  generates its compiled-cache file with compile already off) + `UNSLOTH_COMPILE_DISABLE=1`
  best-effort hint + `torch._dynamo.config.disable=True` (after import + re-asserted before train()).
- **This barely affects the speed metric.** Our generation is ALREADY native/uncompiled
  (`fast_inference=False`, `use_vllm=False`) and the autoregressive rollout DOMINATES s/step.
  Disabling Dynamo only makes loss/backward eager — a small slice. So the v3 s/step is
  representative; a GREEN is real, and native-gen rollout time is identical with/without compile.
- If v3 STILL fails (e.g. Unsloth hard-requires compile, or a new earlier error), next levers:
  (a) GROUP=16 to skip the chunked branch while keeping compile, (b) a newer Unsloth wheelhouse,
  (c) Unsloth-embedded vLLM rollout.

## Cost so far
v1 ~13 min + v2 ~15 min = ~28 min of the 17h week spent de-risking. Still ample buffer.

---

## v3 run result (2026-06-02, ~21:01 UTC) — full-eager WORKED, revealed the REAL bug
`TORCHDYNAMO_DISABLE=1` did its job: **no more Dynamo error.** The run advanced PAST compile
into pure eager and surfaced the true underlying bug (which the compiler had masked):

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (91x131072 and 2688x131072)
  @ chunk_hidden_states.to(lm_head.dtype) @ lm_head.t()   (eager, zero Dynamo frames)
```

### ROOT CAUSE (verified across 4 sources — this is a real wall, not a config typo)
**Unsloth's GRPO loss expects the model to return HIDDEN STATES (dim 2688); NemotronH's
remote-code forward returns LOGITS (dim 131072 = vocab).** So Unsloth's memory-saving
"compute logits chunk-by-chunk from hidden states" trick (`efficient_log_softmax` →
`chunked_hidden_states_selective_log_softmax`) tries to project already-projected logits
through `lm_head` again → `[*,131072] @ [2688,131072]` collision.

Evidence:
1. **unsloth#4711** "Is GRPO training supported for NVIDIA-Nemotron-3-Nano-30B-A3B?" —
   CLOSED, label `feature request`, **zero comments**. GRPO-on-this-model is a *pending
   roadmap feature*, not confirmed-working in released Unsloth.
2. **unsloth#3069** (Gemma-3 GRPO) — same `compute_loss` matmul-mismatch family, but the
   Gemma case is the NORMAL direction (hidden=1152, lm_head=[262144,1152]) and is still
   open/help-wanted. Ours is worse: the "hidden states" already carry the vocab dim.
3. **R2 half-called it**: flagged `chunked_hidden_states_selective_log_softmax` as the
   danger zone (predicted OOM via unsloth#4985) — right function, different failure mode.
4. **prometheus04 (the ONLY proven GRPO-on-this-model run) used a DIFFERENT STACK:**
   unsloth **2026.3.7** + transformers **5.3.0** + torch 2.6/cu124, **WITH vLLM**, on 2×A100.
   OUR wheelhouse (`mayukh18/nemotron-packages`, built for SFT) = unsloth **2026.3.17** +
   transformers **4.57.6** + torch 2.10/cu128. SFT never exercises this GRPO logprob path,
   so the version gap only surfaced now. The NemotronH remote code likely needs the newer
   transformers (5.3.0) to honor `logits_to_keep`/return hidden states.

### v4 fix attempt (applied, build-don't-run): LOGITS-AWARE LOGPROB monkey-patch
Wrap `efficient_log_softmax` at module-global scope so that when the input is already logits
(last dim == vocab), it skips the re-projection and does the mathematically-correct
`log_softmax(logits).gather(index)` directly. NOT a hack — that IS the right per-token logprob
when logits are pre-materialized (the chunked path only existed to avoid materializing them,
which NemotronH already did). Re-scanned right before `train()` in case the compiled cache
generates lazily. Diagnostic prints whether the patch found the function.
- **If it works** → smoke completes, we get s/step, RLVR feasibility resolved cheaply.
- **If the patch doesn't "take" or the next unwired thing bites** → this confirms native
  GRPO on NemotronH needs a STACK REBUILD (match prometheus04's versions) or a different
  path. That's the strategic fork below.

### STRATEGIC FORK (if the v4 patch fails) — open options
| Option | What | Cost / risk |
|---|---|---|
| **A. Monkey-patch** (v4, current) | logits-aware logprob | cheapest; may whack-a-mole the next unwired path |
| **B. Rebuild wheelhouse** to a GRPO-working stack | match prometheus04: unsloth 2026.3.7 + transformers 5.3.0, but Blackwell torch (cu128) | real fix; offline-staging work; A100→Blackwell version compat uncertain |
| **C. vLLM path** (`fast_inference=True`) | may dodge the native logprob path entirely (prometheus04 used vLLM) | flips our native decision; Blackwell-vLLM + MoE-LoRA unproven |
| **D. NVIDIA NeMo-RL** | the ONLY officially-documented GRPO recipe for this model (grpo_nanov3.yaml) | 32-node default; single-GPU prototype; hard 12h fit |
| **E. Bank SFT 0.84/0.86 + methodology** | stop chasing RLVR given the 2026-06-15 deadline | no leaderboard upside from RL, but ships clean |

**Recommendation:** try A (one cheap run). If it fails, the honest call given the deadline is
likely **C then E** — vLLM is the path prometheus04 actually used, and if that's also fiddly
on Blackwell, banking the verified 0.84/0.86 + original methodology is a defensible finish.

### Cost so far
v1+v2+v3 ≈ 43 min of the 17h week. v4 (patch) would be ~1 more run.

---

## v4 run result (2026-06-02, ~21:39 UTC) — my installer bug, fixed in v5
The logits-aware patch crashed at INSTALL time (before testing the actual fix):
```
RuntimeError: Tried to instantiate class 'efficient_log_softmax._logits_aware', but it does not exist!
  @ getattr(_orig, "_logits_aware", False)   (torch/_classes.py __getattr__)
```
**Cause:** my installer scanned ALL `sys.modules` and hit `torch.classes` — a magic module
whose `__getattr__` returns a proxy for *any* attribute name, so `hasattr(torch.classes,
"efficient_log_softmax")` was falsely True, then resolving `._logits_aware` on the proxy
raised. Classic torch.classes footgun. NOT a library issue — my over-broad scan.

**Silver lining:** with Dynamo disabled, GRPOTrainer init was FAST (~4s, no compile) — so
once the installer is fixed, the next run actually tests the patch logic on the real fn.

## v5 fix (applied, LOCALLY TESTED this time)
- Installer now only inspects modules whose name contains "unsloth" (skips torch.classes /
  torch._C entirely) + try/except per module + callable() check.
- **Locally validated before push** (addressing v4's under-testing): (a) simulated the
  torch.classes footgun → installer no longer crashes, patches the unsloth module, idempotent;
  (b) numpy-checked the already-logits `log_softmax(logits).gather(index)` path → matches
  brute-force per-token logprob, correct [B,S] shape, all ≤0.
- v5 is the real test of whether the logits-aware patch unblocks GRPO on NemotronH.

### Cost so far
v1–v4 ≈ ~52 min of the 17h week (v4 died fast at install). v5 = ~1 more run.

---

## v5 run result (2026-06-02, ~22:04 UTC) — patch CAN'T attach (lazy codegen) + ROOT CAUSE is versions
Diagnostic printed:
```
[patch] logits-aware log_softmax installed on: NONE (fn not found — will hit the shape bug)
[patch] re-scan before train patched: (already patched / none new)
... then the SAME mat1/mat2 (68x131072 and 2688x131072) crash inside efficient_log_softmax.
```
**Why the patch didn't attach:** Unsloth GENERATES + imports `unsloth_compiled_cache.UnslothGRPOTrainer`
**lazily, inside `trainer.train()`** — so at both patch points (after trainer init, and right before
train) the module + its `efficient_log_softmax` don't exist yet. Catching it would require hooking
Unsloth's internal code-gen (`unsloth_zoo` create_new_function / exec_module), which bypasses normal
import machinery — fragile surgery on an unsupported feature.

### THE REAL ROOT CAUSE — confirmed version-sensitivity (free research, no quota)
- **TRL #2770** "model.forward requires `num_logits_to_keep`, not `logits_to_keep`" = exactly this
  bug class. The GRPO trainer asks the model to return hidden states via `logits_to_keep`; if the
  model's forward signature/version doesn't honor it, the model computes FULL logits → Unsloth's
  chunked path re-projects them → shape collision.
- **NemotronH** ships `num_logits_to_keep=1` default and was only **added to HF transformers 2026-03-02**.
  Its `logits_to_keep` handling is version-dependent.
- **Our stack** = transformers **4.57.6** + unsloth **2026.3.17** → returns logits (broken for GRPO).
- **prometheus04's PROVEN stack** = transformers **5.3.0** + unsloth **2026.3.7** → worked.
- **unsloth#4711** (GRPO-on-this-model) opened 2026-03-31, closed, no resolution comment.

**Conclusion: native GRPO on NemotronH needs the RIGHT transformers/unsloth versions — NOT a monkey-patch,
and NOT (only) vLLM** (vLLM can't avoid the new-logprob forward, which is where this breaks). The fix is a
new offline wheelhouse with a newer transformers compatible with Blackwell torch 2.10/cu128 — real work,
uncertain compat, and even then s/step (the ORIGINAL concern) is still unmeasured.

### STRATEGIC STATE (decision point — 5 runs in, ~70 min of 17h used)
The monkey-patch path is exhausted. The remaining options, honestly weighed:
- **Bank SFT 0.84/0.86 + methodology** — deadline-safe (2026-06-15), zero further risk. RLVR upside was
  always speculative; it's now confirmed to need infra we can't cheaply stand up on Blackwell.
- **Version-rebuild RLVR** — build a newer-transformers offline wheelhouse (prometheus04's stack), retest.
  Evidence-backed BUT a STACK of compounding uncertainties: (1) newer transformers+unsloth runs on
  Blackwell? (2) GRPO then works end-to-end? (3) s/step fits 12h? (4) RLVR actually lifts the score?
- **Cheaper SFT-side 0.86 lever** — `dgxchen-vs-tong-v20-gap-analysis.md` suggested Tong's 0.86 may be a
  loss-weighting move (SFT-side), which would beat 0.84 WITHOUT the RLVR infra gamble. Worth re-examining.

**Recommendation: bank the verified SFT result + methodology as the safe finish; if pushing higher, the
loss-weighting 0.86 lever is a better bet than the RLVR version-rebuild given the deadline.** The
version-rebuild is a legitimate next-week 45h gamble ONLY if we want a speculative shot at the
NVIDIA-style RLVR gain and accepts the compounding-uncertainty risk.

### Cost so far
v1–v5 ≈ ~70 min of the 17h week. The native-GRPO wall is now fully characterized + documented.
