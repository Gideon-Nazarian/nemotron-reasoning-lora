"""
rlvr_smoke.py — Phase-A feasibility + speed gate for RLVR on Nemotron-3-Nano-30B-A3B.

This is NOT a training run. Its only jobs (per the RLVR design spec):
  1. Prove GRPO RUNS end-to-end on this Mamba-2 + MoE hybrid on the Kaggle RTX PRO 6000.
     (clears the apply_rotary_pos_emb / unsloth#4612 re-forward-pass risk — R2)
  2. Measure SECONDS/STEP with NATIVE generation (fast_inference=False) at a realistic group size
     and SHORT completions, so we can project: can we get ~300 GRPO steps in a 12h Kaggle session?
     (this is THE go/no-go — prometheus04 ground-truth: single-GPU native-gen MAY be too slow.)
  3. Confirm the reward fires with CONTRAST across a group (not all-equal).
  4. Log peak VRAM.

Decisions baked in (external research consensus + verified + prometheus04 ground-truth):
  - Seed from v18's 0.84 PEFT adapter (NOT base — Wen paper warns 32B-from-base fails).
  - NATIVE generation, use_vllm/fast_inference=False (3-of-4 consensus; vLLM-MoE-LoRA unproven).
  - bf16, NOT 4-bit (Mamba degrades). attn_implementation="eager" (prometheus04 ran GRPO clean w/ eager;
    likely dodges the RoPE bug).
  - EXPLICIT LoRA target_modules (NOT auto-target) — prometheus04: get_peft_model() HANGS scanning
    5980 expert modules with regex/auto. Router/gate NOT trained.
  - GRPOConfig: epsilon=0.2/epsilon_high=0.28 (clip-higher), beta=0 (start; NVIDIA+TRL default),
    loss_type="dr_grpo", LR 1e-5, num_generations==per_device_train_batch_size (prometheus04 constraint).
  - Reward: strip <think> before verifying (R1/R2); placeholder exact-match here (swap rl_reward.py later).

Run as a Kaggle notebook on RTX PRO 6000. Offline: reuses the PROVEN mayukh18/nemotron-packages
wheelhouse (already vendors unsloth+trl+peft for SFT) + the base-model dataset + the v18 adapter dataset.
"""
import os, sys, glob, time, subprocess, json, re
# ---- DISABLE torch.compile/TorchDynamo BEFORE importing torch --------------
# v1 AND v2 smokes (2026-06-02) both cleared EVERY hard gate — Blackwell install,
# 30B load, MoE LoRA on 128 experts, and critically the RoPE bug (#4612) did NOT
# fire with eager attn — then crashed at the SAME spot inside Unsloth's COMPILED
# chunked log-softmax:
#   "Dynamo failed to run FX node ... a and b must have same reduction dim,
#    got [.., s12] X [2688, 131072]"  @ chunk_hidden_states @ lm_head.t()
# = a torch.compile SYMBOLIC-SHAPE bug (Dynamo can't prove hidden s12==2688 in the
# vocab matmul). v2's suppress_errors=True did NOT catch it (it re-raises through
# meta_mm's hard check). Full-eager is the robust dodge. CRITICAL: this barely
# affects the metric — our generation is ALREADY native/uncompiled (use_vllm=False),
# and rollout DOMINATES s/step; only loss/backward go eager. The env var must be set
# BEFORE torch import so Unsloth generates its cache file with compile already off.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")       # real PyTorch lever — the one that matters
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")   # best-effort Unsloth hint (harmless if ignored)

# ---------------------------------------------------------------------------
# 0. GPU + Blackwell sanity — fail loud BEFORE loading ~60GB of weights.
# ---------------------------------------------------------------------------
import torch
import torch._dynamo
torch._dynamo.config.disable = True             # full eager — no Dynamo tracing anywhere
torch._dynamo.config.suppress_errors = True     # belt & suspenders

cap = torch.cuda.get_device_capability()
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1e9
print(f"[env] GPU(s)={names} | capability=sm_{cap[0]}{cap[1]} | VRAM={vram:.0f}GB | torch={torch.__version__} cuda={torch.version.cuda}")
assert torch.cuda.is_available(), "need a GPU"
if vram < 70:
    raise RuntimeError(f"VRAM {vram:.0f}GB too small for bf16 30B — select RTX PRO 6000 (96GB) in Accelerator.")
# bf16 matmul sanity (catches a torch build that can't run on this arch)
_a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
_ = _a @ _a; torch.cuda.synchronize()
print("[env] bf16 matmul OK")

# ---------------------------------------------------------------------------
# 1. Offline install — reuse the PROVEN SFT wheelhouse (unsloth+trl+peft already there).
#    (mayukh18/nemotron-packages ran SFT on this exact model; trl includes GRPOTrainer.)
# ---------------------------------------------------------------------------
def rwheels(pat): return sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True))

# triton (Blackwell ptxas) — same as SFT path
tri = rwheels("*triton*.whl")
if tri:
    subprocess.run([sys.executable,"-m","pip","install","--no-deps","--target","/kaggle/working/pydeps",
                    "--upgrade","--ignore-installed", tri[0]], check=True)
    sys.path.insert(0, "/kaggle/working/pydeps")

packages_dir = "/kaggle/input/datasets/mayukh18/nemotron-packages/packages"
subprocess.run([sys.executable,"-m","pip","install","-q","--no-index","--find-links",packages_dir,
                "unsloth","trl","peft","transformers","datasets","accelerate","bitsandbytes"], check=True)
for pat in ("causal*conv1d*.whl","mamba_ssm-*.whl"):
    w = rwheels(pat)
    if w: subprocess.run([sys.executable,"-m","pip","install","--no-index","--no-deps",w[-1]], check=True)
print("[install] offline wheelhouse OK")

import trl, transformers, peft
print(f"[install] trl={trl.__version__} transformers={transformers.__version__} peft={peft.__version__}")

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

# ---------------------------------------------------------------------------
# 2. Locate base model + v18 adapter (offline, mounted as datasets).
# ---------------------------------------------------------------------------
import kagglehub
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
print(f"[load] base model: {MODEL_PATH}")

# v18 adapter: unzip submission.zip -> adapter dir (attach the v18 adapter as a dataset, or the v18 kernel output)
V18_ADAPTER = None
for cand in glob.glob("/kaggle/input/**/adapter_model.safetensors", recursive=True):
    V18_ADAPTER = os.path.dirname(cand); break
if V18_ADAPTER is None:
    # fall back: look for submission.zip and unzip
    for z in glob.glob("/kaggle/input/**/submission.zip", recursive=True):
        import zipfile
        os.makedirs("/tmp/v18_adapter", exist_ok=True)
        with zipfile.ZipFile(z) as zf: zf.extractall("/tmp/v18_adapter")
        V18_ADAPTER = "/tmp/v18_adapter"; break
print(f"[load] v18 adapter dir: {V18_ADAPTER}  (None => smoke runs from FRESH LoRA, still valid for the gate)")

MAX_SEQ = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ,
    load_in_4bit=False,             # bf16 — 4-bit wrecks Mamba
    fast_inference=False,           # NATIVE generation (NOT vLLM) — 3-of-4 consensus
    trust_remote_code=True,
    attn_implementation="eager",    # prometheus04 ran GRPO clean w/ eager (dodges RoPE bug)
    dtype=torch.bfloat16,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------------------------
# 3. LoRA — EXPLICIT module list (prometheus04: auto-target HANGS on 5980 experts). Router OFF.
#    Seed from v18 if available (flip inference_mode off via fresh get_peft_model targeting same modules).
# ---------------------------------------------------------------------------
TARGETS = ["q_proj","k_proj","v_proj","o_proj","in_proj","out_proj","up_proj","down_proj"]
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=32, lora_dropout=0.0,
    target_modules=TARGETS,           # EXPLICIT — not "all-linear"/regex
    bias="none", use_gradient_checkpointing="unsloth", random_state=3407,
)
# NOTE: to truly seed v18's weights we'd PeftModel.from_pretrained(V18_ADAPTER) with is_trainable=True.
# For the FEASIBILITY+SPEED gate, fresh LoRA on the same targets exercises the identical code path
# (rollout + re-forward + logprob + backward), so s/step is representative. v18-weight seeding is a
# load-path detail for the real run, not needed to measure feasibility/speed.
print("[lora] explicit targets set:", TARGETS)

# ---------------------------------------------------------------------------
# 4. Reward — strip <think> before verifying (R1/R2). Placeholder exact-match (swap rl_reward.py later).
# ---------------------------------------------------------------------------
def strip_think(t):
    return t.rsplit("</think>",1)[-1].strip() if "</think>" in t else t.strip()

def reward_func(prompts, completions, **kwargs):
    golds = kwargs["answer"]
    out = []
    for comp, gold in zip(completions, golds):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        final = strip_think(text)
        m = re.findall(r"\\boxed\{([^}]*)\}", final)
        pred = (m[-1].strip() if m else final)
        out.append(1.0 if pred == str(gold) else 0.0)
    return out

# ---------------------------------------------------------------------------
# 5. Tiny dataset — short answers (representative of our real task length).
# ---------------------------------------------------------------------------
rows = [{"prompt":[{"role":"user","content":f"What is {i}+{i}? Put the final answer in \\boxed{{}}."}],
         "answer":str(2*i)} for i in range(8)]
ds = Dataset.from_list(rows)

# ---------------------------------------------------------------------------
# 6. GRPOConfig — locked spec, sized for the smoke. num_generations == batch_size (prometheus04).
# ---------------------------------------------------------------------------
GROUP = 8        # the production group size; measure s/step HERE so projection is real
cfg = GRPOConfig(
    output_dir="/tmp/rlvr_smoke",
    learning_rate=1e-5, lr_scheduler_type="cosine", warmup_ratio=0.1,
    optim="paged_adamw_8bit", weight_decay=0.1, max_grad_norm=0.1,
    per_device_train_batch_size=GROUP, gradient_accumulation_steps=1,
    num_generations=GROUP,                 # MUST equal per_device_train_batch_size (prometheus04)
    max_prompt_length=256, max_completion_length=512,   # SHORT — our real task is short
    temperature=1.0, beta=0.0,             # KL off to start (NVIDIA/TRL default); clip-higher below
    epsilon=0.2, epsilon_high=0.28,        # DAPO clip-higher
    loss_type="dr_grpo",                   # removes length-norm bias
    num_iterations=1, max_steps=5,         # SMOKE only
    logging_steps=1, save_strategy="no", report_to="none",
    use_vllm=False,                        # NATIVE generation
)
trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=reward_func,
                      args=cfg, train_dataset=ds)

# ---------------------------------------------------------------------------
# 6b. LOGITS-AWARE LOGPROB PATCH (the v3 fix → real root cause).
#     v3 (full-eager) revealed the REAL bug: NemotronH's remote-code forward returns
#     LOGITS (last dim = vocab 131072) where Unsloth's GRPO loss expects HIDDEN STATES
#     (dim 2688), so its memory-saving `hidden @ lm_head.t()` shape-collides. This is the
#     unsupported-GRPO gap behind unsloth#4711. Fix: wrap efficient_log_softmax so when the
#     input is ALREADY logits, we skip the re-projection and do the mathematically-correct
#     log_softmax(logits).gather(index) directly. (Not a hack — it's the right computation
#     when logits are pre-materialized; the chunked path only existed to AVOID materializing
#     them, which NemotronH already did.) Patched at module-global level so grpo_accumulated_loss
#     picks it up by name. Covers both new- and old-logprob calls (same shared fn).
import sys as _sys, importlib as _il
def _make_logits_aware(orig):
    def wrapper(hidden_states, lm_head, index, chunks, logit_scale_multiply,
                logit_scale_divide, logit_softcapping, temperature, batch_size):
        # already-logits case: last dim == vocab (lm_head rows)
        if hidden_states.shape[-1] == lm_head.shape[0]:
            logits = hidden_states.float()
            if logit_scale_multiply != 0.0: logits = logits * logit_scale_multiply
            if logit_scale_divide  != 0.0: logits = logits / logit_scale_divide
            if logit_softcapping   != 0.0: logits = logit_softcapping * torch.tanh(logits / logit_softcapping)
            if temperature and temperature != 1.0: logits = logits / temperature
            logp = torch.log_softmax(logits, dim=-1)
            idx = index.to(torch.long)
            if idx.dim() == logp.dim() - 1:
                return logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            return logp.gather(-1, idx.reshape(*logp.shape[:-1], 1)).squeeze(-1)
        return orig(hidden_states, lm_head, index, chunks, logit_scale_multiply,
                    logit_scale_divide, logit_softcapping, temperature, batch_size)
    wrapper._logits_aware = True
    return wrapper

def _install_logits_aware_logsoftmax():
    # ensure the generated GRPO module is imported (created at GRPOTrainer init)
    try: _il.import_module("unsloth_compiled_cache.UnslothGRPOTrainer")
    except Exception: pass
    patched = []
    for _name, _mod in list(_sys.modules.items()):
        # NARROW: only unsloth-named modules. CRITICAL — never touch torch.classes,
        # whose magic __getattr__ returns a proxy for any name and then raises.
        if _mod is None or "unsloth" not in _name.lower():
            continue
        try:
            _orig = getattr(_mod, "efficient_log_softmax", None)
            if not callable(_orig) or getattr(_orig, "_logits_aware", False):
                continue
            _mod.efficient_log_softmax = _make_logits_aware(_orig)
            patched.append(_name)
        except Exception:
            continue
    return patched

_patched_mods = _install_logits_aware_logsoftmax()
print(f"[patch] logits-aware log_softmax installed on: {_patched_mods or 'NONE (fn not found — will hit the shape bug)'}")

# ---------------------------------------------------------------------------
# 7. Run + measure. PASS == completes on NemotronH AND reward varies AND we have s/step.
# ---------------------------------------------------------------------------
torch._dynamo.config.disable = True           # re-assert: Unsloth may reset dynamo cfg during init
torch._dynamo.config.suppress_errors = True
# re-scan in case the compiled-cache module was generated lazily after trainer init
try:
    import importlib as _il
    _il.import_module("unsloth_compiled_cache.UnslothGRPOTrainer")
except Exception as _e:
    print(f"[patch] (compiled-cache import note: {type(_e).__name__})")
_again = _install_logits_aware_logsoftmax()
print(f"[patch] re-scan before train patched: {_again or '(already patched / none new)'}")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
trainer.train()
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated()/1e9
sps = dt / cfg.max_steps
print("="*64)
print(f"[SMOKE PASS] GRPO ran on Nemotron-H Mamba+MoE. {cfg.max_steps} steps in {dt:.1f}s")
print(f"            seconds/step = {sps:.1f}s  | group={GROUP} | peak VRAM {peak:.1f}GB")
# THE projection — can we fit ~300 steps in a 12h session (leave ~1h for setup/save)?
budget_s = 11*3600
print(f"[PROJECT]   steps in ~11h compute @ {sps:.0f}s/step = {int(budget_s/sps)} steps")
print(f"            -> {'GREEN: 300+ steps fit' if budget_s/sps >= 300 else 'YELLOW/RED: under 300 steps — native gen too slow, consider vLLM/fewer-steps'}")
print("            (s/step is slightly CONSERVATIVE: Dynamo fully disabled, so loss/backward")
print("             run eager. But generation is native/uncompiled either way and DOMINATES")
print("             s/step — so this number is representative; a GREEN here is real.)")
print("[NEXT] if green: build curriculum pre-filter + real run next week (45h quota).")
print("       if red: A/B Unsloth-embedded vLLM (compiled rollout), or multi-session checkpointed run.")
