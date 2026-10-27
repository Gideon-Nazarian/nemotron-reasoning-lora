"""
Reusable held-out eval for Track A SFT variants (v18 baseline, v20 masked, future A2/A3).

held-out = train.csv problems whose prompt is NOT in the v18 training corpus (never trained on)
=> a real (category-skewed) local proxy. Reports overall + per-category exact-match accuracy.

CPU-only parts (build_heldout_sample, extract_answer, category) are unit-tested locally.
The GPU part (score_adapter) runs in the Kaggle notebook against a loaded model+tokenizer.
"""
import re, json, ast, random
from collections import defaultdict, Counter

# the exact suffix v18/v5 trained with (verified from the corpus)
PROMPT_SUFFIX = " Please put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
IM_END = "<|im_end|>"
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_norm = lambda s: " ".join(str(s).split())


def category(answer):
    a = str(answer).strip()
    if re.fullmatch(r"[01]{8}", a):            return "bit_manipulation"
    if re.fullmatch(r"[IVXLCDM]+", a):         return "roman_numeral"
    if re.fullmatch(r"-?\d+(\.\d+)?", a):      return "numeric_int" if "." not in a else "numeric_decimal"
    if re.fullmatch(r"[a-z]+( [a-z]+)*", a):   return "cipher_phrase"
    return "other"


def extract_answer(completion):
    """Last \\boxed{} after </think> if present, else last \\boxed{} anywhere, else post-think text."""
    seg = completion.split("</think>", 1)[1] if "</think>" in completion else completion
    seg = seg.split(IM_END, 1)[0]
    in_seg = _BOXED.findall(seg)
    if in_seg:
        return in_seg[-1].strip()
    anywhere = _BOXED.findall(completion)
    if anywhere:
        return anywhere[-1].strip()
    return seg.strip()


def compute_heldout(train_csv_path, corpus_path):
    """Return list of dict(id,prompt,answer) for train problems NOT in the corpus (untrained)."""
    import pandas as pd
    df = pd.read_csv(train_csv_path)
    pk = {_norm(r.prompt): r for r in df.itertuples()}
    idx = defaultdict(list)
    for p in pk:
        idx[p[:60]].append(p)
    covered = set()
    with open(corpus_path) as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)["messages"]
            if isinstance(m, str):
                m = ast.literal_eval(m)
            c = _norm(m[0]["content"])
            for p in idx.get(c[:60], ()):
                if c.startswith(p):
                    covered.add(pk[p].id)
                    break
    held = [{"id": r.id, "prompt": r.prompt, "answer": str(r.answer)}
            for r in df.itertuples() if r.id not in covered]
    return held


def build_heldout_sample(train_csv_path, corpus_path, n=160, seed=3407):
    """Deterministic, category-stratified sample of the untrained held-out."""
    held = compute_heldout(train_csv_path, corpus_path)
    by = defaultdict(list)
    for r in held:
        by[category(r["answer"])].append(r)
    rng = random.Random(seed)
    per = max(1, n // max(1, len(by)))
    samp = []
    for k in sorted(by):
        v = by[k][:]
        rng.shuffle(v)
        samp += v[:per]
    rng.shuffle(samp)
    samp = samp[:n]
    return held, samp


def score_adapter(model, tokenizer, sample, max_new_tokens=1024, label="adapter", log_every=20):
    """GPU: greedy-generate + exact-match score. Returns (overall_acc, per_cat dict)."""
    import time, torch
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        pass
    per = defaultdict(lambda: [0, 0])
    ok = 0
    t0 = time.time()
    for i, r in enumerate(sample):
        msgs = [{"role": "user", "content": r["prompt"] + PROMPT_SUFFIX}]
        inp = tokenizer.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
        comp = tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=False)
        pred = extract_answer(comp)
        gold = str(r["answer"]).strip()
        hit = int(pred == gold)
        ok += hit
        k = category(gold)
        per[k][0] += hit
        per[k][1] += 1
        if (i + 1) % log_every == 0:
            print(f"[EVAL:{label}] {i+1}/{len(sample)} acc={ok/(i+1):.3f} ({(time.time()-t0)/60:.1f}m)", flush=True)
    overall = ok / max(1, len(sample))
    print("=" * 60)
    print(f"[EVAL RESULT] {label} | held-out n={len(sample)} | OVERALL acc={overall:.4f}")
    for k in sorted(per):
        c, n_ = per[k]
        print(f"   {k:18s} {c}/{n_} = {c/max(1,n_):.3f}")
    return overall, {k: (per[k][0], per[k][1]) for k in per}
