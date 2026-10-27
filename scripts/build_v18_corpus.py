#!/usr/bin/env python3
"""
build_v18_corpus.py — v18 "targeted hybrid" training corpus.

Design (decided 2026-05-31, see EXPERIMENTS.md row 20 diagnosis):
  v17 (faithful kuangyicheng replication) scored 0.58 — a regression. Root cause,
  confirmed by near-controlled on-disk comparison: v13 (0.84) and v17 (0.58) share
  architecture exactly; the ONLY major variable was DATA. v17's "real x2" supervision
  is reasoning-FREE (16-word boilerplate + bare \boxed answers) which suppresses the
  base model's reasoning. dgxchen's data (v13=0.84) is real multi-step CoT (mean 1156 w).

  v18 = dgxchen's PROVEN 0.84 corpus  +  our 100%-solver-verified traces, but ONLY
  where additive/non-redundant, all reformatted to dgxchen's EXACT record format so
  there is no v14-style dual-template collapse. Single rich-CoT record per problem
  (NO x2 doubling — the answer-only doubling is what we proved hurts).

  Additions (verified by per-category depth analysis, EXPERIMENTS row 20):
    - +506 transformation_rules : dgxchen has ZERO coverage (~16% of test set untrained)
    - +N    ours-only problem_ids : verified-correct coverage dgxchen lacks
  We KEEP dgxchen's cipher/bit_manipulation/numeral (his are deeper) — no swaps.

Output: data/v18-corpus.jsonl  (records: {messages:[user,assistant], type})
        format byte-identical to v13's record builder so the v13/v17 notebook loads it as-is.
"""
import json, re, sys, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DG_CSV   = os.path.join(ROOT, 'data/dgxchen-corpus/problem_ids_matched.csv')
OURS_JSONL = os.path.join(ROOT, 'data/training-data.jsonl')
OUT      = os.path.join(ROOT, 'data/v18-corpus.jsonl')

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

def v13_record(prompt, answer, cot, rtype):
    """Build a record byte-identical to v13's data builder (the 0.84 format)."""
    cot = str(cot)
    if not cot or cot == 'nan' or len(cot.strip()) < 5:
        return None
    cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()   # strip any inline boxed
    user_content = str(prompt) + PROMPT_SUFFIX
    assistant_content = cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"
    return {'messages': [
        {'role': 'user', 'content': user_content},
        {'role': 'assistant', 'content': assistant_content},
    ], 'type': str(rtype)}

def main():
    dg = pd.read_csv(DG_CSV)
    dg['id8'] = dg['id'].astype(str).str.zfill(8)
    dg_ids = set(dg['id8'])

    ours = pd.DataFrame([json.loads(l) for l in open(OURS_JSONL) if l.strip()])
    ours['id8'] = ours['problem_id'].astype(str).str.zfill(8)
    # integrity: every ours record must be solver-verified (answer == gold)
    bad = (ours['answer'].astype(str).str.strip() != ours['gold'].astype(str).str.strip()).sum()
    assert bad == 0, f"{bad} ours records are NOT solver-verified (answer != gold) — abort"

    records = []
    counts = {'dgxchen_base': 0, 'ours_transformation_rules': 0, 'ours_only_problems': 0, 'skipped_empty_cot': 0}

    # 1) dgxchen base — verbatim v13 format
    for _, r in dg.iterrows():
        rec = v13_record(r['prompt'], r['answer'], r['generated_cot'], r['type'])
        if rec is None:
            counts['skipped_empty_cot'] += 1; continue
        records.append(rec); counts['dgxchen_base'] += 1

    # 2) ours: transformation_rules (dgxchen has NONE) — all of them
    tr = ours[ours['category'] == 'transformation_rules']
    for _, r in tr.iterrows():
        rec = v13_record(r['prompt'], r['answer'], r['reasoning'], 'transformation_rules')
        if rec is None:
            counts['skipped_empty_cot'] += 1; continue
        records.append(rec); counts['ours_transformation_rules'] += 1

    # 3) ours: problems dgxchen does NOT cover (additive coverage), EXCLUDING transformation
    #    (already added) — verified-correct traces for problem_ids dgxchen lacks.
    ours_only = ours[(~ours['id8'].isin(dg_ids)) & (ours['category'] != 'transformation_rules')]
    for _, r in ours_only.iterrows():
        rec = v13_record(r['prompt'], r['answer'], r['reasoning'], r['category'])
        if rec is None:
            counts['skipped_empty_cot'] += 1; continue
        records.append(rec); counts['ours_only_problems'] += 1

    with open(OUT, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    print("=== v18 corpus built ===")
    for k, v in counts.items():
        print(f"  {k:28s}: {v}")
    print(f"  {'TOTAL records':28s}: {len(records)}")
    print(f"  written -> {OUT}")

    # format self-check: every assistant content must end with the boxed close, exactly once
    import collections
    end_ok = sum(1 for r in records if r['messages'][1]['content'].rstrip().count('\\boxed{') >= 1
                 and r['messages'][1]['content'].rstrip().endswith('}'))
    has_think_close = sum(1 for r in records if '</think>' in r['messages'][1]['content'])
    print("\n=== format self-check (must avoid v14 dual-template mix) ===")
    print(f"  records ending in \\boxed{{...}}: {end_ok}/{len(records)}")
    print(f"  records with </think> close   : {has_think_close}/{len(records)}")
    bytype = collections.Counter(r['type'] for r in records)
    print(f"  by type: {dict(sorted(bytype.items()))}")

if __name__ == '__main__':
    main()
