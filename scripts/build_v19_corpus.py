#!/usr/bin/env python3
"""build_v19_corpus.py — v19 corpus: v18 base, but the 3 weak categories get PROCEDURAL traces.

Single-variable change vs v18 (decided 2026-06-01, EXPERIMENTS row 21 diagnosis):
  v18 (dgxchen deep traces + our verified additions) plateaued at 0.84 with final loss 0.50 =
  the model MEMORIZES stated rules, doesn't learn to DERIVE them. v19 swaps the "oracle" traces
  for PROCEDURAL-SEARCH traces (show the test-and-eliminate the solver actually does) on the 3
  weak categories — bit_manipulation, numeral_system, transformation_rules — each critique-hardened
  (explicit failed tests, vertical/stepwise arithmetic, uniform style, grounded hypotheses).

Composition decision: REPLACE dgxchen's deep bit_manip/numeral traces with our procedural ones
  (uniform single style — avoids the v14 dual-template collapse; clean attribution). KEEP dgxchen's
  cipher/gravity/unit_conversion/cryptarithm/equation byte-identical (those scored 0.84, not weak).

  v19 categories (our procedural solvers, 100% solver-verified, answer==gold gated):
    - bit_manipulation     -> solvers.bit_manipulation_v19
    - numeral_system/numeral -> solvers.roman_numeral_v19
    - transformation_rules -> solvers.transformation_rules_v19
  KEEP from dgxchen (byte-identical v13 format): cipher, gravity, unit_conversion,
    cryptarithm_deduce, cryptarithm_guess, equation_numeric_deduce, equation_numeric_guess.

Output: data/v19-corpus.jsonl, records {messages:[user,assistant], type} — same loader as v18.
"""
import json, re, sys, os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from solvers import bit_manipulation_v19, roman_numeral_v19, transformation_rules_v19

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DG_CSV = os.path.join(ROOT, 'data/dgxchen-corpus/problem_ids_matched.csv')
TRAIN = os.path.join(ROOT, 'data/train.csv')
OUT = os.path.join(ROOT, 'data/v19-corpus.jsonl')

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# dgxchen categories we KEEP verbatim (not weak; scored 0.84)
# KEEP dgxchen's traces for these (his coverage >= ours, OR he's the only source).
# unit_conversion REMOVED from this set (2026-06-01, reviewer critique #2): our solver
# verifies 83% vs dgxchen's 67%, and ours is 100%-verified vs his label-only. Swapping in
# ours both raises coverage AND removes unverified traces (corpus-integrity fix). Our unit
# solver already emits long-division arithmetic (not oracle-style), so no procedural rewrite
# needed — it's a clean verified swap.
KEEP_DG = {'cipher', 'gravity',
           'cryptarithm_deduce', 'cryptarithm_guess',
           'equation_numeric_deduce', 'equation_numeric_guess'}

# train.csv prompt-signal -> our solver (categories we REPLACE with verified traces).
# bit/numeral/transformation use the _v19 PROCEDURAL variants (rewritten to show the search);
# unit_conversion uses the current solver (already long-arithmetic, verified-only swap).
from solvers import unit_conversion
V19_SOLVERS = [
    ('bit manipulation', bit_manipulation_v19, 'bit_manipulation'),
    ('numeral system', roman_numeral_v19, 'numeral_system'),
    ('transformation rules', transformation_rules_v19, 'transformation_rules'),
    ('secret unit', unit_conversion, 'unit_conversion'),
]


def v13_record(prompt, answer, cot, rtype):
    """Wrap a reasoning string into v13's exact record format (assistant ends </think>+boxed)."""
    cot = str(cot)
    if not cot or cot == 'nan' or len(cot.strip()) < 5:
        return None
    cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()
    user = str(prompt) + PROMPT_SUFFIX
    assistant = cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"
    return {'messages': [{'role': 'user', 'content': user},
                         {'role': 'assistant', 'content': assistant}], 'type': str(rtype)}


def main():
    records = []
    counts = {}

    # ---- 1) KEEP dgxchen's strong categories byte-identical (v13 format) ----
    dg = pd.read_csv(DG_CSV)
    for _, r in dg.iterrows():
        if r['type'] not in KEEP_DG:
            continue  # dgxchen's bit_manip/numeral are REPLACED by our procedural below
        rec = v13_record(r['prompt'], r['answer'], r['generated_cot'], r['type'])
        if rec is None:
            continue
        records.append(rec)
        counts[f"dgxchen:{r['type']}"] = counts.get(f"dgxchen:{r['type']}", 0) + 1

    # ---- 2) REPLACE the 3 weak categories with our v19 PROCEDURAL traces ----
    train = pd.read_csv(TRAIN)
    for signal, solver, typename in V19_SOLVERS:
        sub = train[train['prompt'].str.contains(signal, case=False, na=False)]
        kept = dropped_wrong = dropped_unsolved = 0
        for _, row in sub.iterrows():
            sol = solver.solve(str(row['prompt']))
            if not sol.success:
                dropped_unsolved += 1
                continue
            # 100%-verified gate: only keep solver-correct traces
            if sol.answer != str(row['answer']).strip():
                dropped_wrong += 1
                continue
            rec = v13_record(row['prompt'], sol.answer, sol.reasoning, typename)
            if rec is None:
                continue
            records.append(rec)
            kept += 1
        counts[f"v19:{typename}"] = kept
        counts[f"  dropped_wrong:{typename}"] = dropped_wrong
        counts[f"  dropped_unsolved:{typename}"] = dropped_unsolved

    with open(OUT, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    print("=== v19 corpus built ===")
    for k, v in counts.items():
        print(f"  {k:34s}: {v}")
    print(f"  {'TOTAL records':34s}: {len(records)}")
    print(f"  written -> {OUT}")

    # format self-check (no v14 dual-template mix)
    end_ok = sum(1 for r in records if r['messages'][1]['content'].rstrip().endswith('}')
                 and '</think>' in r['messages'][1]['content'])
    print(f"\n  format check: {end_ok}/{len(records)} end with </think>+\\boxed{{}} "
          f"({'UNIFORM ✓' if end_ok == len(records) else 'MIXED ✗'})")
    from collections import Counter
    print("  by type:", dict(sorted(Counter(r['type'] for r in records).items())))


if __name__ == '__main__':
    main()
