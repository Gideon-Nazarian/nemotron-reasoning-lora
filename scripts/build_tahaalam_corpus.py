#!/usr/bin/env python3
"""build_tahaalam_corpus.py — training corpus from a PUBLIC pre-logprob-filtered dataset.

Plan #2 (alternative to filtering our own data). Source:
  Kaggle dataset `tahaalam2009/nemotron-logprob`, file `filtered_merged_dataset.csv`
  (8,703 rows; cols: id, prompt, answer, type, generated_cot, base_loss).
  It is a logprob-filtered + category-REBALANCED training set: unlike our drop-approach,
  it KEEPS the hard categories (cryptarithm_*, transformation_rules) but caps them by loss.

Output: data/tahaalam-filtered-corpus.jsonl
  one JSON object per line, drop-in compatible with the v18 SFTTrainer pipeline:
    {"messages":[{"role":"user","content": <prompt>+PROMPT_SUFFIX},
                 {"role":"assistant","content": <cot ... </think>\\boxed{answer}>}],
     "type": <category>}

KEY FORMAT DECISION (verified against scripts/build_v18_corpus.py::v13_record and the
on-disk data/v18-corpus.jsonl, 2026-06-05):

  tahaalam's `generated_cot` is ALREADY very close to our target format:
    * 100% of rows already contain `\\boxed{}` (4-5 occurrences: the prompt-echo
      "put your final answer inside \\boxed{}", a "return the answer in \\boxed{}"
      narration line, "The answer in \\boxed{–} is \\boxed{<ANSWER>}", and — for the
      79.5% complete rows — a final standalone `\\boxed{<ANSWER>}` AFTER `</think>`.
    * 79.5% (6,923) already contain exactly one `</think>` (then the final boxed line).
    * 20.5% (1,780) are missing the trailing `</think>` + final boxed (truncated).

  The raw v13_record (the proven-0.84 builder) does NOT strip an existing `</think>`,
  only inline `\\boxed{}`. Applied verbatim to tahaalam it would emit a DOUBLE `</think>`
  on 79.5% of rows (`...</think>\n</think>\n\\boxed{ans}`) — byte-identical to the
  dgxchen/tong artifact already sitting in data/v18-corpus.jsonl (7,830/10,311 dbl-think).

  The task is explicit: match v13_record's output but WITHOUT double-wrapping `</think>`
  or `\\boxed{}` when already present. So this builder:
    1. Truncates the CoT at the FIRST `</think>` (drops the dup tag + the redundant
       post-think final boxed line; the answer is re-emitted canonically from `answer`).
    2. Strips inline `\\boxed{...}` (same regex as v13_record).
    3. Strips the trailing "I will now return the answer in ... / The answer in ... is"
       narration boilerplate left dangling after step 2 (the canonical clean
       solver-trace target in v18 — e.g. `...Result: 6644\n</think>\n\\boxed{6644}` —
       has no such narration tail).
    4. Appends the canonical `\n</think>\n\\boxed{<answer>}` exactly once.

  Result: every assistant message has EXACTLY one `</think>` and EXACTLY one `\\boxed{}`,
  ending `...\n</think>\n\\boxed{<answer>}` — matching the clean v18 record shape with no
  double-wrap. User content is identical to v13_record (prompt + PROMPT_SUFFIX).
"""
import json, re, os, collections
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = '/tmp/tah/results/filtered_merged_dataset.csv'
OUT  = os.path.join(ROOT, 'data/tahaalam-filtered-corpus.jsonl')

# byte-identical to build_v18_corpus.py / build_v19_corpus.py
PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# trailing narration boilerplate the source appends right before its </think>.
# After \boxed{} stripping it degrades to e.g.
#   "\n\nI will now return the answer in \nThe answer in  is \n"
# We remove it so the reasoning ends on its actual computed conclusion, matching the
# clean v18 solver-trace tail.  Anchored to END of (already </think>-truncated) string.
NARRATION_TAIL = re.compile(
    r'\s*I will now return the answer in\b.*\Z',
    flags=re.IGNORECASE | re.DOTALL,
)


def build_record(prompt, answer, cot, rtype):
    """tahaalam CoT -> v18-compatible record, single </think> + single \\boxed{} (no double-wrap)."""
    cot = str(cot)
    if not cot or cot == 'nan' or len(cot.strip()) < 5:
        return None
    # 1) truncate at the source </think> (drops the dup tag + redundant post-think boxed)
    idx = cot.find('</think>')
    if idx != -1:
        cot = cot[:idx]
    # 2) strip inline \boxed{...} (identical to v13_record)
    cot = re.sub(r'\\boxed\{[^}]*\}', '', cot)
    # 3) strip the dangling "I will now return the answer in ... is" narration tail
    cot = NARRATION_TAIL.sub('', cot)
    cot_cleaned = cot.rstrip()
    if len(cot_cleaned) < 5:
        return None
    user_content = str(prompt) + PROMPT_SUFFIX
    assistant_content = cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"
    return {'messages': [
        {'role': 'user', 'content': user_content},
        {'role': 'assistant', 'content': assistant_content},
    ], 'type': str(rtype)}


def main():
    df = pd.read_csv(SRC)
    records = []
    skipped_empty = 0
    for _, r in df.iterrows():
        rec = build_record(r['prompt'], r['answer'], r['generated_cot'], r['type'])
        if rec is None:
            skipped_empty += 1
            continue
        records.append(rec)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    print("=== tahaalam filtered corpus built ===")
    print(f"  source rows           : {len(df)}")
    print(f"  skipped (empty cot)   : {skipped_empty}")
    print(f"  TOTAL records written : {len(records)}")
    print(f"  written -> {OUT}")

    # ---- format self-check: EXACTLY one </think> + one \boxed, ending </think>+boxed ----
    end_ok = sum(1 for r in records
                 if r['messages'][1]['content'].rstrip().endswith('}')
                 and '</think>' in r['messages'][1]['content'])
    exactly_one_think = sum(1 for r in records if r['messages'][1]['content'].count('</think>') == 1)
    exactly_one_box   = sum(1 for r in records if r['messages'][1]['content'].count('\\boxed{') == 1)
    print("\n=== format self-check (must avoid the v14/dgxchen double-wrap) ===")
    print(f"  end with </think>+\\boxed{{}}    : {end_ok}/{len(records)}")
    print(f"  EXACTLY one </think>           : {exactly_one_think}/{len(records)}")
    print(f"  EXACTLY one \\boxed{{}}          : {exactly_one_box}/{len(records)}")
    uniform = (end_ok == len(records) == exactly_one_think == exactly_one_box)
    print(f"  -> {'UNIFORM, no double-wrap ✓' if uniform else 'NON-UNIFORM ✗ (investigate)'}")

    bytype = collections.Counter(r['type'] for r in records)
    print("\n=== per-category counts ===")
    for k, v in sorted(bytype.items()):
        print(f"  {k:28s}: {v}")


if __name__ == '__main__':
    main()
