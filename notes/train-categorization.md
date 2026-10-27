# Train.csv Problem Categorization

Generated: 2026-05-10 by `scripts/categorize_train.py`. Per-row results saved to `notes/train-categories.csv`.

## 5 puzzle types, evenly distributed

| # | Category | Count | % | Answer format | Prompt signal |
|---|---|---|---|---|---|
| 1 | numeric_decimal (gravity + unit conversion) | 3,191 | 33.6% | Decimal (e.g. `57.0`) | "gravitational constant", "secret unit", "secretly converted" |
| 2 | bit_manipulation | 1,602 | 16.9% | 8-bit binary string (e.g. `01100110`) | "bit manipulation rule", "8-bit binary" |
| 3 | cipher_phrase | 1,576 | 16.6% | multi-word lowercase phrase (e.g. `the curious mouse creates`) | "encryption rule", "cipher", "decrypt" |
| 4 | roman_numeral | 1,576 | 16.6% | Roman numerals (e.g. `XLVIII`) | "numeral system" |
| 5 | transformation_rules | 1,555 | 16.4% | Symbolic OR small integer (e.g. `@&`, `17`, `''#`) | "secret set of transformation rules is applied to equations" |

## Cross-tab is perfectly clean

Prompt-keyword classification matches answer-format classification 1:1 for the 4 well-structured types. Only `transformation_rules` has answers split between symbolic (~868) and numeric_int (~687) — but both come from the same prompt template, just with different operator semantics in the puzzle.

## Insight on `transformation_rules`

These are operator-redefinition puzzles. Example:
```
55"38 = "17
78^42 = 120
51^87 = 138
24^73 = 97
11$15 = 164
Now, determine the result for: 13^18
→ Gold: 31
```

Decoding: `78^42 = 120` means `78 + 42 = 120`, so `^` here means addition. Then `13^18 = 13 + 18 = 31` ✓.

So `transformation_rules` problems have **per-problem operator semantics** that the model must infer from the example pairs, then apply. Some operators yield numeric results, others yield symbolic results (character permutation, digit substitution, etc.). Genuinely hard.

## Implications for fine-tuning

1. **Stratified sampling matters.** If we sample uniformly across train.csv, the numeric_decimal class will dominate at 33.6%. For balanced training, sample equal counts per category.
2. **5 different answer formats means 5 different extraction strategies.** Important for both training-data formatting and downstream eval.
3. **Hardest categories (most room to improve via fine-tuning):**
   - `cipher_phrase` — base model derives mapping but runs out of tokens
   - `bit_manipulation` — base model explores patterns but doesn't converge
   - `transformation_rules` — base model rarely gets these right (purely speculative, need to verify)
   - Math problems converge most reliably (per our v4 eval)
4. **Per-category fine-tuning** is a possible advanced approach: train one adapter per puzzle type, ensemble at inference.

## Prompt length stats per category

```
                    min  median   mean  max  count
bit_manipulation    447   489.0  479.0  510   1602
cipher_phrase       265   370.0  370.0  485   1576
numeric_decimal     197   280.0  271.0  357   3191
numeric_int         178   195.0  195.0  212    687  (transformation_rules subset)
other               177   195.0  196.0  212    868  (transformation_rules subset)
roman_numeral       199   219.0  219.0  241   1576
```

`bit_manipulation` prompts are the longest (lots of example pairs), `transformation_rules` are shortest.
