# Public 0.87+ methods — ranked by published chance, with STRICT proof status (2026-06-04)

> Question: rank the PUBLIC (not-ours) notebooks/methods by their published chance of hitting 0.87,
> then mark — ONLY from our submission record + verified-faithful logs/cell outputs — whether WE have
> **definitively proved** it worked or didn't. "Attempted but our execution was off/broken" ≠ proof → BLANK.
> Not attempted / can't confirm → BLANK. Authoritative score source: `kaggle competitions submissions`.

## Ranked table

| # | Public method (NOT ours) | Published | Reproducible inputs? | Did WE DEFINITIVELY prove it hits 0.87? | Faithfulness evidence (why filled / why blank) |
|---|---|---|---|---|---|
| 1 | **kuangyicheng** "0.88 Final" — warm-start Tong **v27** + continue-train on train.csv + inline synthetic | **0.88** | YES (public adapter + self-contained data) | **BLANK — never faithfully tested** | The warm-start **never loaded** on our stack (v16 → PEFT zero-init, *verified by weight inspection*: lora_B exactly 0). We only ran the **no-warm-start** variant faithfully (v17, cell-6 **byte-identical** to theirs) → **0.58**. So the actual published method (with warm-start) is **untested**; only the from-base variant is disproven (0.58, 2026-05-31). |
| 2 | **hammadfarooq** — SVD-compress Tong v20 → rank-32 | **0.87** | YES (public adapter, deterministic, no training) | **PROVED IT DOES NOT — got 0.84–0.85** | FAITHFUL: ran the public notebook's **own 5 candidates** + submitted → 0.84 / 0.85 / 0.85 (2026-06-03→04). Definitive disproof of 0.87 for us. |
| 3 | **Kh0a** — category-rebalance + boxed-loss 5× + **PRIVATE** distilled traces | **0.87** | **NO** (private `llkh0a/…distiled` unreachable) | **BLANK — cannot run faithfully** | The method *requires* data we can't access. Our v12 (recipe **minus** private data) → 0.72 (2026-05-18) — not the method. Isolated levers tested: boxed-loss×5 → 0.84 (neutral, v16-old); rebalance → 0.76 (v21). None = the full method. |
| 4 | **asalhi** — tinker-wrapper of Tong v20 | 0.86 | YES | **PROVED IT WORKS at 0.86** (its claim; **not** 0.87) | FAITHFUL direct submit → **0.86** (2026-06-03 21:25). BANKED. Definitively a 0.86 method, not 0.87. |
| 5 | **dgxchen** — public adapter | 0.85 | YES | **PROVED IT WORKS at 0.84** (≈ its claim; **not** 0.87) | FAITHFUL direct submit of his actual adapter → **0.84** (2026-05-20). A 0.84/0.85 method, not 0.87. |
| — | Tong **v27** *direct* (our SVD converter re-pointed at v27) | (not a published method) | — | **BLANK — broken execution** | SVD-v27 → **0.51** (2026-06-04), but our v20-tuned converter mangled v27's keys (0.51 = untrained-base). Not a clean test of v27. |

## What this proves (the honest bottom line)

- **Of the 3 public methods claiming ≥ 0.87:**
  - **SVD (hammadfarooq)** — DEFINITIVELY DISPROVEN for us (0.84–0.85). Closed.
  - **Kh0a** — BLANK and effectively closed: unreachable (private data). Cannot ever run faithfully.
  - **kuangyicheng warm-start** — BLANK, and it is the **ONLY** published ≥0.87 method that is *both* (a) reproducible-input (public adapter + self-contained data) *and* (b) **not yet definitively tested by us** — because our warm-start LOAD broke (zero-init), so we never actually executed it.
- **Every faithfully-tested public method maxed at 0.86** (asalhi). We have **not** definitively proved *any* public method reaches 0.87 for us — and we have **not** definitively *disproven* the two we couldn't run faithfully (kuangyicheng-warm-start, Kh0a).
- ⇒ The single published 0.87+ path still genuinely open (reproducible **and** unclosed) is **kuangyicheng's warm-start, run with the load actually verified.** Kh0a is closed-by-unreachability; SVD is closed-by-disproof. This is exactly why the warm-start keeps surfacing as "the one stone left."
