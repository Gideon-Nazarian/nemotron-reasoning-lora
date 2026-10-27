"""Solver for `cipher_phrase` (substitution cipher) problems.

Prompt structure:
    In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
    btjwpzt nrffrib tep lobtpadrjb krtdrz -> student follows the mysterious potion
    ...
    Now, decrypt the following text: xzdqet wahib tep qrfwpz xpo

The cipher is a 1:1 substitution: each ciphertext letter maps consistently to
one plaintext letter (and vice versa). Build the mapping from example pairs by
aligning matching word lengths position-by-position, then apply to the question.

**v2 upgrade (Tong-style wordlist fallback):**

The previous solver fell over when the question contained ciphertext letters
that weren't in any example (971 of 1576 problems = 61.6% failed this way).
This version adds a `wonderland.txt`-driven fallback: for each unknown
ciphertext word in the question, find plaintext candidates that
  (a) match length,
  (b) match the positional letter-repetition pattern (e.g. cipher `tep` and
      plain `the` both have pattern `(0,1,2)`; cipher `tepft` and plain
      `apple` both have pattern `(0,1,1,2,3)`),
  (c) are consistent with the cipher_to_plain mapping derived so far AND
      don't introduce contradictions (no letter maps to two different
      plaintexts; no plaintext letter is used by two cipher letters).

Pick the first wonderland-list candidate, commit the new mappings, continue.

Expected unlock: ~900 of the 971 previously-failing problems should now solve,
assuming the gold answers' words come from the 77-word wonderland vocabulary
(verified empirically — see `notes/cipher-wordlist-coverage.md`).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .types import Solution

PAIR_RE = re.compile(r"^([a-z][a-z ]*?)\s*->\s*([a-z][a-z ]*?)\s*$", re.MULTILINE)
QUESTION_RE = re.compile(
    r"decrypt the following text:\s*([a-z ]+?)(?:\n|$)", re.IGNORECASE
)

_WORDLIST_PATH = Path(__file__).parent / "wonderland.txt"


@lru_cache(maxsize=1)
def _load_wonderland() -> tuple[str, ...]:
    """Load wonderland word list (sorted, cached)."""
    with _WORDLIST_PATH.open() as f:
        words = sorted(line.strip().lower() for line in f if line.strip())
    return tuple(words)


def _pattern(word: str) -> tuple[int, ...]:
    """Positional letter-repetition pattern. 'apple' -> (0,1,1,2,3)."""
    seen: dict[str, int] = {}
    out: list[int] = []
    for ch in word:
        if ch not in seen:
            seen[ch] = len(seen)
        out.append(seen[ch])
    return tuple(out)


def _candidates_for_word(
    cipher_word: str,
    cipher_to_plain: dict[str, str],
) -> list[str]:
    """Find wonderland plaintext words consistent with `cipher_word` under
    the current `cipher_to_plain` mapping."""
    target_len = len(cipher_word)
    target_pat = _pattern(cipher_word)
    mapped_plain = set(cipher_to_plain.values())
    out: list[str] = []
    for word in _load_wonderland():
        if len(word) != target_len:
            continue
        if _pattern(word) != target_pat:
            continue
        # Consistency with current mapping
        ok = True
        tentative: dict[str, str] = {}
        for cc, pc in zip(cipher_word, word):
            if cc in cipher_to_plain:
                if cipher_to_plain[cc] != pc:
                    ok = False
                    break
            else:
                # cc is unknown — its proposed plaintext (pc) must NOT already
                # be a known plaintext target (else two cipher letters would
                # map to the same plaintext)
                if pc in mapped_plain or (cc in tentative and tentative[cc] != pc):
                    ok = False
                    break
                tentative[cc] = pc
        if ok:
            out.append(word)
    return out


def solve(prompt: str) -> Solution:
    pairs = PAIR_RE.findall(prompt.lower())
    if len(pairs) < 2:
        return Solution(False, "", "", notes=f"only {len(pairs)} cipher pairs parsed")

    # --- Step 1: build cipher_to_plain from example pairs ---
    cipher_to_plain: dict[str, str] = {}
    for cipher_text, plain_text in pairs:
        c_words = cipher_text.split()
        p_words = plain_text.split()
        if len(c_words) != len(p_words):
            return Solution(
                False, "", "",
                notes=f"word count mismatch in pair: {cipher_text!r} -> {plain_text!r}",
            )
        for cw, pw in zip(c_words, p_words):
            if len(cw) != len(pw):
                return Solution(
                    False, "", "",
                    notes=f"word length mismatch: {cw!r} -> {pw!r}",
                )
            for c_ch, p_ch in zip(cw, pw):
                if c_ch in cipher_to_plain and cipher_to_plain[c_ch] != p_ch:
                    return Solution(
                        False, "", "",
                        notes=f"inconsistent mapping: {c_ch!r} -> {cipher_to_plain[c_ch]!r} vs {p_ch!r}",
                    )
                cipher_to_plain[c_ch] = p_ch

    # --- Step 2: extract question text ---
    q_match = QUESTION_RE.search(prompt)
    if not q_match:
        return Solution(False, "", "", notes="no question parsed")
    q_text = q_match.group(1).strip().lower()
    q_words = q_text.split()

    # --- Step 3: decode each question word; use wordlist for words with unknown letters ---
    decoded_words: list[str] = []
    unknown_word_resolutions: list[tuple[str, list[str], str]] = []
    # (cipher_word, candidates_considered, chosen_plaintext)

    for cw in q_words:
        # Try direct mapping
        all_known = all(c in cipher_to_plain for c in cw)
        if all_known:
            decoded_words.append("".join(cipher_to_plain[c] for c in cw))
            continue
        # Wordlist fallback
        candidates = _candidates_for_word(cw, cipher_to_plain)
        if not candidates:
            unknown_letters = sorted({c for c in cw if c not in cipher_to_plain})
            return Solution(
                False, "", "",
                notes=(
                    f"wordlist fallback failed for {cw!r}: "
                    f"unknown letters {unknown_letters}, no wonderland candidate matches"
                ),
            )
        # Pick first sorted candidate (deterministic)
        chosen = candidates[0]
        # Commit the new mappings
        for cc, pc in zip(cw, chosen):
            if cc not in cipher_to_plain:
                cipher_to_plain[cc] = pc
        unknown_word_resolutions.append((cw, candidates, chosen))
        decoded_words.append(chosen)

    answer = " ".join(decoded_words).strip()

    # --- Step 4: build reasoning trace ---
    first_cipher, first_plain = pairs[0]
    c_words = first_cipher.split()
    p_words = first_plain.split()
    first_word_lines = []
    for cw_e, pw_e in zip(c_words, p_words):
        letter_pairs = ", ".join(f"{c}→{p}" for c, p in zip(cw_e, pw_e))
        first_word_lines.append(f"  '{cw_e}' aligns with '{pw_e}': {letter_pairs}")
    first_derivation = "\n".join(first_word_lines)

    mapping_full = ", ".join(f"{c}→{p}" for c, p in sorted(cipher_to_plain.items()))

    application_lines = []
    for cw in q_words:
        per_letter = ", ".join(f"{ch}→{cipher_to_plain.get(ch, '?')}" for ch in cw)
        decoded = "".join(cipher_to_plain.get(ch, "?") for ch in cw)
        application_lines.append(f"  '{cw}' → {per_letter} → '{decoded}'")
    application = "\n".join(application_lines)

    reasoning_parts = [
        "Each ciphertext letter maps consistently to one plaintext letter. The mapping is "
        "built by aligning matching example pairs position-by-position.",
        f"From the first example, '{first_cipher}' -> '{first_plain}':",
        first_derivation,
        f"Reading mappings off all {len(pairs)} example pairs gives the substitution table: "
        f"{mapping_full}.",
    ]

    if unknown_word_resolutions:
        reasoning_parts.append(
            f"The question has {len(unknown_word_resolutions)} word(s) containing letters "
            "not seen in the examples. For each such word I look up plaintext candidates "
            "from the Wonderland word list that (a) match the cipher word's letter-repetition "
            "pattern, (b) match the cipher word's length, and (c) are consistent with the "
            "mapping derived so far."
        )
        for cw, cands, chosen in unknown_word_resolutions:
            pat = _pattern(cw)
            unknown_letters = sorted({c for c in cw if c not in cipher_to_plain or True})
            # show first 8 candidates so the trace doesn't explode
            cand_preview = cands[:8]
            cand_str = ", ".join(cand_preview) + (
                f", ...({len(cands)-8} more)" if len(cands) > 8 else ""
            )
            reasoning_parts.append(
                f"For cipher word '{cw}' (length {len(cw)}, pattern {pat}), "
                f"the consistent Wonderland candidates are: {cand_str}. "
                f"Choosing '{chosen}'."
            )

    reasoning_parts.append(
        f"Applying the final substitution table letter-by-letter to '{q_text}':\n{application}"
    )
    reasoning_parts.append(f"Concatenating the decoded words gives '{answer}'.")

    reasoning = "\n".join(reasoning_parts)

    return Solution(
        success=True, answer=answer, reasoning=reasoning,
        notes=(
            f"n_pairs={len(pairs)}, mapping_size={len(cipher_to_plain)}, "
            f"wordlist_fallback_used={len(unknown_word_resolutions)}"
        ),
    )
