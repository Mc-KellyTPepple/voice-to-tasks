"""Word error rate with a small, documented text normaliser, plus a bootstrap interval over utterances."""
from __future__ import annotations

import random
import re


def _digits_to_words(text: str) -> str:
    try:
        from num2words import num2words
    except ImportError:
        return text
    return re.sub(r"\d+", lambda m: " " + num2words(int(m.group())) + " ", text)


def normalize(text: str, spell_numbers: bool = True) -> str:
    """Lowercase, spell out digits, drop punctuation (keep apostrophes inside words), collapse spaces."""
    t = text.lower().replace("’", "'").replace("-", " ")
    if spell_numbers:
        t = _digits_to_words(t)
    t = re.sub(r"[^a-z' ]", " ", t.replace("-", " "))
    t = re.sub(r"(^|\s)'+|'+(\s|$)", " ", t)
    return " ".join(t.split())


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def utterance_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    return edit_distance(ref, hyp), len(ref)


def corpus_wer(pairs: list[tuple[int, int]]) -> float:
    errs, words = sum(e for e, _ in pairs), sum(n for _, n in pairs)
    return errs / words if words else 0.0


def bootstrap_ci(pairs: list[tuple[int, int]], n: int = 2000, seed: int = 1, alpha: float = 0.05) -> tuple[float, float]:
    if not pairs:
        return 0.0, 0.0
    rng = random.Random(seed)
    vals = sorted(corpus_wer([pairs[rng.randrange(len(pairs))] for _ in pairs]) for _ in range(n))
    return vals[int(n * alpha / 2)], vals[int(n * (1 - alpha / 2)) - 1]
