"""Controlled text-level ASR error simulation.

Lets the language/decision layers be evaluated under transcription errors
without running an ASR thousands of times. Error operations are drawn per word
with probability ``rate``:

* phonetic substitution (Spanish: b/v, s/z/c, ll/y, silent h, g/j, accents;
  English: common homophone-like spellings),
* random character edit (substitute / delete / insert),
* word deletion, and merging with the next word.

These are approximations of ASR behaviour and are documented as such; they do
not reproduce the error distribution of any particular recogniser.
"""
from __future__ import annotations

import re

import numpy as np

_ES_RULES = [
    (r"v", "b"), (r"b", "v"), (r"z", "s"), (r"ce", "se"), (r"ci", "si"), (r"s", "z"),
    (r"ll", "y"), (r"y", "ll"), (r"^h", ""), (r"ge", "je"), (r"gi", "ji"), (r"x", "s"),
    (r"á", "a"), (r"é", "e"), (r"í", "i"), (r"ó", "o"), (r"ú", "u"), (r"qu", "k"), (r"rr", "r"),
]
_EN_RULES = [
    (r"ph", "f"), (r"ck", "k"), (r"ee", "ea"), (r"ou", "ow"), (r"tion", "shun"), (r"c", "k"),
    (r"s$", "z"), (r"wh", "w"), (r"ight", "ite"), (r"th", "d"), (r"oo", "u"), (r"y$", "ie"),
]
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _phonetic(word: str, lang: str, rng: np.random.Generator) -> str:
    rules = _ES_RULES if lang == "es" else _EN_RULES
    applicable = [(p, r) for p, r in rules if re.search(p, word)]
    if not applicable:
        return _char_edit(word, rng)
    p, r = applicable[int(rng.integers(len(applicable)))]
    matches = list(re.finditer(p, word))
    m = matches[int(rng.integers(len(matches)))]
    return word[:m.start()] + r + word[m.end():]


def _char_edit(word: str, rng: np.random.Generator) -> str:
    if not word:
        return word
    op = int(rng.integers(3))
    i = int(rng.integers(len(word)))
    c = _ALPHABET[int(rng.integers(len(_ALPHABET)))]
    if op == 0:
        return word[:i] + c + word[i + 1:]
    if op == 1 and len(word) > 1:
        return word[:i] + word[i + 1:]
    return word[:i] + c + word[i:]


def perturb(text: str, rate: float, rng: np.random.Generator, lang: str = "es") -> str:
    """Return ``text`` with simulated ASR errors at per-word probability ``rate``."""
    if rate <= 0:
        return text
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if rng.random() < rate:
            op = rng.random()
            if op < 0.55:
                out.append(_phonetic(w.lower(), lang, rng))
            elif op < 0.80:
                out.append(_char_edit(w.lower(), rng))
            elif op < 0.92 and len(words) > 1:
                pass  # deletion
            elif i + 1 < len(words):
                out.append(w + words[i + 1])
                i += 1
            else:
                out.append(w)
        else:
            out.append(w)
        i += 1
    return " ".join(out) if out else text


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein word error rate between two strings."""
    r, h = reference.split(), hypothesis.split()
    if not r:
        return float(len(h) > 0)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[len(r), len(h)] / len(r)
