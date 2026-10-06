"""Text normalisation shared by every model, so that train and test see the same form."""
from __future__ import annotations

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9.%]+")
_SPACES = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(text: str) -> str:
    """Lower-case, remove accents and punctuation (keeps digits, '.', '%')."""
    t = strip_accents(text.lower())
    t = _NON_ALNUM.sub(" ", t)
    t = re.sub(r"(?<!\w)\.|\.(?!\w)", " ", t)
    return _SPACES.sub(" ", t).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def contains_phrase(normalized_text: str, phrase: str) -> bool:
    """Whole-word phrase match on already-normalised text."""
    p = normalize(phrase)
    if not p:
        return False
    return re.search(rf"(?<!\w){re.escape(p)}(?!\w)", normalized_text) is not None
