"""Incremental adaptation from user corrections.

``CorrectionMemory`` keeps M_user separate from the frozen global model
M_global: corrected utterances are stored as embeddings and, at prediction
time, a sufficiently similar stored correction shifts probability mass to the
corrected intent. The global model never changes, so previously learned
intents cannot be forgotten through this path.

``ReplayBuffer`` supports the fine-tuning strategies that *do* modify a
model, so their forgetting can be measured.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from ..text import normalize


class MemoryEncoder:
    """Key used to match a new utterance against stored corrections.

    Discourse wrappers ("por favor", "right away", ...) are removed first, and
    hashed character n-grams represent words never seen in training — exactly
    the personal expressions corrections are about.
    """

    def __init__(self, stop_phrases: list[str] | None = None, entities: dict[str, list[str]] | None = None,
                 n_features: int = 2 ** 18):
        def alt(phrases):
            ps = sorted({normalize(p) for p in phrases if normalize(p)}, key=len, reverse=True)
            return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, ps)) + r")(?!\w)") if ps else None

        self._stop = alt(stop_phrases or [])
        self._entities = [(alt(v), f" z{k} ") for k, v in (entities or {}).items() if v]
        self.vec = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=n_features,
                                     alternate_sign=False, norm="l2")

    _FILE = re.compile(r"\b[\w\-]+\.(?:docx|txt|png|xlsx|csv|zip|pptx|pdf|log)\b")
    _NUM = re.compile(r"\b\d+\b")

    def strip(self, text: str) -> str:
        t = normalize(text)
        t = self._FILE.sub(" zfile ", t)
        t = self._NUM.sub(" znum ", t)
        for rx, token in self._entities:
            if rx is not None:
                t = rx.sub(token, t)
        if self._stop is not None:
            t = self._stop.sub(" ", t)
        t = " ".join(t.split())
        return t or normalize(text)

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.vec.transform([self.strip(t) for t in texts]).toarray()

    @classmethod
    def from_registry(cls, registry) -> "MemoryEncoder":
        stops = [w.strip() for lang in registry.wrappers.values() for part in lang.values() for w in part]
        apps = [f for d in registry.apps.values() for lang in ("es", "en") for f in d.get(lang, [])]
        apps += [a for lang in registry.ambiguous_app_aliases.values() for a in lang]
        contacts = [c for lang in registry.fillers.values() for c in lang.get("contact", [])]
        return cls(stops, {"app": apps, "contact": contacts})


@dataclass
class CorrectionRecord:
    user: str
    text: str
    wrong_intent: str | None
    correct_intent: str
    timestamp: float
    uncertainty: float | None = None
    model_version: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class CorrectionMemory:
    def __init__(self, classes: list[str], embed_fn, threshold: float = 0.70, max_weight: float = 0.95):
        self.classes = list(classes)
        self.index = {c: i for i, c in enumerate(self.classes)}
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.max_weight = max_weight
        self._Z: dict[str, list[np.ndarray]] = {}
        self._y: dict[str, list[int]] = {}
        self.records: list[CorrectionRecord] = []

    def add(self, user: str, text: str, correct_intent: str, wrong_intent: str | None = None,
            uncertainty: float | None = None, model_version: str = "") -> None:
        z = self.embed_fn([text])[0]
        self._Z.setdefault(user, []).append(z)
        self._y.setdefault(user, []).append(self.index[correct_intent])
        self.records.append(CorrectionRecord(user, text, wrong_intent, correct_intent, time.time(),
                                             uncertainty, model_version))

    def size(self, user: str | None = None) -> int:
        if user is None:
            return sum(len(v) for v in self._y.values())
        return len(self._y.get(user, []))

    def nearest(self, Z: np.ndarray, users: list[str | None]) -> tuple[np.ndarray, np.ndarray]:
        """Max cosine similarity to the user's stored corrections and its label (-1 if none)."""
        sims = np.zeros(len(Z))
        labels = np.full(len(Z), -1)
        for u in set(users):
            if u is None or u not in self._Z:
                continue
            rows = np.flatnonzero(np.array([x == u for x in users]))
            S = Z[rows] @ np.stack(self._Z[u]).T
            j = S.argmax(axis=1)
            sims[rows] = S[np.arange(len(rows)), j]
            labels[rows] = np.asarray(self._y[u])[j]
        return sims, labels

    def blend(self, probs: np.ndarray, Z: np.ndarray, users: list[str | None]) -> tuple[np.ndarray, np.ndarray]:
        """p' = (1 - lam) p + lam * onehot(memory label), lam grows with similarity above threshold."""
        sims, labels = self.nearest(Z, users)
        lam = np.clip((sims - self.threshold) / max(1.0 - self.threshold, 1e-9), 0.0, 1.0) * self.max_weight
        lam[labels < 0] = 0.0
        out = np.asarray(probs, dtype=np.float64).copy()
        hit = lam > 0
        if hit.any():
            onehot = np.zeros_like(out[hit])
            onehot[np.arange(hit.sum()), labels[hit]] = 1.0
            out[hit] = (1 - lam[hit, None]) * out[hit] + lam[hit, None] * onehot
        return out, lam


class ReplayBuffer:
    """Bounded memory of past examples. Strategies: fifo, class_balanced, uncertainty, diversity."""

    STRATEGIES = ("fifo", "class_balanced", "uncertainty", "diversity")

    def __init__(self, capacity: int = 200, strategy: str = "class_balanced", seed: int = 0):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"unknown replay strategy {strategy!r}")
        self.capacity, self.strategy = capacity, strategy
        self.rng = np.random.default_rng(seed)
        self.texts: list[str] = []
        self.labels: list[str] = []
        self.unc: list[float] = []
        self.Z: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.texts)

    def add(self, text: str, label: str, uncertainty: float = 0.0, z: np.ndarray | None = None) -> None:
        self.texts.append(text)
        self.labels.append(label)
        self.unc.append(float(uncertainty))
        self.Z.append(z if z is not None else np.zeros(1))
        if len(self.texts) > self.capacity:
            self._evict()

    def _evict(self) -> None:
        if self.strategy == "fifo":
            i = 0
        elif self.strategy == "class_balanced":
            counts: dict[str, int] = {}
            for l in self.labels:
                counts[l] = counts.get(l, 0) + 1
            big = max(counts, key=counts.get)
            i = self.labels.index(big)
        elif self.strategy == "uncertainty":
            i = int(np.argmin(self.unc))            # keep the hard (uncertain/corrected) examples
        else:  # diversity: drop the example most similar to another one
            Z = np.stack(self.Z)
            S = Z @ Z.T
            np.fill_diagonal(S, -np.inf)
            i = int(S.max(axis=1).argmax())
        for lst in (self.texts, self.labels, self.unc, self.Z):
            lst.pop(i)

    def sample(self, n: int) -> tuple[list[str], list[str]]:
        if not self.texts:
            return [], []
        idx = self.rng.choice(len(self.texts), size=min(n, len(self.texts)), replace=False)
        return [self.texts[i] for i in idx], [self.labels[i] for i in idx]
