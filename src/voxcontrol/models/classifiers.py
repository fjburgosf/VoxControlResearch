"""Intent classifiers: rules (B1), TF-IDF logistic regression, embedding softmax,
nearest prototype and a bootstrap ensemble (model disagreement)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..embeddings.lsa import LSAEmbedder, SparseTfidf
from ..intents.registry import IntentRegistry
from ..text import contains_phrase, normalize
from .base import IntentModel, softmax


class RuleModel(IntentModel):
    """Keyword baseline. Longer keyword phrases weigh more. No match => uniform."""
    name = "rules"

    def __init__(self, registry: IntentRegistry, scale: float = 4.0):
        super().__init__(registry.names)
        self.scale = scale
        self.keywords = {
            n: sorted({normalize(k) for kws in registry.spec(n).keywords.values() for k in kws})
            for n in registry.names
        }

    def fit(self, texts, labels):
        return self

    def scores(self, texts: list[str]) -> np.ndarray:
        S = np.zeros((len(texts), len(self.classes)))
        for i, t in enumerate(texts):
            nt = normalize(t)
            for j, c in enumerate(self.classes):
                S[i, j] = sum(len(k.split()) for k in self.keywords[c] if contains_phrase(nt, k))
        return S

    def logits(self, texts):
        return self.scale * self.scores(texts)


class TfidfLogReg(IntentModel):
    """Character+word TF-IDF with multinomial logistic regression."""
    name = "tfidf_logreg"

    def __init__(self, classes: list[str], C: float = 10.0, seed: int = 0):
        super().__init__(classes)
        self.C, self.seed = C, seed
        self.features = SparseTfidf()
        self.clf: LogisticRegression | None = None

    def fit(self, texts, labels):
        X = self.features.fit_transform(texts)
        self.clf = LogisticRegression(C=self.C, max_iter=5000, random_state=self.seed)
        self.clf.fit(X, self._y(labels))
        if len(self.clf.classes_) != len(self.classes):
            raise ValueError("every intent must appear in the training data")
        return self

    def logits(self, texts):
        return self.clf.decision_function(self.features.transform(texts))

    @property
    def vocabulary(self) -> set[str]:
        return self.features.vocabulary


class EmbeddingSoftmax(IntentModel):
    """Multinomial logistic regression on LSA embeddings, trained with Adam.

    Supports warm-started gradient updates, which is what incremental
    fine-tuning strategies (naive / replay) need.
    """
    name = "embedding_softmax"

    def __init__(self, classes: list[str], embedder: LSAEmbedder | None = None, l2: float = 1e-4,
                 epochs: int = 300, lr: float = 0.05, seed: int = 0):
        super().__init__(classes)
        self.embedder = embedder
        self.l2, self.epochs, self.lr, self.seed = l2, epochs, lr, seed
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def embed(self, texts):
        return self.embedder.transform(texts)

    def _train(self, Z: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> None:
        K = len(self.classes)
        Y = np.eye(K)[y]
        m = [np.zeros_like(self.W), np.zeros_like(self.b)]
        v = [np.zeros_like(self.W), np.zeros_like(self.b)]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, epochs + 1):
            P = softmax(Z @ self.W + self.b)
            G = (P - Y) / len(y)
            grads = [Z.T @ G + self.l2 * self.W, G.sum(axis=0)]
            for i, (param, g) in enumerate(((self.W, grads[0]), (self.b, grads[1]))):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                mh, vh = m[i] / (1 - b1 ** t), v[i] / (1 - b2 ** t)
                param -= lr * mh / (np.sqrt(vh) + eps)

    def fit(self, texts, labels):
        if self.embedder is None:
            self.embedder = LSAEmbedder(seed=self.seed).fit(texts)
        Z = self.embed(texts)
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, 0.01, (Z.shape[1], len(self.classes)))
        self.b = np.zeros(len(self.classes))
        self._train(Z, self._y(labels), self.epochs, self.lr)
        return self

    def logits(self, texts):
        return self.embed(texts) @ self.W + self.b

    def update(self, texts, labels, epochs: int = 30, lr: float = 0.01):
        self._train(self.embed(texts), self._y(labels), epochs, lr)


class NearestPrototype(IntentModel):
    """Cosine similarity to class means; prototypes update as running means."""
    name = "prototype"

    def __init__(self, classes: list[str], embedder: LSAEmbedder | None = None, scale: float = 20.0,
                 seed: int = 0):
        super().__init__(classes)
        self.embedder, self.scale, self.seed = embedder, scale, seed
        self.sums: np.ndarray | None = None
        self.counts: np.ndarray | None = None

    def embed(self, texts):
        return self.embedder.transform(texts)

    @property
    def prototypes(self) -> np.ndarray:
        P = self.sums / np.maximum(self.counts[:, None], 1)
        return P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)

    def fit(self, texts, labels):
        if self.embedder is None:
            self.embedder = LSAEmbedder(seed=self.seed).fit(texts)
        Z, y = self.embed(texts), self._y(labels)
        self.sums = np.zeros((len(self.classes), Z.shape[1]))
        self.counts = np.zeros(len(self.classes))
        np.add.at(self.sums, y, Z)
        np.add.at(self.counts, y, 1)
        return self

    def logits(self, texts):
        return self.scale * (self.embed(texts) @ self.prototypes.T)

    def update(self, texts, labels):
        Z, y = self.embed(texts), self._y(labels)
        np.add.at(self.sums, y, Z)
        np.add.at(self.counts, y, 1)


class BootstrapEnsemble(IntentModel):
    """M TF-IDF logistic regressions on bootstrap resamples; logits = mean member logits."""
    name = "ensemble"

    def __init__(self, classes: list[str], members: int = 5, C: float = 10.0, seed: int = 0):
        super().__init__(classes)
        self.members_n, self.C, self.seed = members, C, seed
        self.members: list[TfidfLogReg] = []

    def fit(self, texts, labels):
        rng = np.random.default_rng(self.seed)
        texts, labels = list(texts), list(labels)
        y = self._y(labels)
        self.members = []
        for m in range(self.members_n):
            idx = []
            for k in range(len(self.classes)):   # stratified bootstrap keeps every class present
                cls = np.flatnonzero(y == k)
                idx.extend(rng.choice(cls, size=len(cls), replace=True))
            member = TfidfLogReg(self.classes, C=self.C, seed=self.seed + m)
            member.fit([texts[i] for i in idx], [labels[i] for i in idx])
            self.members.append(member)
        return self

    def member_probs(self, texts) -> np.ndarray:
        return np.stack([m.predict_proba(texts) for m in self.members])   # (M, N, K)

    def logits(self, texts):
        return np.mean([m.logits(texts) for m in self.members], axis=0)
