"""Out-of-distribution scores (higher = more likely OOD).

Output-based: maximum softmax probability, energy.
Feature-based (on embeddings of the training set): kNN distance, prototype
distance, class-conditional Mahalanobis with a shared shrinkage covariance.
Lexical: share of words never seen in training.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf

from ..text import tokens


def msp_score(probs: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(probs).max(axis=1)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Negative free energy -> we return  -T*logsumexp(logits/T)  (higher = more OOD)."""
    return -temperature * logsumexp(np.asarray(logits, dtype=np.float64) / temperature, axis=1)


class KNNDetector:
    def __init__(self, k: int = 5):
        self.k = k
        self.bank: np.ndarray | None = None

    def fit(self, Z: np.ndarray, y: np.ndarray | None = None):
        self.bank = np.asarray(Z)
        return self

    def score(self, Z: np.ndarray) -> np.ndarray:
        sims = np.asarray(Z) @ self.bank.T
        k = min(self.k, sims.shape[1])
        kth = -np.partition(-sims, k - 1, axis=1)[:, k - 1]
        return 1.0 - kth


class PrototypeDetector:
    def fit(self, Z: np.ndarray, y: np.ndarray):
        y = np.asarray(y)
        P = np.stack([Z[y == c].mean(axis=0) for c in np.unique(y)])
        self.protos = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        return self

    def score(self, Z: np.ndarray) -> np.ndarray:
        return 1.0 - (np.asarray(Z) @ self.protos.T).max(axis=1)


class MahalanobisDetector:
    def fit(self, Z: np.ndarray, y: np.ndarray):
        y = np.asarray(y)
        classes = np.unique(y)
        self.means = np.stack([Z[y == c].mean(axis=0) for c in classes])
        centred = np.concatenate([Z[y == c] - self.means[i] for i, c in enumerate(classes)])
        self.precision = LedoitWolf().fit(centred).precision_
        return self

    def score(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z)
        d = np.empty((len(Z), len(self.means)))
        for i, mu in enumerate(self.means):
            diff = Z - mu
            d[:, i] = np.einsum("ij,jk,ik->i", diff, self.precision, diff)
        return d.min(axis=1)


class LexicalOOVDetector:
    def fit(self, texts: list[str], y=None):
        self.vocab = {w for t in texts for w in tokens(t)}
        return self

    def score_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            toks = tokens(t)
            out.append(np.mean([w not in self.vocab for w in toks]) if toks else 1.0)
        return np.asarray(out, dtype=np.float64)


FEATURE_DETECTORS = {"knn": KNNDetector, "prototype": PrototypeDetector, "mahalanobis": MahalanobisDetector}
ALL_DETECTORS = ("msp", "energy", "knn", "prototype", "mahalanobis", "lexical")
