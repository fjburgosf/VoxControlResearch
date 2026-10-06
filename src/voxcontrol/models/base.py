"""Common interface for intent models.

Every model returns logits over the *registry* intent order so uncertainty,
calibration and decision layers are model-agnostic.
"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class IntentModel:
    name = "base"

    def __init__(self, classes: list[str]):
        self.classes = list(classes)
        self.index = {c: i for i, c in enumerate(self.classes)}

    def _y(self, labels: list[str]) -> np.ndarray:
        return np.array([self.index[l] for l in labels], dtype=int)

    def fit(self, texts: list[str], labels: list[str]) -> "IntentModel":
        raise NotImplementedError

    def logits(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return softmax(self.logits(texts))

    def predict(self, texts: list[str]) -> list[str]:
        return [self.classes[i] for i in self.predict_proba(texts).argmax(axis=1)]

    def update(self, texts: list[str], labels: list[str]) -> None:
        raise NotImplementedError(f"{self.name} does not support incremental updates")

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not expose embeddings")
