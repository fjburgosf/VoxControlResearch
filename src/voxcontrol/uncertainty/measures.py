"""Intent uncertainty measures (higher = more uncertain), each in [0, 1]."""
from __future__ import annotations

import numpy as np


def entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy normalised by log K."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])


def one_minus_max(probs: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(probs).max(axis=1)


def margin_uncertainty(probs: np.ndarray) -> np.ndarray:
    """1 - (p_(1) - p_(2))."""
    s = np.sort(np.asarray(probs), axis=1)
    return 1.0 - (s[:, -1] - s[:, -2])


def ensemble_disagreement(member_probs: np.ndarray) -> np.ndarray:
    """Variance across members of the probability of the ensemble's top class.

    ``member_probs`` has shape (M, N, K). Scaled by 4 so the maximum is 1.
    """
    mp = np.asarray(member_probs)
    top = mp.mean(axis=0).argmax(axis=1)
    p_top = mp[:, np.arange(mp.shape[1]), top]
    return np.clip(4.0 * p_top.var(axis=0), 0.0, 1.0)


def mutual_information(member_probs: np.ndarray) -> np.ndarray:
    """Epistemic part of the ensemble entropy (BALD), normalised by log K."""
    mp = np.clip(np.asarray(member_probs, dtype=np.float64), 1e-12, 1.0)
    mean = mp.mean(axis=0)
    h_mean = -(mean * np.log(mean)).sum(axis=1)
    mean_h = -(mp * np.log(mp)).sum(axis=2).mean(axis=0)
    return np.clip((h_mean - mean_h) / np.log(mp.shape[2]), 0.0, 1.0)


def asr_uncertainty_from_words(word_probs: list[float]) -> float:
    """1 - geometric mean of word probabilities reported by the recogniser."""
    if not word_probs:
        return 1.0
    p = np.clip(np.asarray(word_probs, dtype=np.float64), 1e-6, 1.0)
    return float(1.0 - np.exp(np.log(p).mean()))


def asr_uncertainty_from_agreement(hypotheses: list[str]) -> float:
    """Disagreement among alternative transcripts: 1 - share of the modal hypothesis."""
    if not hypotheses:
        return 1.0
    from collections import Counter

    from ..text import normalize
    counts = Counter(normalize(h) for h in hypotheses)
    return 1.0 - counts.most_common(1)[0][1] / len(hypotheses)


MEASURES = {
    "entropy": entropy,
    "one_minus_max": one_minus_max,
    "margin": margin_uncertainty,
}
