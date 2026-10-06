"""Evaluation metrics: classification, calibration, selective prediction, OOD."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

_EPS = 1e-12


# --------------------------------------------------------------- classification
def classification_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(p),
        "macro_recall": float(r),
    }


def confusion(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


# ------------------------------------------------------------------ calibration
def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> dict:
    conf, correct = np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    acc = np.bincount(idx, weights=correct, minlength=n_bins) / np.maximum(count, 1)
    avg = np.bincount(idx, weights=conf, minlength=n_bins) / np.maximum(count, 1)
    return {"edges": edges, "count": count, "accuracy": acc, "confidence": avg}


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    b = reliability_bins(conf, correct, n_bins)
    w = b["count"] / max(b["count"].sum(), 1)
    return float(np.sum(w * np.abs(b["accuracy"] - b["confidence"])))


def mce(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    b = reliability_bins(conf, correct, n_bins)
    gaps = np.abs(b["accuracy"] - b["confidence"])[b["count"] > 0]
    return float(gaps.max()) if gaps.size else 0.0


def brier(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((np.asarray(conf) - np.asarray(correct, dtype=float)) ** 2))


def binary_nll(conf: np.ndarray, correct: np.ndarray) -> float:
    p = np.clip(np.asarray(conf, dtype=np.float64), _EPS, 1 - _EPS)
    c = np.asarray(correct, dtype=np.float64)
    return float(-np.mean(c * np.log(p) + (1 - c) * np.log(1 - p)))


def multiclass_nll(probs: np.ndarray, y: np.ndarray) -> float:
    probs = np.asarray(probs)
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), np.asarray(y)], _EPS, 1.0))))


def calibration_report(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> dict:
    return {"ece": ece(conf, correct, n_bins), "mce": mce(conf, correct, n_bins),
            "brier": brier(conf, correct), "nll": binary_nll(conf, correct)}


# -------------------------------------------------------- error detection / OOD
def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, tpr_level: float = 0.95) -> float:
    """FPR when the positive class (OOD/error) is detected with TPR >= tpr_level."""
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(fpr[np.searchsorted(tpr, tpr_level, side="left")])


def detection_report(is_positive: np.ndarray, score: np.ndarray) -> dict:
    """Positive = error (or OOD); score = uncertainty (higher = more suspicious)."""
    return {"auroc": _auroc(is_positive, score), "auprc": _auprc(is_positive, score),
            "fpr95": fpr_at_tpr(is_positive, score)}


# --------------------------------------------------------- selective prediction
def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> dict:
    """Execute the most confident first; risk = P(error | executed) at each coverage."""
    order = np.argsort(-np.asarray(confidence), kind="stable")
    err = 1.0 - np.asarray(correct, dtype=np.float64)[order]
    n = len(err)
    k = np.arange(1, n + 1)
    risk = np.cumsum(err) / k
    coverage = k / n
    return {"coverage": coverage, "risk": risk, "aurc": float(np.mean(risk))}


def risk_at_coverage(rc: dict, coverage: float) -> float:
    i = min(int(np.ceil(coverage * len(rc["coverage"]))) - 1, len(rc["coverage"]) - 1)
    return float(rc["risk"][max(i, 0)])


def optimal_aurc(correct: np.ndarray) -> float:
    """AURC of an oracle that ranks all correct samples first (lower bound)."""
    c = np.asarray(correct, dtype=np.float64)
    return risk_coverage(c, c)["aurc"]
