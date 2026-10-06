"""Confidence calibrators.

* ``TemperatureScaler`` rescales multiclass logits (one parameter, NLL fit).
* ``IsotonicCalibrator`` / ``PlattCalibrator`` map a scalar confidence to P(correct).
* ``FusionCalibrator`` (UCIL) learns P(correct execution) from several
  uncertainty signals at once with a regularised logistic model.

All are fitted on a calibration split that is disjoint from training and test.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..models.base import softmax

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        logits, y = np.asarray(logits, dtype=np.float64), np.asarray(y, dtype=int)

        def nll(log_t: float) -> float:
            p = softmax(logits, np.exp(log_t))
            return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)))

        res = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
        self.temperature = float(np.exp(res.x))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return softmax(logits, self.temperature)


class IdentityCalibrator:
    def fit(self, conf, correct):
        return self

    def transform(self, conf):
        return np.asarray(conf, dtype=np.float64)


class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, conf, correct):
        self.iso.fit(np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64))
        return self

    def transform(self, conf):
        return self.iso.predict(np.asarray(conf, dtype=np.float64))


class PlattCalibrator:
    def __init__(self):
        self.lr = LogisticRegression(C=1e4, max_iter=2000)

    def fit(self, conf, correct):
        correct = np.asarray(correct, dtype=int)
        if len(np.unique(correct)) < 2:
            raise ValueError("Platt scaling needs both correct and incorrect calibration samples")
        self.lr.fit(_logit(conf)[:, None], correct)
        return self

    def transform(self, conf):
        return self.lr.predict_proba(_logit(conf)[:, None])[:, 1]


class FusionCalibrator:
    """P(target | uncertainty features) with standardised logistic regression."""

    def __init__(self, C: float = 1.0):
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(C=C, max_iter=2000)
        self.feature_names: list[str] = []

    def fit(self, features: np.ndarray, target: np.ndarray, names: list[str] | None = None):
        target = np.asarray(target, dtype=int)
        if len(np.unique(target)) < 2:
            raise ValueError("fusion calibrator needs both outcomes in the calibration split")
        self.feature_names = names or [f"f{i}" for i in range(features.shape[1])]
        self.lr.fit(self.scaler.fit_transform(features), target)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(self.scaler.transform(features))[:, 1]

    def coefficients(self) -> dict[str, float]:
        return {n: float(c) for n, c in zip(self.feature_names, self.lr.coef_[0])}


SCALAR_CALIBRATORS = {
    "none": IdentityCalibrator,
    "isotonic": IsotonicCalibrator,
    "platt": PlattCalibrator,
}
