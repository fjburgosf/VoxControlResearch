"""Descriptive statistics, confidence intervals and paired comparisons.

Reports effect sizes and intervals; p-values are secondary.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def describe(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    out = {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
           "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
           "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
           "min": float(v.min()), "max": float(v.max())}
    if v.size > 1:
        half = sps.t.ppf(0.975, v.size - 1) * out["sd"] / np.sqrt(v.size)
        out["ci95_low"], out["ci95_high"] = out["mean"] - half, out["mean"] + half
    else:
        out["ci95_low"] = out["ci95_high"] = out["mean"]
    return out


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, stat=np.mean) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_bootstrap_diff(a, b, n_boot: int = 4000, seed: int = 0) -> dict:
    """Mean of (a - b) over paired observations with a percentile bootstrap CI."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = a - b
    lo, hi = bootstrap_ci(d, n_boot, seed)
    return {"mean_diff": float(d.mean()), "ci95_low": lo, "ci95_high": hi, "n": int(d.size)}


def cohens_dz(a, b) -> float:
    """Standardised mean of paired differences."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    return float(d.mean() / sd) if sd > 0 else float("nan")


def wilcoxon_paired(a, b) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if d.size < 2 or np.allclose(d, 0):
        return float("nan")
    return float(sps.wilcoxon(d, zero_method="zsplit").pvalue)


def mcnemar(a_correct, b_correct) -> dict:
    """Exact McNemar test on paired binary outcomes."""
    a, b = np.asarray(a_correct, dtype=bool), np.asarray(b_correct, dtype=bool)
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    n = n01 + n10
    p = float(sps.binomtest(n01, n, 0.5).pvalue) if n > 0 else 1.0
    return {"a_only": n10, "b_only": n01, "p_value": p}
