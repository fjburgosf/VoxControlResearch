"""Decision layer: EXECUTE / CONFIRM / REJECT.

Cost model
----------
C_err(a)   cost of executing the wrong action, by the action's risk level;
C_confirm  cost of asking the user;
C_reject   cost of refusing a valid request (the user must repeat it).

Realised cost of one decision (used for evaluation):
    execute: 0 if the request is in-domain and the intent is correct, else C_err(a_pred)
    confirm: C_confirm, plus C_reject when an in-domain prediction was wrong
    reject : C_reject if the request was in-domain, else 0

UCIL chooses  argmin_d E[C(d) | x]  with p = P(correct execution | x) and
pi = P(in-domain | x):
    E[execute] = (1 - p) * C_err(a)
    E[confirm] = C_confirm + max(pi - p, 0) * C_reject
    E[reject]  = pi * C_reject
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EXECUTE, CONFIRM, REJECT = "execute", "confirm", "reject"
DECISIONS = (EXECUTE, CONFIRM, REJECT)


@dataclass
class CostModel:
    """Default error cost = C_reject (repeat the request) + cost of undoing the wrong action."""
    error: dict = field(default_factory=lambda: {"low": 2.0, "medium": 5.0, "high": 20.0})
    confirm: float = 0.3
    reject: float = 1.0

    def execution_threshold(self, risk: str, p_in: float = 1.0) -> float:
        """Smallest P(correct) at which EXECUTE is the minimum-expected-cost decision."""
        for p in np.linspace(0.0, 1.0, 10001):
            e = self.expected(float(p), max(p_in, float(p)), risk)
            if e[EXECUTE] <= min(e[CONFIRM], e[REJECT]):
                return float(p)
        return 1.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "CostModel":
        d = d or {}
        base = cls()
        return cls(error={**base.error, **d.get("error", {})}, confirm=float(d.get("confirm", base.confirm)),
                   reject=float(d.get("reject", base.reject)))

    def c_err(self, risk: str) -> float:
        return float(self.error[risk])

    def mean_error_cost(self) -> float:
        return float(np.mean(list(self.error.values())))

    def expected(self, p_correct: float, p_in: float, risk: str, uniform_risk: bool = False) -> dict:
        c_err = self.mean_error_cost() if uniform_risk else self.c_err(risk)
        return {EXECUTE: (1.0 - p_correct) * c_err,
                CONFIRM: self.confirm + max(p_in - p_correct, 0.0) * self.reject,
                REJECT: p_in * self.reject}

    def realised(self, decision: str, in_domain: bool, correct: bool, pred_risk: str) -> float:
        if decision == EXECUTE:
            return 0.0 if (in_domain and correct) else self.c_err(pred_risk)
        if decision == CONFIRM:
            return self.confirm + (self.reject if (in_domain and not correct) else 0.0)
        return self.reject if in_domain else 0.0


def threshold_decisions(conf: np.ndarray, tau_exec: float, tau_reject: float | None = None) -> np.ndarray:
    conf = np.asarray(conf)
    out = np.full(conf.shape, CONFIRM, dtype=object)
    out[conf >= tau_exec] = EXECUTE
    if tau_reject is not None:
        out[conf < tau_reject] = REJECT
    return out


def rule_decisions(rule_scores: np.ndarray) -> np.ndarray:
    """B1: execute a unique keyword match, confirm a tie, reject when nothing matched."""
    s = np.asarray(rule_scores)
    top = s.max(axis=1)
    n_top = (s == top[:, None]).sum(axis=1)
    out = np.full(len(s), EXECUTE, dtype=object)
    out[(top > 0) & (n_top > 1)] = CONFIRM
    out[top <= 0] = REJECT
    return out


def risk_aware_decisions(p_correct: np.ndarray, p_in: np.ndarray, risks: list[str], costs: CostModel,
                         uniform_risk: bool = False) -> tuple[np.ndarray, list[dict]]:
    out, expected = [], []
    for p, pi, r in zip(p_correct, p_in, risks):
        e = costs.expected(float(p), float(pi), r, uniform_risk)
        expected.append(e)
        out.append(min(DECISIONS, key=lambda d: (e[d], DECISIONS.index(d))))
    return np.asarray(out, dtype=object), expected


def tune_thresholds(conf: np.ndarray, in_domain: np.ndarray, correct: np.ndarray, risks: list[str],
                    costs: CostModel, allow_reject: bool = True) -> tuple[float, float | None]:
    """Grid-search (tau_exec, tau_reject) minimising mean realised cost on a calibration split."""
    conf = np.asarray(conf, dtype=np.float64)
    c_ex, c_cf, c_rj = realised_cost_table(in_domain, correct, risks, costs)
    grid = np.linspace(0.0, 1.0, 201)
    best = (np.inf, 1.0, None)
    for te in grid:
        ex = conf >= te
        rejects = [None] + (list(grid[grid <= te]) if allow_reject else [])
        for tr in rejects:
            rj = ~ex & (conf < tr) if tr is not None else np.zeros_like(ex)
            c = np.where(ex, c_ex, np.where(rj, c_rj, c_cf)).mean()
            if c < best[0] - 1e-12:
                best = (c, float(te), None if tr is None else float(tr))
    return best[1], best[2]


def realised_cost_table(in_domain, correct, risks, costs: CostModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample realised cost of execute / confirm / reject."""
    ind = np.asarray(in_domain, dtype=bool)
    ok = ind & np.asarray(correct, dtype=bool)
    c_err = np.array([costs.c_err(r) for r in risks])
    c_ex = np.where(ok, 0.0, c_err)
    c_cf = costs.confirm + np.where(ind & ~ok, costs.reject, 0.0)
    c_rj = np.where(ind, costs.reject, 0.0)
    return c_ex, c_cf, c_rj


def decision_metrics(decisions: np.ndarray, in_domain: np.ndarray, correct: np.ndarray, risks: list[str],
                     costs: CostModel) -> dict:
    d = np.asarray(decisions)
    ind = np.asarray(in_domain, dtype=bool)
    ok = ind & np.asarray(correct, dtype=bool)
    ex, cf, rj = d == EXECUTE, d == CONFIRM, d == REJECT
    risks = np.asarray(risks)
    n = max(len(d), 1)
    c_ex, c_cf, c_rj = realised_cost_table(ind, ok, risks, costs)
    realised = np.where(ex, c_ex, np.where(rj, c_rj, c_cf))
    out = {
        "n": int(len(d)),
        "incorrect_execution_rate": float(np.sum(ex & ~ok) / n),
        "correct_execution_rate": float(np.sum(ex & ok) / n),
        "clarification_rate": float(cf.mean()) if len(d) else 0.0,
        "rejection_rate": float(rj.mean()) if len(d) else 0.0,
        "coverage": float(ex.mean()) if len(d) else 0.0,
        "selective_risk": float(np.sum(ex & ~ok) / max(ex.sum(), 1)),
        "false_rejection_rate": float(np.sum(rj & ind) / max(ind.sum(), 1)),
        "ood_execution_rate": float(np.sum(ex & ~ind) / max((~ind).sum(), 1)),
        "ood_rejection_rate": float(np.sum(rj & ~ind) / max((~ind).sum(), 1)),
        "high_risk_incorrect_execution_rate": float(np.sum(ex & ~ok & (risks == "high")) / n),
        "wrong_execution_cost": float(np.sum(np.where(ex & ~ok, c_ex, 0.0)) / n),
        "mean_cost": float(realised.mean()) if len(d) else 0.0,
        "burden_confirmations": int(cf.sum()),
        "burden_repeats": int(np.sum(rj & ind) + np.sum(cf & ind & ~ok)),
        "burden_corrections": int(np.sum(ex & ~ok)),
    }
    out["user_burden_per_command"] = (out["burden_confirmations"] + out["burden_repeats"]
                                      + out["burden_corrections"]) / n
    return out
