"""Scientific baselines.

B1  Rules                 keyword matching (execute unique match / confirm tie / reject none)
B2  Intent classifier     always executes argmax_y p(y | x)
B3  Fixed threshold       raw max-prob >= tau (fixed a priori) -> execute, otherwise confirm
B4  Calibrated threshold  temperature-scaled confidence with execute/reject thresholds tuned on
                          the calibration split for minimum realised cost; no OOD, no risk
                          routing, no adaptation
UCIL                      proposed method

B2-B4 share UCIL's predictor, so differences come from the decision layer only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decision.policies import (EXECUTE, CostModel, risk_aware_decisions, rule_decisions, threshold_decisions,
                                tune_thresholds)
from .models.classifiers import RuleModel
from .ucil import UCIL, Evidence

METHODS = ("B1_rules", "B2_argmax", "B3_fixed_threshold", "B4_calibrated_threshold", "UCIL")


@dataclass
class MethodOutput:
    decisions: np.ndarray
    pred: np.ndarray          # predicted class indices
    correct: np.ndarray       # in-domain and pred == truth
    risks: list
    confidence: np.ndarray    # the score the method ranks by


class BaselineSuite:
    def __init__(self, ucil: UCIL, tau_fixed: float = 0.7):
        self.ucil = ucil
        self.rules = RuleModel(ucil.registry)
        self.tau_fixed = tau_fixed
        self.b4_thresholds: tuple[float, float | None] = (0.5, None)

    @property
    def costs(self) -> CostModel:
        return self.ucil.config.costs

    def truth(self, samples) -> tuple[np.ndarray, np.ndarray]:
        idx = self.ucil.registry.index
        y = np.array([idx[s.intent] if s.in_domain else -1 for s in samples])
        return y, y >= 0

    def fit(self, cal_samples) -> "BaselineSuite":
        ev = self.ucil.evidence([s.text for s in cal_samples])
        y, ind = self.truth(cal_samples)
        pred = ev.probs_t.argmax(axis=1)
        risks = [self.ucil.registry.risk(self.ucil.classes[i]) for i in pred]
        self.b4_thresholds = tune_thresholds(ev.probs_t.max(axis=1), ind, pred == y, risks, self.costs)
        return self

    def run(self, samples, ev: Evidence | None = None, ucil_ev: Evidence | None = None,
            methods=METHODS) -> dict[str, MethodOutput]:
        """``ev`` = evidence without user/context information (baselines);
        ``ucil_ev`` = evidence as UCIL sees it (may include context, memory, ASR)."""
        texts = [s.text for s in samples]
        ev = ev or self.ucil.evidence(texts)
        ucil_ev = ucil_ev or ev
        y, ind = self.truth(samples)
        reg, cls = self.ucil.registry, self.ucil.classes
        out: dict[str, MethodOutput] = {}

        def pack(dec, pred, conf):
            return MethodOutput(np.asarray(dec, dtype=object), pred, ind & (pred == y),
                                [reg.risk(cls[i]) for i in pred], np.asarray(conf, dtype=np.float64))

        if "B1_rules" in methods:
            S = self.rules.scores(texts)
            out["B1_rules"] = pack(rule_decisions(S), S.argmax(axis=1), S.max(axis=1))
        pred_raw = ev.probs_raw.argmax(axis=1)
        conf_raw = ev.probs_raw.max(axis=1)
        conf_t = ev.probs_t.max(axis=1)
        if "B2_argmax" in methods:
            out["B2_argmax"] = pack(np.full(len(texts), EXECUTE, dtype=object), pred_raw, conf_raw)
        if "B3_fixed_threshold" in methods:
            out["B3_fixed_threshold"] = pack(threshold_decisions(conf_raw, self.tau_fixed), pred_raw, conf_raw)
        if "B4_calibrated_threshold" in methods:
            te, tr = self.b4_thresholds
            out["B4_calibrated_threshold"] = pack(threshold_decisions(conf_t, te, tr), pred_raw, conf_t)
        if "UCIL" in methods:
            dec, _ = self.ucil.decide(ucil_ev)
            out["UCIL"] = pack(dec, ucil_ev.pred, ucil_ev.p_correct)
        return out

    def ucil_sweep(self, ev: Evidence, scales=None) -> list[tuple[float, np.ndarray]]:
        """UCIL decisions as the confirmation cost is scaled (traces the IER-intervention curve)."""
        scales = np.geomspace(0.02, 50, 40) if scales is None else scales
        base = self.costs
        out = []
        for s in scales:
            c = CostModel(error=dict(base.error), confirm=base.confirm * s, reject=base.reject * s)
            dec, _ = risk_aware_decisions(ev.p_correct, ev.p_in, ev.risks, c,
                                          uniform_risk=not self.ucil.config.use_risk)
            out.append((float(s), dec))
        return out
