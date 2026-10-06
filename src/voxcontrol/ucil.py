"""UCIL — Uncertainty-Calibrated Incremental Intent Learning.

    text -> intent posterior -> uncertainty signals -> calibrated risk -> decision

Components (each can be switched off for ablation):

1. Intent predictor        p(y | x)                     (any IntentModel)
2. Uncertainty estimator   entropy, margin, max-prob, U_ASR, U_context
3. OOD detector            U_OOD, selected on the calibration split
4. Calibrator              temperature scaling, then a fusion model that maps
                           all signals to P(correct execution | x) and P(in-domain | x)
5. Risk router             argmin_d E[C(d) | x] with action-dependent error cost
6. Incremental learner     per-user correction memory (M_user) over a frozen M_global
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__
from .adaptation.memory import CorrectionMemory, MemoryEncoder
from .calibration.calibrators import FusionCalibrator, IsotonicCalibrator, TemperatureScaler
from .context import ContextPrior
from .decision.policies import CONFIRM, EXECUTE, REJECT, CostModel, risk_aware_decisions
from .embeddings.lsa import LSAEmbedder
from .intents.registry import IntentRegistry
from .metrics.core import detection_report
from .models.base import softmax
from .models.classifiers import BootstrapEnsemble, EmbeddingSoftmax, NearestPrototype, TfidfLogReg
from .ood.detectors import (FEATURE_DETECTORS, LexicalOOVDetector, energy_score, msp_score)
from .slots.extractor import SlotExtractor
from .uncertainty.measures import entropy, margin_uncertainty

_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


@dataclass
class UCILConfig:
    predictor: str = "tfidf_logreg"
    use_ood: bool = True
    use_calibration: bool = True
    use_risk: bool = True
    use_incremental: bool = True
    use_asr_uncertainty: bool = True
    use_context: bool = False
    ood_detector: str = "fused"   # fused | auto | msp | energy | knn | prototype | mahalanobis | lexical
    ood_candidates: tuple[str, ...] = ("msp", "energy", "knn", "prototype", "mahalanobis", "lexical")
    calibrator: str = "auto"      # auto | fusion | product_temperature | product_isotonic
    memory_threshold: float = 0.70
    context_alpha: float = 1.0
    costs: CostModel = field(default_factory=CostModel)
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "UCILConfig":
        d = dict(d or {})
        costs = CostModel.from_dict(d.pop("costs", None))
        if "ood_candidates" in d:
            d["ood_candidates"] = tuple(d["ood_candidates"])
        return cls(costs=costs, **d)


@dataclass
class Evidence:
    """Everything the decision layer and the evaluation need, for a batch of utterances."""
    texts: list
    logits: np.ndarray
    probs_raw: np.ndarray
    probs_t: np.ndarray
    probs: np.ndarray            # after context prior and correction memory
    pred: np.ndarray             # indices into classes, from `probs`
    pred_raw: np.ndarray         # argmax of the uncalibrated predictor
    ood_scores: dict
    ood: np.ndarray              # selected detector, z-scored on in-domain calibration data
    asr_unc: np.ndarray
    ctx_shift: np.ndarray
    mem_weight: np.ndarray
    p_correct: np.ndarray
    p_in: np.ndarray
    risks: list
    features: np.ndarray
    feature_names: list


@dataclass
class Result:
    text: str
    intent: str
    slots: dict
    slot_status: str
    confidence: float            # raw max probability of the predictor
    calibrated_confidence: float # P(correct execution | x)
    p_in_domain: float
    ood_score: float
    uncertainty: dict
    risk: str
    decision: str
    expected_costs: dict
    options: list
    explanation: dict

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class UCIL:
    def __init__(self, registry: IntentRegistry | None = None, config: UCILConfig | None = None):
        self.registry = registry or IntentRegistry.default()
        self.config = config or UCILConfig()
        self.classes = self.registry.names
        self.version = __version__
        self.fitted = False

    # ------------------------------------------------------------------ build
    def _make_predictor(self):
        c, seed = self.classes, self.config.seed
        kind = self.config.predictor
        if kind == "tfidf_logreg":
            return TfidfLogReg(c, seed=seed)
        if kind == "embedding_softmax":
            return EmbeddingSoftmax(c, self.embedder, seed=seed)
        if kind == "prototype":
            return NearestPrototype(c, self.embedder, seed=seed)
        if kind == "ensemble":
            return BootstrapEnsemble(c, seed=seed)
        raise ValueError(f"unknown predictor {kind!r}")

    def fit(self, train, cal_id, cal_ood=()) -> "UCIL":
        cfg = self.config
        texts = [s.text for s in train]
        labels = [s.intent for s in train]
        self.embedder = LSAEmbedder(seed=cfg.seed).fit(texts)
        self.predictor = self._make_predictor().fit(texts, labels)
        self._train_Z = self.embedder.transform(texts)
        self._train_y = np.array([self.registry.index[l] for l in labels])
        self.feature_detectors = {k: cls().fit(self._train_Z, self._train_y) for k, cls in FEATURE_DETECTORS.items()}
        self.lexical = LexicalOOVDetector().fit(texts)
        self.context_prior = ContextPrior(self.classes, alpha=cfg.context_alpha)
        self.memory = CorrectionMemory(self.classes, MemoryEncoder.from_registry(self.registry),
                                       threshold=cfg.memory_threshold)
        self.slots = SlotExtractor(self.registry, allowed_templates={s.template_id for s in train})

        cal = list(cal_id) + list(cal_ood)
        cal_texts = [s.text for s in cal]
        in_dom = np.array([s.in_domain for s in cal])
        cal_logits = self.predictor.logits(cal_texts)
        y_id = np.array([self.registry.index[s.intent] for s in cal_id])
        self.temperature = TemperatureScaler().fit(cal_logits[in_dom], y_id)

        # OOD: individual detectors and a learned fusion of all of them. The fusion's
        # calibration-split scores are out-of-fold so downstream calibrators are not
        # fitted on in-sample OOD scores. Selection uses calibration AUROC.
        raw = self._raw_ood_scores(cal_texts, cal_logits)
        has_ood = bool((~in_dom).any())
        self.ood_fuser = None
        oof = None
        if has_ood:
            X = np.column_stack([raw[k] for k in cfg.ood_candidates])
            self.ood_fuser = FusionCalibrator().fit(X, ~in_dom, list(cfg.ood_candidates))
            oof = self._out_of_fold_fused(X, ~in_dom)
            raw["fused"] = oof
        self.ood_selection = {k: detection_report(~in_dom, v)["auroc"] for k, v in raw.items()} if has_ood else {}
        if cfg.ood_detector == "auto":
            self.ood_name = max(self.ood_selection, key=self.ood_selection.get) if has_ood else "msp"
        elif cfg.ood_detector == "fused" and not has_ood:
            self.ood_name = "msp"
        else:
            self.ood_name = cfg.ood_detector
        cal_ood_raw = raw[self.ood_name]
        ref = cal_ood_raw[in_dom]
        self.ood_mu, self.ood_sd = float(ref.mean()), float(ref.std() + 1e-9)
        self.ood_threshold_z = float(np.percentile((ref - self.ood_mu) / self.ood_sd, 95))

        self.fitted = True
        ev = self.evidence(cal_texts, contexts=[s.context for s in cal],
                           asr_unc=[s.meta.get("asr_unc", 0.0) for s in cal], _fit_stage=True,
                           _ood_raw=cal_ood_raw)
        y_true = np.array([self.registry.index[s.intent] if s.in_domain else -1 for s in cal])
        correct_exec = in_dom & (ev.pred == y_true)
        use_in = cfg.use_ood and has_ood
        Fin = np.column_stack([ev.ood, _logit(ev.probs.max(axis=1))])
        self.calibrator_selection = self._cv_select_calibrator(ev.features, Fin, ev.probs.max(axis=1),
                                                               in_dom, correct_exec, use_in)
        if cfg.calibrator == "auto":
            self.calibrator_name = min(self.calibrator_selection, key=self.calibrator_selection.get)
        else:
            self.calibrator_name = cfg.calibrator
        self.fusion = FusionCalibrator().fit(ev.features, correct_exec, ev.feature_names)
        self.isotonic = IsotonicCalibrator().fit(ev.probs.max(axis=1)[in_dom], correct_exec[in_dom])
        self.in_domain_cal = FusionCalibrator().fit(Fin, in_dom, ["ood_z", "logit_max_prob"]) if use_in else None
        return self

    def _cv_select_calibrator(self, F, Fin, maxp, in_dom, correct_exec, use_in, folds: int = 5) -> dict:
        """Cross-validated Brier score of each way to estimate P(correct execution | x)."""
        from sklearn.model_selection import StratifiedKFold
        strata = in_dom.astype(int) * 2 + correct_exec.astype(int)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.config.seed)
        preds = {k: np.zeros(len(F)) for k in ("fusion", "product_temperature", "product_isotonic")}
        for tr, te in skf.split(F, strata):
            preds["fusion"][te] = FusionCalibrator().fit(F[tr], correct_exec[tr]).transform(F[te])
            p_in = (FusionCalibrator().fit(Fin[tr], in_dom[tr]).transform(Fin[te]) if use_in
                    else np.ones(len(te)))
            preds["product_temperature"][te] = maxp[te] * p_in
            tr_id = tr[in_dom[tr]]
            iso = IsotonicCalibrator().fit(maxp[tr_id], correct_exec[tr_id])
            preds["product_isotonic"][te] = iso.transform(maxp[te]) * p_in
        return {k: float(np.mean((v - correct_exec) ** 2)) for k, v in preds.items()}

    # --------------------------------------------------------------- evidence
    def _out_of_fold_fused(self, X: np.ndarray, is_ood: np.ndarray, folds: int = 5) -> np.ndarray:
        from sklearn.model_selection import StratifiedKFold
        out = np.zeros(len(X))
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.config.seed)
        for tr, te in skf.split(X, is_ood):
            f = FusionCalibrator().fit(X[tr], is_ood[tr])
            out[te] = _logit(f.transform(X[te]))
        return out

    def _selected_ood(self, raw: dict) -> np.ndarray:
        if self.ood_name == "fused":
            X = np.column_stack([raw[k] for k in self.config.ood_candidates])
            return _logit(self.ood_fuser.transform(X))
        return raw[self.ood_name]

    def _raw_ood_scores(self, texts, logits, Z=None) -> dict:
        Z = self.embedder.transform(texts) if Z is None else Z
        out = {}
        for k in self.config.ood_candidates:
            if k == "msp":
                out[k] = msp_score(softmax(logits))
            elif k == "energy":
                out[k] = energy_score(logits)
            elif k == "lexical":
                out[k] = self.lexical.score_texts(texts)
            else:
                out[k] = self.feature_detectors[k].score(Z)
        return out

    def evidence(self, texts, contexts=None, users=None, asr_unc=None, _fit_stage: bool = False,
                 _ood_raw: np.ndarray | None = None) -> Evidence:
        if not self.fitted:
            raise RuntimeError("UCIL.fit must be called first")
        cfg = self.config
        texts = list(texts)
        n = len(texts)
        logits = self.predictor.logits(texts)
        probs_raw = softmax(logits)
        probs_t = self.temperature.transform(logits) if cfg.use_calibration else probs_raw
        Z = self.embedder.transform(texts)
        raw_ood = self._raw_ood_scores(texts, logits, Z)
        selected = self._selected_ood(raw_ood) if _ood_raw is None else _ood_raw
        ood = (selected - self.ood_mu) / self.ood_sd

        probs = probs_t
        ctx_shift = np.zeros(n)
        if cfg.use_context and contexts is not None:
            cats = [(c or {}).get("category") for c in contexts]
            probs, ctx_shift = self.context_prior.apply(probs, cats)
        mem_weight = np.zeros(n)
        if cfg.use_incremental and users is not None and self.memory.size() > 0:
            probs, mem_weight = self.memory.blend(probs, self.memory.embed_fn(texts), list(users))
            ood = np.where(mem_weight > 0, np.minimum(ood, 0.0), ood)
        asr = np.zeros(n) if asr_unc is None else np.asarray(asr_unc, dtype=np.float64)

        feats, names = [_logit(probs.max(axis=1)), 1 - margin_uncertainty(probs), entropy(probs)], \
            ["logit_max_prob", "margin", "entropy"]
        if cfg.use_ood:
            feats.append(ood); names.append("ood_z")
        if cfg.use_asr_uncertainty:
            feats.append(asr); names.append("asr_uncertainty")
        if cfg.use_context:
            feats.append(ctx_shift); names.append("context_shift")
        F = np.column_stack(feats)

        pred = probs.argmax(axis=1)
        if _fit_stage:
            p_correct = probs.max(axis=1)
            p_in = np.ones(n)
        elif cfg.use_calibration:
            maxp = probs.max(axis=1)
            p_in = (self.in_domain_cal.transform(np.column_stack([ood, _logit(maxp)]))
                    if self.in_domain_cal is not None else np.ones(n))
            if self.calibrator_name == "fusion":
                p_correct = self.fusion.transform(F)
            elif self.calibrator_name == "product_isotonic":
                p_correct = self.isotonic.transform(maxp) * p_in
            else:
                p_correct = maxp * p_in
            # A correction-memory hit is user-provided evidence the calibrator never saw.
            p_correct = np.where(mem_weight > 0, np.maximum(p_correct, maxp * p_in), p_correct)
            p_in = np.maximum(p_in, p_correct)
        else:
            p_correct = probs.max(axis=1)
            p_in = np.where(ood > self.ood_threshold_z, 0.0, 1.0) if cfg.use_ood else np.ones(n)
        risks = [self.registry.risk(self.classes[i]) for i in pred]
        return Evidence(texts, logits, probs_raw, probs_t, probs, pred, probs_raw.argmax(axis=1), raw_ood, ood,
                        asr, ctx_shift, mem_weight, p_correct, p_in, risks, F, names)

    def decide(self, ev: Evidence):
        return risk_aware_decisions(ev.p_correct, ev.p_in, ev.risks, self.config.costs,
                                    uniform_risk=not self.config.use_risk)

    # ------------------------------------------------------------ single item
    def process_text(self, text: str, context: dict | None = None, user: str | None = "default",
                     asr_uncertainty: float = 0.0) -> Result:
        ev = self.evidence([text], contexts=[context or {}], users=[user], asr_unc=[asr_uncertainty])
        decisions, expected = self.decide(ev)
        decision, exp = str(decisions[0]), expected[0]
        intent = self.classes[int(ev.pred[0])]
        slot_res = self.slots.extract(text, intent, context)
        options = []
        if decision == EXECUTE and slot_res.status != "ok":
            decision = CONFIRM
        elif decision == REJECT and slot_res.status == "ambiguous":
            # concrete candidates for the intent's slot: ask which one instead of refusing
            decision = CONFIRM
        if decision == CONFIRM:
            order = np.argsort(-ev.probs[0])[:2]
            options = [self.classes[i] for i in order if ev.probs[0, i] >= 0.15 * ev.probs[0, order[0]]]
            if slot_res.candidates.get("app"):
                options = [f"{intent}:{a}" for a in slot_res.candidates["app"]]
        p_sorted = np.sort(ev.probs[0])[::-1]
        explanation = {
            "top_intent_prob": float(p_sorted[0]), "second_intent_prob": float(p_sorted[1]),
            "entropy": float(entropy(ev.probs)[0]), "ood_score_z": float(ev.ood[0]),
            "ood_detector": self.ood_name, "asr_uncertainty": float(ev.asr_unc[0]),
            "context_shift": float(ev.ctx_shift[0]), "memory_weight": float(ev.mem_weight[0]),
            "action_risk": ev.risks[0], "slot_status": slot_res.status,
            "expected_costs": {k: round(v, 4) for k, v in exp.items()},
        }
        return Result(
            text=text, intent=intent, slots=slot_res.values, slot_status=slot_res.status,
            confidence=float(ev.probs_raw[0].max()), calibrated_confidence=float(ev.p_correct[0]),
            p_in_domain=float(ev.p_in[0]), ood_score=float(ev.ood[0]),
            uncertainty={"entropy": float(entropy(ev.probs)[0]),
                         "margin": float(margin_uncertainty(ev.probs)[0]),
                         "ood_z": float(ev.ood[0]), "asr": float(ev.asr_unc[0]),
                         "context": float(ev.ctx_shift[0])},
            risk=ev.risks[0], decision=decision, expected_costs=exp, options=options, explanation=explanation,
        )

    def correct(self, text: str, correct_intent: str, user: str = "default", wrong_intent: str | None = None,
                uncertainty: float | None = None) -> None:
        if correct_intent not in self.registry:
            raise ValueError(f"unknown intent {correct_intent!r}")
        self.memory.add(user, text, correct_intent, wrong_intent, uncertainty, self.version)

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> "UCIL":
        """Load a model saved by :meth:`save`. Only load files you created (pickle)."""
        with open(path, "rb") as fh:
            return pickle.load(fh)


DECISION_LABELS = {EXECUTE: "EXECUTE", CONFIRM: "CONFIRM", REJECT: "REJECT"}
