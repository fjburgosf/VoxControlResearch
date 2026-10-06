"""Text-mode experiments (no microphone, no OS automation).

Every function takes (seed, cfg) and returns {"tables": {name: rows}, "curves": {...}}.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from ..adaptation.memory import CorrectionMemory, MemoryEncoder, ReplayBuffer
from ..baselines import METHODS
from ..calibration.calibrators import IsotonicCalibrator, PlattCalibrator
from ..decision.policies import CONFIRM, EXECUTE, REJECT, decision_metrics, threshold_decisions
from ..embeddings.lsa import LSAEmbedder
from ..metrics.core import (calibration_report, classification_report, confusion, detection_report,
                            multiclass_nll, optimal_aurc, reliability_bins, risk_coverage)
from ..models.classifiers import (BootstrapEnsemble, EmbeddingSoftmax, NearestPrototype, RuleModel,
                                  TfidfLogReg)
from ..uncertainty.measures import (ensemble_disagreement, entropy, margin_uncertainty, mutual_information,
                                    one_minus_max)
from .common import COVERAGE_GRID, build, interp_curve, method_rows, noisy

SWEEP_TAUS = np.linspace(0.0, 1.0, 41)


# =============================================================== E1/E2/E4/E9
def exp_main(seed: int, cfg: dict) -> dict:
    """E1 clean recognition, E2 paraphrase difficulty, E4 OOD, E9 risk-sensitive execution."""
    b = build(seed, cfg)
    d, ucil, suite, costs = b.data, b.ucil, b.suite, b.ucil.config.costs
    reg = ucil.registry
    test = d.test + d.ood_test
    texts = [s.text for s in test]
    ev = ucil.evidence(texts)
    outs = suite.run(test, ev)
    y, ind = suite.truth(test)
    tables: dict[str, list] = {}
    curves: dict = {}
    tag = {"seed": seed}

    tables["decision_metrics"] = method_rows(outs, test, costs, **tag)

    # --- E1: classifiers -------------------------------------------------
    tr_x, tr_y = [s.text for s in d.train], [s.intent for s in d.train]
    id_test = d.test
    id_x, id_y = [s.text for s in id_test], [s.intent for s in id_test]
    levels = np.array([s.level for s in id_test])
    emb = LSAEmbedder(seed=seed).fit(tr_x)
    models = {"rules": RuleModel(reg), "tfidf_logreg": ucil.predictor,
              "embedding_softmax": EmbeddingSoftmax(reg.names, emb, seed=seed).fit(tr_x, tr_y),
              "prototype": NearestPrototype(reg.names, emb, seed=seed).fit(tr_x, tr_y)}
    ens = BootstrapEnsemble(reg.names, members=5, seed=seed).fit(tr_x, tr_y)
    models["ensemble"] = ens
    rows, lvl_rows = [], []
    for name, m in models.items():
        pred = np.array(m.predict(id_x))
        rows.append({**tag, "model": name, **classification_report(id_y, list(pred), reg.names)})
        for lv in range(4):
            mask = levels == lv
            lvl_rows.append({**tag, "model": name, "level": lv, "n": int(mask.sum()),
                             "accuracy": float(np.mean(pred[mask] == np.array(id_y)[mask]))})
    tables["classifier_metrics"] = rows
    tables["accuracy_by_level"] = lvl_rows
    curves["confusion"] = confusion(id_y, ucil.predictor.predict(id_x), reg.names)

    # --- E2: decisions by difficulty level -------------------------------
    rows = []
    id_mask = ind
    lv_all = np.array([s.level for s in test])
    for name, o in outs.items():
        for lv in range(4):
            mask = id_mask & (lv_all == lv)
            m = decision_metrics(o.decisions[mask], ind[mask], o.correct[mask], list(np.array(o.risks)[mask]), costs)
            rows.append({**tag, "method": name, "level": lv, "intent_accuracy": float(o.correct[mask].mean()),
                         **{k: m[k] for k in ("incorrect_execution_rate", "clarification_rate", "rejection_rate",
                                              "mean_cost")}})
    tables["decisions_by_level"] = rows

    # --- E1: uncertainty measures as error detectors ----------------------
    ev_id = ucil.evidence(id_x)
    y_id = np.array([reg.index[t] for t in id_y])
    err = ev_id.probs_raw.argmax(1) != y_id
    mp = ens.member_probs(id_x)
    ens_err = mp.mean(0).argmax(1) != y_id
    scores = {"entropy": entropy(ev_id.probs_raw), "one_minus_max_prob": one_minus_max(ev_id.probs_raw),
              "margin": margin_uncertainty(ev_id.probs_raw), "ucil_1_minus_p_correct": 1 - ev_id.p_correct}
    rows = [{**tag, "measure": k, "errors_of": "tfidf_logreg", **detection_report(err, v)} for k, v in scores.items()]
    rows += [{**tag, "measure": "ensemble_disagreement", "errors_of": "ensemble",
              **detection_report(ens_err, ensemble_disagreement(mp))},
             {**tag, "measure": "mutual_information", "errors_of": "ensemble",
              **detection_report(ens_err, mutual_information(mp))},
             {**tag, "measure": "ensemble_entropy", "errors_of": "ensemble",
              **detection_report(ens_err, entropy(mp.mean(0)))}]
    tables["error_detection"] = rows

    # --- E1: calibration ---------------------------------------------------
    cal = d.cal + d.ood_cal
    ev_cal = ucil.evidence([s.text for s in cal])
    yc, indc = suite.truth(cal)
    okc = ev_cal.probs_t.argmax(1) == yc
    conf_t_cal = ev_cal.probs_t.max(1)
    iso = IsotonicCalibrator().fit(conf_t_cal[indc], okc[indc])
    platt = PlattCalibrator().fit(conf_t_cal[indc], okc[indc])
    ok_id = ~err
    conf_id = {"none": ev_id.probs_raw.max(1), "temperature": ev_id.probs_t.max(1),
               "isotonic": iso.transform(ev_id.probs_t.max(1)), "platt": platt.transform(ev_id.probs_t.max(1)),
               "ucil": ev_id.p_correct}
    rows = [{**tag, "calibrator": k, "population": "in_domain", **calibration_report(v, ok_id)}
            for k, v in conf_id.items()]
    ok_all = np.array(outs["UCIL"].correct)
    ok_raw = ind & (ev.probs_raw.argmax(1) == y)
    conf_all = {"none": (ev.probs_raw.max(1), ok_raw), "temperature": (ev.probs_t.max(1), ok_raw),
                "isotonic": (iso.transform(ev.probs_t.max(1)), ok_raw), "ucil": (ev.p_correct, ok_all)}
    rows += [{**tag, "calibrator": k, "population": "in_domain_plus_ood", **calibration_report(v, c)}
             for k, (v, c) in conf_all.items()]
    rows.append({**tag, "calibrator": "none", "population": "in_domain", "multiclass_nll":
                 multiclass_nll(ev_id.probs_raw, y_id)})
    rows.append({**tag, "calibrator": "temperature", "population": "in_domain", "multiclass_nll":
                 multiclass_nll(ev_id.probs_t, y_id), "temperature": ucil.temperature.temperature})
    tables["calibration"] = rows
    tables["ucil_components"] = [{**tag, "selected_calibrator": ucil.calibrator_name,
                                  "selected_ood_detector": ucil.ood_name,
                                  **{f"cv_brier_{k}": v for k, v in ucil.calibrator_selection.items()},
                                  **{f"cal_auroc_{k}": v for k, v in ucil.ood_selection.items()}}]
    curves["reliability"] = {k: reliability_bins(v, ok_id) for k, v in conf_id.items() if k != "ucil"}
    curves["reliability"]["ucil"] = reliability_bins(ev.p_correct, ok_all)   # its target: correct execution, ID+OOD

    # --- risk-coverage (ID + OOD; executing OOD counts as an error) --------
    rc_rows, rc_curves = [], {}
    rc_inputs = {"B2_raw_confidence": (ev.probs_raw.max(1), ok_raw),
                 "B4_temperature_confidence": (ev.probs_t.max(1), ok_raw),
                 "UCIL_p_correct": (ev.p_correct, ok_all)}
    for k, (conf, ok) in rc_inputs.items():
        rc = risk_coverage(conf, ok)
        rc_rows.append({**tag, "confidence": k, "aurc": rc["aurc"], "oracle_aurc": optimal_aurc(ok),
                        "risk_at_50": float(np.interp(0.5, rc["coverage"], rc["risk"])),
                        "risk_at_80": float(np.interp(0.8, rc["coverage"], rc["risk"]))})
        rc_curves[k] = np.interp(COVERAGE_GRID, rc["coverage"], rc["risk"])
    tables["risk_coverage"] = rc_rows
    curves["risk_coverage"] = rc_curves

    # --- IER vs intervention-rate trade-off ---------------------------------
    trade = []
    for name, conf in (("B3_fixed_threshold", ev.probs_raw.max(1)), ("B4_calibrated_threshold", ev.probs_t.max(1))):
        for tau in SWEEP_TAUS:
            dec = threshold_decisions(conf, tau)
            m = decision_metrics(dec, ind, ok_raw, outs["B2_argmax"].risks, costs)
            trade.append({**tag, "method": name, "param": float(tau), **_trade_point(m)})
    for s, dec in suite.ucil_sweep(ev):
        m = decision_metrics(dec, ind, ok_all, outs["UCIL"].risks, costs)
        trade.append({**tag, "method": "UCIL", "param": s, **_trade_point(m)})
    tables["tradeoff_curve"] = trade

    # --- E4: OOD detection ---------------------------------------------------
    kinds = np.array([s.kind for s in test])
    scores = dict(ev.ood_scores)
    scores["fused"] = ucil._selected_ood(ev.ood_scores) if ucil.ood_fuser is not None else ev.ood
    scores["ucil_1_minus_p_in"] = 1 - ev.p_in
    rows = []
    for k, v in scores.items():
        for subset, mask in (("all", np.ones(len(test), bool)), ("near", (kinds == "ood_near") | ind),
                             ("far", (kinds == "ood_far") | ind)):
            rows.append({**tag, "detector": k, "subset": subset, **detection_report(~ind[mask], v[mask])})
    tables["ood_detection"] = rows
    rows = []
    for name, o in outs.items():
        for subset in ("ood_near", "ood_far"):
            mask = kinds == subset
            dd = o.decisions[mask]
            rows.append({**tag, "method": name, "subset": subset, "execute": float(np.mean(dd == EXECUTE)),
                         "confirm": float(np.mean(dd == CONFIRM)), "reject": float(np.mean(dd == REJECT))})
    tables["ood_decisions"] = rows

    # --- E9: risk-sensitive behaviour ---------------------------------------
    rows = []
    true_risk = np.array([reg.risk(s.intent) if s.in_domain else "ood" for s in test])
    for name, o in outs.items():
        for r in ("low", "medium", "high"):
            mask = true_risk == r
            m = decision_metrics(o.decisions[mask], ind[mask], o.correct[mask], list(np.array(o.risks)[mask]), costs)
            correct_conf = mask & o.correct
            rows.append({**tag, "method": name, "risk": r, "n": int(mask.sum()),
                         "incorrect_execution_rate": m["incorrect_execution_rate"],
                         "clarification_rate": m["clarification_rate"], "rejection_rate": m["rejection_rate"],
                         "confirm_rate_when_correct": float(np.mean(o.decisions[correct_conf] == CONFIRM))
                         if correct_conf.any() else float("nan")})
    tables["risk_sensitivity"] = rows
    tables["execution_thresholds"] = [{**tag, "risk": r, "c_err": costs.c_err(r),
                                       "min_p_correct_to_execute": costs.execution_threshold(r)}
                                      for r in ("low", "medium", "high")]
    rows = []
    for name in ("UCIL", "B4_calibrated_threshold"):
        o = outs[name]
        for r in ("low", "medium", "high"):
            pr = np.array(o.risks) == r
            for lo, hi in ((0.7, 0.9), (0.9, 0.98), (0.98, 1.01)):
                mask = pr & (ev.probs_t.max(1) >= lo) & (ev.probs_t.max(1) < hi)
                if mask.sum() >= 3:
                    rows.append({**tag, "method": name, "predicted_risk": r, "conf_bin": f"[{lo},{min(hi, 1)})",
                                 "n": int(mask.sum()), "execute_rate": float(np.mean(o.decisions[mask] == EXECUTE))})
    tables["confidence_matched_execution"] = rows
    return {"tables": tables, "curves": curves}


def _trade_point(m: dict) -> dict:
    return {"incorrect_execution_rate": m["incorrect_execution_rate"], "wrong_execution_cost": m["wrong_execution_cost"],
            "high_risk_incorrect_execution_rate": m["high_risk_incorrect_execution_rate"],
            "clarification_rate": m["clarification_rate"],
            "rejection_rate": m["rejection_rate"],
            "intervention_rate": m["clarification_rate"] + m["rejection_rate"], "mean_cost": m["mean_cost"]}


# ===================================================================== E3
def exp_ambiguity(seed: int, cfg: dict) -> dict:
    b = build(seed, cfg)
    ctx = build(seed, cfg, data=b.data, use_context=True)
    d, costs = b.data, b.ucil.config.costs
    amb = d.ambiguous_intent
    texts = [s.text for s in amb]
    contexts = [s.context for s in amb]
    ev = b.ucil.evidence(texts)
    ev_ctx = ctx.ucil.evidence(texts, contexts=contexts)
    outs = b.suite.run(amb, ev)
    outs["UCIL_context"] = ctx.suite.run(amb, ev, ucil_ev=ev_ctx, methods=("UCIL",))["UCIL"]
    tag = {"seed": seed}
    rows = method_rows(outs, amb, costs, **tag, set="intent_ambiguous")

    # clear L0 commands as reference for uncertainty
    clear = [s for s in d.test if s.level == 0]
    ev_clear = b.ucil.evidence([s.text for s in clear])
    unc_rows = [{**tag, "set": name, "mean_entropy": float(entropy(e.probs).mean()),
                 "mean_margin_uncertainty": float(margin_uncertainty(e.probs).mean()),
                 "mean_p_correct": float(e.p_correct.mean())}
                for name, e in (("clear_level0", ev_clear), ("intent_ambiguous", ev))]

    # slot ambiguity: UCIL validates slots before executing; B2-B4 do not
    slot_rows = []
    for s in d.ambiguous_slot:
        r = b.ucil.process_text(s.text, context=s.context, user=None)
        slot_rows.append({**tag, "text": s.text, "lang": s.lang, "predicted_intent": r.intent,
                          "intent_correct": r.intent == s.intent, "slot_status": r.slot_status,
                          "ucil_decision": r.decision, "n_options": len(r.options),
                          "options_cover_candidates": set(f"{r.intent}:{a}" for a in s.meta["app_candidates"])
                          <= set(r.options)})
    sb = b.suite.run(d.ambiguous_slot)
    summary = [{**tag, "method": k, "asks_when_target_ambiguous": float(np.mean(o.decisions != EXECUTE))}
               for k, o in sb.items() if k != "UCIL"]
    summary.append({**tag, "method": "UCIL", "asks_when_target_ambiguous":
                    float(np.mean([r["ucil_decision"] != EXECUTE for r in slot_rows]))})
    return {"tables": {"ambiguity_decisions": rows, "ambiguity_uncertainty": unc_rows,
                       "slot_ambiguity_items": slot_rows, "slot_ambiguity_summary": summary}, "curves": {}}


# ===================================================================== E6
def exp_asr_text(seed: int, cfg: dict) -> dict:
    from ..datasets.asr_noise import word_error_rate
    from ..text import normalize
    rates = cfg.get("noise_rates", [0.0, 0.05, 0.1, 0.2, 0.3])
    b = build(seed, cfg)
    d = b.data
    aug_cal = d.cal + [s for i, r in enumerate((0.05, 0.1, 0.2, 0.3)) for s in noisy(d.cal, r, seed * 97 + i)]
    from ..ucil import UCIL
    from .common import ucil_config
    noise_ucil = UCIL(config=ucil_config(cfg, seed)).fit(d.train, aug_cal, d.ood_cal)
    rows = []
    for i, rate in enumerate(rates):
        test = noisy(d.test + d.ood_test, rate, seed * 1000 + i)
        clean = d.test + d.ood_test
        wer = float(np.mean([word_error_rate(normalize(a.text), normalize(c.text)) for a, c in zip(test, clean)]))
        outs = b.suite.run(test)
        ev_n = noise_ucil.evidence([s.text for s in test])
        dec, _ = noise_ucil.decide(ev_n)
        from ..baselines import MethodOutput
        y, ind = b.suite.truth(test)
        outs["UCIL_noise_calibrated"] = MethodOutput(dec, ev_n.pred, ind & (ev_n.pred == y), ev_n.risks, ev_n.p_correct)
        for r in method_rows(outs, test, b.ucil.config.costs, seed=seed, noise_rate=rate, wer=wer):
            rows.append(r)
    return {"tables": {"asr_noise": rows}, "curves": {}}


# ===================================================================== E7
def _new_expression_split(d, seed):
    rng = np.random.default_rng(seed + 7)
    by_tpl: dict[str, list] = {}
    for s in d.test:
        if s.level >= 2:
            by_tpl.setdefault(s.template_id, []).append(s)
    stream, evaluation = [], []
    for tid in sorted(by_tpl):
        items = by_tpl[tid]
        stream.append(items[0])
        evaluation.extend(items[1:])
    rng.shuffle(stream)
    old = [s for s in d.test if s.level <= 1]
    return stream, evaluation, old


class _Adaptive:
    """Uniform wrapper so every adaptation strategy is driven the same way."""

    def __init__(self, kind: str, base, train, seed: int):
        self.kind = kind
        self.model = copy.deepcopy(base)
        self.memory = None
        self.buffer = None
        if kind == "memory":
            from ..intents.registry import IntentRegistry
            self.memory = CorrectionMemory(self.model.classes, MemoryEncoder.from_registry(IntentRegistry.default()))
        if kind.startswith("replay_"):
            self.buffer = ReplayBuffer(capacity=200, strategy=kind[len("replay_"):], seed=seed)
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(train), size=min(400, len(train)), replace=False)
            sub = [train[i] for i in idx]
            P = self.model.predict_proba([s.text for s in sub])
            Z = self.model.embed([s.text for s in sub])
            for s, p, z in zip(sub, entropy(P), Z):
                self.buffer.add(s.text, s.intent, float(p), z)

    def predict(self, texts, user="u"):
        if self.memory is not None and self.memory.size() > 0:
            P = self.model.predict_proba(texts)
            P, _ = self.memory.blend(P, self.memory.embed_fn(texts), [user] * len(texts))
            return np.array([self.model.classes[i] for i in P.argmax(1)])
        return np.array(self.model.predict(texts))

    def correct(self, s, user="u"):
        if self.kind == "none":
            return
        if self.memory is not None:
            self.memory.add(user, s.text, s.intent)
        elif self.kind == "prototype":
            self.model.update([s.text], [s.intent])
        elif self.kind == "finetune_naive":
            self.model.update([s.text], [s.intent])
        elif self.buffer is not None:
            unc = float(entropy(self.model.predict_proba([s.text]))[0])
            self.buffer.add(s.text, s.intent, max(unc, 1.0), self.model.embed([s.text])[0])
            bx, by = self.buffer.sample(31)
            self.model.update([s.text] + bx, [s.intent] + by)


STRATEGIES = ("none", "memory", "prototype", "finetune_naive", "replay_fifo", "replay_class_balanced",
              "replay_uncertainty", "replay_diversity")


def exp_incremental(seed: int, cfg: dict) -> dict:
    from ..datasets.synthetic import generate
    d = generate(seed, **cfg.get("dataset", {}))
    reg_names = sorted({s.intent for s in d.train})
    tr_x, tr_y = [s.text for s in d.train], [s.intent for s in d.train]
    emb = LSAEmbedder(seed=seed).fit(tr_x)
    softmax_base = EmbeddingSoftmax(reg_names, emb, seed=seed).fit(tr_x, tr_y)
    proto_base = NearestPrototype(reg_names, emb, seed=seed).fit(tr_x, tr_y)
    stream, evaluation, old = _new_expression_split(d, seed)
    checkpoints = sorted({c for c in cfg.get("checkpoints", [0, 5, 10, 20, 40, 80, 160]) if c <= len(stream)}
                         | {len(stream)})
    ev_x, ev_y = [s.text for s in evaluation], np.array([s.intent for s in evaluation])
    old_x, old_y = [s.text for s in old], np.array([s.intent for s in old])
    rows = []
    for kind in STRATEGIES:
        a = _Adaptive(kind, proto_base if kind == "prototype" else softmax_base, d.train, seed)
        n = 0
        for target in checkpoints:
            while n < target:
                a.correct(stream[n])
                n += 1
            rows.append({"seed": seed, "strategy": kind, "n_corrections": n,
                         "acc_new_expressions": float(np.mean(a.predict(ev_x) == ev_y)),
                         "acc_old_knowledge": float(np.mean(a.predict(old_x) == old_y))})
    base_old = {r["strategy"]: r["acc_old_knowledge"] for r in rows if r["n_corrections"] == 0}
    base_new = {r["strategy"]: r["acc_new_expressions"] for r in rows if r["n_corrections"] == 0}
    for r in rows:
        r["forgetting"] = base_old[r["strategy"]] - r["acc_old_knowledge"]
        r["adaptation_gain"] = r["acc_new_expressions"] - base_new[r["strategy"]]
    return {"tables": {"incremental": rows}, "curves": {}}


# ===================================================================== E8
def exp_personalization(seed: int, cfg: dict) -> dict:
    b = build(seed, cfg)
    d, ucil = b.data, b.ucil
    tr_x, tr_y = [s.text for s in d.train], [s.intent for s in d.train]
    soft = EmbeddingSoftmax(ucil.classes, ucil.embedder, seed=seed).fit(tr_x, tr_y)
    glob_x, glob_y = [s.text for s in d.test], np.array([s.intent for s in d.test])
    users = sorted(d.user_adapt)
    checkpoints = cfg.get("checkpoints", [0, 2, 4, 8, 12, 16, 24])
    rows = []
    for user in users:
        stream = list(d.user_adapt[user])
        np.random.default_rng(seed).shuffle(stream)
        others = [u for u in users if u != user]
        for strategy in ("none", "ucil_memory", "replay_class_balanced"):
            model = copy.deepcopy(ucil)
            model.memory = CorrectionMemory(model.classes, MemoryEncoder.from_registry(model.registry),
                                            threshold=model.config.memory_threshold)
            ada = _Adaptive("replay_class_balanced", soft, d.train, seed) if strategy.startswith("replay") else None
            n = 0
            for target in [c for c in checkpoints if c <= len(stream)]:
                while n < target:
                    s = stream[n]
                    if strategy == "ucil_memory":
                        model.correct(s.text, s.intent, user=user)
                    elif ada is not None:
                        ada.correct(s, user)
                    n += 1

                def acc(samples, u):
                    x, yy = [s.text for s in samples], np.array([s.intent for s in samples])
                    if ada is not None:
                        return float(np.mean(ada.predict(x, u) == yy))
                    ev = model.evidence(x, users=[u] * len(x))
                    return float(np.mean(np.array([model.classes[i] for i in ev.pred]) == yy))

                ev_u = model.evidence([s.text for s in d.user_eval[user]], users=[user] * len(d.user_eval[user]))
                dec, _ = model.decide(ev_u)
                yu = np.array([model.registry.index[s.intent] for s in d.user_eval[user]])
                m = decision_metrics(dec, np.ones(len(yu), bool), ev_u.pred == yu, ev_u.risks, model.config.costs)
                rows.append({"seed": seed, "user": user, "strategy": strategy, "n_corrections": n,
                             "user_accuracy": acc(d.user_eval[user], user),
                             "global_accuracy": acc(d.test, "default"),
                             "other_users_accuracy": float(np.mean([acc(d.user_eval[o], o) for o in others])),
                             "ucil_ier_on_user": m["incorrect_execution_rate"] if ada is None else float("nan"),
                             "ucil_clarification_on_user": m["clarification_rate"] if ada is None else float("nan")})
    return {"tables": {"personalization": rows}, "curves": {}}


# ================================================================ E10 + ablation
def _combined_set(d, seed, rate):
    hard = [s for s in d.test if s.level >= 2]
    users_eval = [s for u in sorted(d.user_eval) for s in d.user_eval[u]]
    return noisy(hard + d.ood_test + users_eval, rate, seed * 31 + 5)


def _ucil_with_memory(model, d):
    for u in sorted(d.user_adapt):
        for s in d.user_adapt[u]:
            model.correct(s.text, s.intent, user=u)
    return model


def exp_combined(seed: int, cfg: dict) -> dict:
    rate = float(cfg.get("combined_noise_rate", 0.15))
    b = build(seed, cfg)
    d = b.data
    test = _combined_set(d, seed, rate)
    users = [s.user or "default" for s in test]
    ev = b.ucil.evidence([s.text for s in test])
    outs = b.suite.run(test, ev)
    outs["UCIL_no_incremental"] = outs.pop("UCIL")
    _ucil_with_memory(b.ucil, d)
    ev_mem = b.ucil.evidence([s.text for s in test], users=users)
    outs["UCIL"] = b.suite.run(test, ev, ucil_ev=ev_mem, methods=("UCIL",))["UCIL"]
    rows = method_rows(outs, test, b.ucil.config.costs, seed=seed, noise_rate=rate)
    kinds = np.array([s.kind for s in test])
    sub = []
    for name, o in outs.items():
        for k in ("id", "ood_near", "ood_far", "user"):
            mask = kinds == k
            m = decision_metrics(o.decisions[mask], np.array([s.in_domain for s in test])[mask], o.correct[mask],
                                 list(np.array(o.risks)[mask]), b.ucil.config.costs)
            sub.append({"seed": seed, "method": name, "subset": k,
                        **{x: m[x] for x in ("incorrect_execution_rate", "clarification_rate", "rejection_rate",
                                             "mean_cost")}})
    return {"tables": {"combined_shift": rows, "combined_by_subset": sub}, "curves": {}}


ABLATIONS = {"UCIL_full": {}, "no_ood": {"use_ood": False}, "no_calibration": {"use_calibration": False},
             "no_risk_routing": {"use_risk": False}, "no_incremental": {"use_incremental": False}}


def exp_ablation(seed: int, cfg: dict) -> dict:
    from ..datasets.synthetic import generate
    d = generate(seed, **cfg.get("dataset", {}))
    rate = float(cfg.get("combined_noise_rate", 0.15))
    settings = {"clean": d.test + d.ood_test, "combined_shift": _combined_set(d, seed, rate)}
    rows = []
    for variant, flags in ABLATIONS.items():
        b = build(seed, cfg, data=d, **flags)
        _ucil_with_memory(b.ucil, d)
        for setting, test in settings.items():
            users = [s.user or "default" for s in test]
            ev = b.ucil.evidence([s.text for s in test], users=users)
            o = b.suite.run(test, ucil_ev=ev, methods=("UCIL",))["UCIL"]
            for r in method_rows({variant: o}, test, b.ucil.config.costs, seed=seed, setting=setting):
                rows.append(r)
    return {"tables": {"ablation": rows}, "curves": {}}


EXPERIMENTS = {
    "main": exp_main,                     # E1, E2, E4, E9
    "ambiguity": exp_ambiguity,           # E3
    "asr_text_noise": exp_asr_text,       # E6
    "incremental": exp_incremental,       # E7
    "personalization": exp_personalization,  # E8
    "combined_shift": exp_combined,       # E10
    "ablation": exp_ablation,
}
