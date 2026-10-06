"""Conceptual sanity tests (A-E) and end-to-end behaviour of UCIL."""
import copy

import numpy as np

from voxcontrol.decision import CONFIRM, EXECUTE, REJECT, CostModel, risk_aware_decisions
from voxcontrol.metrics import calibration_report
from voxcontrol.uncertainty import entropy


def test_A_clear_commands_less_uncertain_than_ambiguous(model, data):
    clear = [s.text for s in data.test if s.level == 0]
    amb = [s.text for s in data.ambiguous_intent]
    assert entropy(model.evidence(clear).probs).mean() < entropy(model.evidence(amb).probs).mean()


def test_B_ood_scores_higher_than_in_domain(model, data):
    ev_id = model.evidence([s.text for s in data.test])
    ev_ood = model.evidence([s.text for s in data.ood_test])
    assert ev_ood.ood.mean() > ev_id.ood.mean()
    assert ev_ood.p_in.mean() < ev_id.p_in.mean()


def test_C_high_risk_needs_at_least_as_much_confidence():
    costs = CostModel()
    p = np.linspace(0, 1, 201)
    for pi in (1.0, 0.9):
        low, _ = risk_aware_decisions(p, np.full_like(p, pi), ["low"] * len(p), costs)
        high, _ = risk_aware_decisions(p, np.full_like(p, pi), ["high"] * len(p), costs)
        # wherever high-risk executes, low-risk executes too
        assert np.all((high != EXECUTE) | (low == EXECUTE))
        assert (low == EXECUTE).sum() > (high == EXECUTE).sum()


def test_D_consistent_corrections_do_not_reduce_personal_accuracy(model, data):
    m = copy.deepcopy(model)
    for user in sorted(data.user_adapt):
        ev_x = [s.text for s in data.user_eval[user]]
        y = np.array([m.registry.index[s.intent] for s in data.user_eval[user]])
        before = np.mean(m.evidence(ev_x, users=[user] * len(ev_x)).pred == y)
        for s in data.user_adapt[user]:
            m.correct(s.text, s.intent, user=user)
        after = np.mean(m.evidence(ev_x, users=[user] * len(ev_x)).pred == y)
        assert after >= before


def test_D2_memory_does_not_change_global_or_other_users(model, data):
    m = copy.deepcopy(model)
    x = [s.text for s in data.test]
    before = m.evidence(x, users=["someone_else"] * len(x)).pred
    for s in data.user_adapt["A"]:
        m.correct(s.text, s.intent, user="A")
    after = m.evidence(x, users=["someone_else"] * len(x)).pred
    assert np.array_equal(before, after)


def test_E_calibration_is_measured_not_assumed(model, data):
    ev = model.evidence([s.text for s in data.test])
    y = np.array([model.registry.index[s.intent] for s in data.test])
    ok = ev.probs_raw.argmax(1) == y
    raw = calibration_report(ev.probs_raw.max(1), ok)
    temp = calibration_report(ev.probs_t.max(1), ok)
    for r in (raw, temp):
        assert 0.0 <= r["ece"] <= 1.0 and np.isfinite(r["brier"])
    assert model.temperature.temperature > 0
    assert set(model.calibrator_selection) == {"fusion", "product_temperature", "product_isotonic"}


def test_vertical_slice_text_to_sandbox(model):
    from voxcontrol.api import VoxModel
    vm = VoxModel(copy.deepcopy(model))
    r = vm.process_text("abre spotify")
    assert r.intent == "open_app" and r.slots.get("app") == "spotify"
    assert r.decision in (EXECUTE, CONFIRM, REJECT)
    res = vm.execute(r, confirmed=True)
    assert res is not None and res.success and "spotify" in vm.sandbox.state.open_apps


def test_slot_ambiguity_forces_confirmation_with_options(model, data):
    checked = 0
    for s in data.ambiguous_slot:
        r = model.process_text(s.text, context=s.context, user=None)
        if r.intent != s.intent:
            continue   # an intent error is handled by the uncertainty path, not by slot validation
        checked += 1
        assert r.slot_status == "ambiguous" and r.decision == CONFIRM
        assert {f"{r.intent}:{a}" for a in s.meta["app_candidates"]} <= set(r.options)
    assert checked >= len(data.ambiguous_slot) // 3


def test_real_mode_always_confirms_high_risk(model):
    from voxcontrol.api import VoxModel
    vm = VoxModel(copy.deepcopy(model), execution="real")
    r = vm.process_text("borra el archivo informe.docx")
    assert r.risk == "high" and r.decision != EXECUTE


def test_ablation_flags_change_features(data):
    from voxcontrol.ucil import UCIL, UCILConfig
    m = UCIL(config=UCILConfig(seed=3, use_ood=False, use_asr_uncertainty=False)).fit(
        data.train[:400] + data.train[-400:], data.cal, data.ood_cal)
    ev = m.evidence(["abre spotify"])
    assert "ood_z" not in ev.feature_names and ev.p_in[0] == 1.0


def test_save_and_load_roundtrip(model, tmp_path):
    from voxcontrol.ucil import UCIL
    path = tmp_path / "m.pkl"
    model.save(path)
    m2 = UCIL.load(path)
    a = model.evidence(["sube el volumen"]).p_correct
    assert np.allclose(a, m2.evidence(["sube el volumen"]).p_correct)
