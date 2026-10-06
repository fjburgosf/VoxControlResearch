import numpy as np
import pytest

from voxcontrol.adaptation import CorrectionMemory, ReplayBuffer
from voxcontrol.audio import add_noise, dropout, resample, save_wav, load_audio, trim_silence
from voxcontrol.calibration import IsotonicCalibrator, PlattCalibrator, TemperatureScaler
from voxcontrol.datasets import perturb, word_error_rate
from voxcontrol.decision import CONFIRM, EXECUTE, REJECT, CostModel, decision_metrics, rule_decisions
from voxcontrol.intents import IntentRegistry
from voxcontrol.metrics import brier, describe, detection_report, ece, risk_coverage
from voxcontrol.models import softmax
from voxcontrol.sandbox import Sandbox
from voxcontrol.slots import SlotExtractor
from voxcontrol.text import normalize
from voxcontrol.uncertainty import entropy, margin_uncertainty, one_minus_max

REG = IntentRegistry.default()


def test_normalize_strips_accents_and_punctuation():
    assert normalize("¡Ábrelo, YA!") == "abrelo ya"
    assert normalize("informe.docx") == "informe.docx"


def test_registry_has_risks_and_templates():
    assert len(REG) >= 15
    assert REG.risk("delete_file") == "high" and REG.risk("open_app") == "low"
    assert all(REG.spec(n).templates.get("es") for n in REG.names)


def test_split_is_disjoint_by_template(data):
    tr = {s.template_id for s in data.train}
    assert tr.isdisjoint({s.template_id for s in data.test})
    assert tr.isdisjoint({s.template_id for s in data.cal})
    assert all(s.level <= 1 for s in data.train)
    assert {s.meta["topic"] for s in data.ood_cal}.isdisjoint({s.meta["topic"] for s in data.ood_test})


def test_generator_is_reproducible():
    from voxcontrol.datasets import generate
    a, b = generate(11, variants_train=2), generate(11, variants_train=2)
    assert [s.text for s in a.test] == [s.text for s in b.test]


def test_uncertainty_measures_bounds():
    p = np.array([[1.0, 0, 0], [1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0.0]])
    assert entropy(p)[0] == pytest.approx(0, abs=1e-6) and entropy(p)[1] == pytest.approx(1)
    assert one_minus_max(p)[0] == 0
    assert margin_uncertainty(p)[2] == pytest.approx(1)


def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    logits = rng.normal(0, 1, (2000, 3))
    logits[np.arange(2000), y] += 1.0
    T = TemperatureScaler().fit(logits * 4, y).temperature
    assert 3.0 < T < 5.0


def test_isotonic_and_platt_reduce_ece_on_miscalibrated_scores():
    rng = np.random.default_rng(1)
    p_true = rng.uniform(0.3, 1.0, 4000)
    correct = rng.random(4000) < p_true
    conf = p_true ** 0.25            # overconfident
    for cal in (IsotonicCalibrator(), PlattCalibrator()):
        cal.fit(conf[:2000], correct[:2000])
        assert ece(cal.transform(conf[2000:]), correct[2000:]) < ece(conf[2000:], correct[2000:])


def test_metrics_reference_values():
    assert brier(np.array([1.0, 0.0]), np.array([1, 1])) == pytest.approx(0.5)
    rc = risk_coverage(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0]))
    assert rc["risk"][1] == 0 and rc["risk"][2] == pytest.approx(1 / 3)
    assert detection_report(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))["auroc"] == 1.0
    d = describe([1, 2, 3, 4])
    assert d["mean"] == 2.5 and d["ci95_low"] < 2.5 < d["ci95_high"]


def test_cost_model_execution_threshold_increases_with_risk():
    c = CostModel()
    t = [c.execution_threshold(r) for r in ("low", "medium", "high")]
    assert t[0] < t[1] < t[2]


def test_realised_costs_and_decision_metrics():
    c = CostModel()
    assert c.realised(EXECUTE, True, True, "high") == 0
    assert c.realised(EXECUTE, False, False, "high") == c.c_err("high")
    assert c.realised(REJECT, False, False, "low") == 0
    m = decision_metrics(np.array([EXECUTE, EXECUTE, CONFIRM, REJECT], dtype=object),
                         np.array([True, True, True, False]), np.array([True, False, True, False]),
                         ["low", "high", "low", "low"], c)
    assert m["incorrect_execution_rate"] == 0.25 and m["clarification_rate"] == 0.25
    assert m["high_risk_incorrect_execution_rate"] == 0.25 and m["ood_rejection_rate"] == 1.0


def test_rule_decisions():
    d = rule_decisions(np.array([[0, 2, 0], [1, 1, 0], [0, 0, 0]]))
    assert list(d) == [EXECUTE, CONFIRM, REJECT]


def test_slot_extraction_and_validation():
    s = SlotExtractor(REG)
    assert s.extract("pon el volumen al 40 por ciento", "set_volume").values["value"] == 40
    assert s.extract("pon el volumen al 140 por ciento", "set_volume").status == "invalid"
    r = s.extract("abre el editor", "open_app")
    assert r.status == "ambiguous" and set(r.candidates["app"]) == {"notepad", "vscode", "paint"}
    assert s.extract("abre visual studio code", "open_app").values["app"] == "vscode"
    assert s.extract("ciérralo", "close_app", {"active_app": "word"}).values["app"] == "word"


def test_sandbox_only_changes_simulated_state():
    sb = Sandbox(REG)
    assert sb.execute("open_app", {"app": "word"}).success
    assert sb.execute("type_text", {"text": "hola"}).success
    assert sb.state.documents["word"]["saved"] is False
    assert sb.execute("delete_file", {"file": "informe.docx"}).success and "informe.docx" in sb.state.trash
    assert not sb.execute("set_volume", {"value": 150}).success
    assert not sb.execute("format_disk", {}).success


def test_asr_text_perturbation_and_wer():
    rng = np.random.default_rng(0)
    assert perturb("abre el navegador", 0.0, rng) == "abre el navegador"
    noisy = [perturb("baja el volumen por favor ahora", 0.5, rng) for _ in range(20)]
    assert any(n != "baja el volumen por favor ahora" for n in noisy)
    assert word_error_rate("a b c", "a x c") == pytest.approx(1 / 3)


def test_replay_buffer_strategies_respect_capacity():
    for strat in ReplayBuffer.STRATEGIES:
        b = ReplayBuffer(capacity=5, strategy=strat, seed=0)
        for i in range(12):
            b.add(f"t{i}", f"c{i % 3}", uncertainty=i / 12, z=np.eye(12)[i])
        assert len(b) == 5


def test_correction_memory_is_per_user():
    emb = lambda texts: np.array([[1.0, 0.0] if "x" in t else [0.0, 1.0] for t in texts])
    mem = CorrectionMemory(["a", "b"], emb, threshold=0.5)
    mem.add("u1", "x", "b")
    P = np.array([[0.9, 0.1], [0.9, 0.1]])
    out, lam = mem.blend(P, emb(["x", "x"]), ["u1", "u2"])
    assert out[0].argmax() == 1 and lam[1] == 0 and np.allclose(out[1], P[1])


def test_audio_roundtrip_and_degradations(tmp_path):
    sr = 16000
    t = np.arange(sr) / sr
    x = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    save_wav(tmp_path / "a.wav", x, sr)
    y, sr2 = load_audio(tmp_path / "a.wav")
    assert sr2 == sr and np.allclose(y[0], x, atol=1e-3)
    noisy = add_noise(x, 10.0, np.random.default_rng(0))
    snr = 10 * np.log10(np.mean(x ** 2) / np.mean((noisy - x) ** 2))
    assert abs(snr - 10.0) < 0.5
    assert len(resample(x, 16000, 8000)) == 8000
    assert np.sum(dropout(x, 0.5, np.random.default_rng(0)) == 0) > 0.4 * len(x)
    padded = np.concatenate([np.zeros(sr), x, np.zeros(sr)])
    assert len(trim_silence(padded)) < len(padded)


def test_softmax_rows_sum_to_one():
    p = softmax(np.random.default_rng(0).normal(size=(5, 4)))
    assert np.allclose(p.sum(1), 1)
