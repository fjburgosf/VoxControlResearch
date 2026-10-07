"""Named, runnable examples (used by the GUI drop-down menu and by "voxcontrol example")."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np

_CACHE: dict = {}


def _quick_data(seed: int = 0):
    if ("data", seed) not in _CACHE:
        from .datasets.synthetic import generate
        _CACHE[("data", seed)] = generate(seed, variants_train=6, variants_cal=4, variants_test=4)
    return _CACHE[("data", seed)]


def quick_model(seed: int = 0, **flags):
    key = ("model", seed, tuple(sorted(flags.items())))
    if key not in _CACHE:
        from .ucil import UCIL, UCILConfig
        d = _quick_data(seed)
        _CACHE[key] = UCIL(config=UCILConfig(seed=seed, **flags)).fit(d.train, d.cal, d.ood_cal)
    return _CACHE[key]


_DEC = {"execute": {"es": "EJECUTAR", "en": "EXECUTE"}, "confirm": {"es": "CONFIRMAR", "en": "CONFIRM"},
        "reject": {"es": "RECHAZAR", "en": "REJECT"}}
_RISK = {"low": {"es": "bajo", "en": "low"}, "medium": {"es": "medio", "en": "medium"},
         "high": {"es": "alto", "en": "high"}}


def T(lang: str, es: str, en: str) -> str:
    return es if lang == "es" else en


def _row(r, lang: str) -> str:
    return (f"  {r.text!r:46s} -> {r.intent:15s} p_max={r.confidence:.2f}  "
            f"{T(lang, 'P(correcta)', 'P(correct)')}={r.calibrated_confidence:.2f}  OOD_z={r.ood_score:+.2f}  "
            f"{T(lang, 'riesgo', 'risk')}={_RISK[r.risk][lang]:6s} => {_DEC[r.decision][lang]}"
            + (f"  {T(lang, 'opciones', 'options')}={r.options}" if r.options else ""))


# ----------------------------------------------------------------- examples
def ex01(lang):
    m = quick_model()
    out = [T(lang, "Texto -> distribución de intenciones -> riesgo calibrado -> decisión",
             "Text -> intent posterior -> calibrated risk -> decision")]
    for t in ["abre spotify", "sube el volumen", "envía el mensaje a ana", "quiero escuchar música"]:
        out.append(_row(m.process_text(t, user=None), lang))
    return "\n".join(out)


def ex02(lang):
    from .api import VoxModel
    from .asr.recognizers import ASRUnavailable
    from .experiments.audio_experiment import synthesise
    from .paths import writable_root
    folder = writable_root() / "cache" / "example_audio"
    try:
        wav = synthesise(["open spotify"], folder)["open spotify"]
        vm = VoxModel(copy.deepcopy(quick_model()))
        r = vm.process_audio(wav, user=None)
    except ASRUnavailable as e:
        return T(lang, f"Modelo de reconocimiento no disponible: {e}", f"Speech recognition model not available: {e}")
    except Exception as e:  # noqa: BLE001 - TTS/ASR are optional at runtime
        return T(lang, f"El ejemplo de audio no pudo ejecutarse en este equipo: {type(e).__name__}: {e}",
                 f"Audio example could not run on this computer: {type(e).__name__}: {e}")
    head = T(lang, f"Archivo de voz sintética: {wav.name}\nTranscripción: ",
             f"Synthetic speech file: {wav.name}\nTranscript: ")
    conf = T(lang, "confianza ASR", "ASR confidence")
    return f"{head}{r.explanation['transcript']!r} ({conf} {r.explanation['asr_confidence']:.2f})\n" + _row(r, lang)


def ex03(lang):
    m = quick_model(use_context=True)
    out = [T(lang, "Mismas palabras, distinta aplicación activa (prior de contexto):",
             "Same words, different active application (context prior):")]
    for cat, app in (("music_player", "spotify"), ("document", "word")):
        r = m.process_text("bájale", context={"category": cat, "active_app": app}, user=None)
        out.append(f"  {T(lang, 'contexto', 'context')}={cat:13s}" + _row(r, lang))
    out.append(T(lang, "Destino ambiguo (varias aplicaciones coinciden):",
                 "Ambiguous target (several applications match):"))
    out.append(_row(m.process_text("cierra el chat", user=None), lang))
    return "\n".join(out)


def ex04(lang):
    m = quick_model()
    out = [T(lang, "Órdenes conocidas frente a peticiones fuera del registro (OOD):",
             "Known commands vs requests outside the registry (OOD):")]
    for t in ["abre el navegador chrome", "calcula la trayectoria óptima de un satélite", "cuéntame un chiste",
              "enciende las luces de la sala"]:
        r = m.process_text(t, user=None)
        out.append(_row(r, lang) + f"  {T(lang, 'P(en dominio)', 'P(in-domain)')}={r.p_in_domain:.2f}")
    return "\n".join(out)


def ex05(lang):
    from .calibration.calibrators import IsotonicCalibrator
    from .metrics.core import calibration_report
    m, d = quick_model(), _quick_data()
    ev = m.evidence([s.text for s in d.test])
    y = np.array([m.registry.index[s.intent] for s in d.test])
    ok = ev.probs_raw.argmax(1) == y
    evc = m.evidence([s.text for s in d.cal])
    yc = np.array([m.registry.index[s.intent] for s in d.cal])
    iso = IsotonicCalibrator().fit(evc.probs_t.max(1), evc.probs_t.argmax(1) == yc)
    temp = m.temperature.temperature
    out = [T(lang, f"Temperatura T = {temp:.3f}, calibrador elegido por UCIL: {m.calibrator_name}",
             f"Temperature T = {temp:.3f}, selected UCIL calibrator: {m.calibrator_name}"),
           "                 ECE     Brier   NLL"]
    names = {"uncal": T(lang, "sin calibrar", "uncalibrated"), "temp": T(lang, "temperatura", "temperature"),
             "iso": T(lang, "isotónica", "isotonic"), "ucil": "UCIL"}
    for key, conf in (("uncal", ev.probs_raw.max(1)), ("temp", ev.probs_t.max(1)),
                      ("iso", iso.transform(ev.probs_t.max(1))), ("ucil", ev.p_correct)):
        r = calibration_report(conf, ok)
        out.append(f"  {names[key]:14s} {r['ece']:.4f}  {r['brier']:.4f}  {r['nll']:.4f}")
    out.append(T(lang, "La calibración se mide en datos de prueba separados. No se asume que mejore.",
                 "Calibration is measured on held-out test data. It is not assumed to improve."))
    return "\n".join(out)


def ex06(lang):
    from .decision.policies import CostModel, risk_aware_decisions
    c = CostModel()
    out = [T(lang, "P(correcta) mínima para ejecutar directamente (P(en dominio)=1):",
             "Minimum P(correct) for direct execution (P(in-domain)=1):")]
    for r in ("low", "medium", "high"):
        out.append(f"  {T(lang, 'riesgo', 'risk')}={_RISK[r][lang]:6s}  C_err={c.c_err(r):5.1f}  "
                   f"{T(lang, 'umbral', 'threshold')}={c.execution_threshold(r):.3f}")
    out.append(T(lang, "Misma confianza, distinto riesgo de la acción:", "Same confidence, different action risk:"))
    for p in (0.80, 0.95, 0.99):
        dec, _ = risk_aware_decisions(np.array([p] * 3), np.ones(3), ["low", "medium", "high"], c)
        cells = "  ".join(f"{_RISK[r][lang]}={_DEC[d][lang]:9s}" for r, d in zip(("low", "medium", "high"), dec))
        out.append(f"  {T(lang, 'P(correcta)', 'P(correct)')}={p:.2f}  {cells}")
    return "\n".join(out)


def ex07(lang):
    m = copy.deepcopy(quick_model())
    t = "cállalo"
    before = m.process_text(t, user="ana")
    m.correct(t, "mute", user="ana", wrong_intent=before.intent)
    after = m.process_text("oye cállalo por favor", user="ana")
    other = m.process_text("oye cállalo por favor", user="luis")
    return "\n".join([T(lang, "Antes de la corrección:", "Before correction:"), _row(before, lang),
                      T(lang, "Después de una corrección del usuario 'ana':", "After one correction by user 'ana':"),
                      _row(after, lang),
                      T(lang, "Otro usuario no se ve afectado (M_user está separado de M_global):",
                        "Another user is unaffected (M_user is separate from M_global):"), _row(other, lang)])


def ex08(lang):
    from .datasets.asr_noise import perturb
    m = quick_model()
    out = [T(lang, "Errores de reconocimiento simulados (tasa por palabra) sobre la misma orden:",
             "Simulated ASR errors (per-word rate) on the same command:")]
    for i, rate in enumerate((0.0, 0.2, 0.4, 0.6)):
        t = perturb("baja el volumen por favor", rate, np.random.default_rng(i + 3))
        out.append(f"  {T(lang, 'tasa', 'rate')}={rate:.1f}" + _row(m.process_text(t, user=None), lang))
    out.append(T(lang, "La degradación real de audio con SNR controlada es el experimento exp_audio_noise.yaml.",
                 "Real audio degradation at controlled SNR is the experiment exp_audio_noise.yaml."))
    return "\n".join(out)


def ex09(lang):
    m, d = copy.deepcopy(quick_model()), _quick_data()
    out = []
    for user in sorted(d.user_adapt):
        ev_x = [s.text for s in d.user_eval[user]]
        y = np.array([m.registry.index[s.intent] for s in d.user_eval[user]])
        before = np.mean(m.evidence(ev_x, users=[user] * len(ev_x)).pred == y)
        for s in d.user_adapt[user]:
            m.correct(s.text, s.intent, user=user)
        after = np.mean(m.evidence(ev_x, users=[user] * len(ev_x)).pred == y)
        n = len(d.user_adapt[user])
        out.append(T(lang, f"  usuario {user}: exactitud en su forma de hablar {before:.2f} -> {after:.2f} "
                           f"tras {n} correcciones",
                     f"  user {user}: accuracy on personal phrasing {before:.2f} -> {after:.2f} after {n} corrections"))
    return T(lang, "Personalización con memorias de usuario separadas:\n",
             "Personalisation with separate user memories:\n") + "\n".join(out)


def ex10(lang):
    from .experiments.text_experiments import exp_main
    res = exp_main(0, {"dataset": {"variants_train": 6, "variants_cal": 4, "variants_test": 4}})
    out = [T(lang, "Método                      IER     IER-alto  Confirma Rechaza Costo",
             "Method                      IER     hiIER     Clarif   Reject  Cost")]
    for r in res["tables"]["decision_metrics"]:
        out.append(f"  {r['method']:24s} {r['incorrect_execution_rate']:.3f}   "
                   f"{r['high_risk_incorrect_execution_rate']:.3f}     {r['clarification_rate']:.3f}    "
                   f"{r['rejection_rate']:.3f}   {r['mean_cost']:.3f}")
    out.append(T(lang, "Una semilla y datos reducidos. Estudio completo: experimento exp_main.yaml.",
                 "One seed, reduced data. Full study: experiment exp_main.yaml."))
    return "\n".join(out)


@dataclass
class Example:
    key: str
    title: dict
    description: dict
    run: Callable[[str], str]


EXAMPLES = {e.key: e for e in [
    Example("example_01_text_intent", {"es": "01 · Intención desde texto", "en": "01 · Text intent"},
            {"es": "Clasifica órdenes escritas y muestra confianza, riesgo y decisión.",
             "en": "Classifies typed commands and shows confidence, risk and decision."}, ex01),
    Example("example_02_voice_command", {"es": "02 · Orden por voz (archivo)", "en": "02 · Voice command (file)"},
            {"es": "Sintetiza una orden, la transcribe con el ASR local y decide.",
             "en": "Synthesises a command, transcribes it with the local ASR and decides."}, ex02),
    Example("example_03_ambiguity", {"es": "03 · Ambigüedad y contexto", "en": "03 · Ambiguity and context"},
            {"es": "Una misma frase según la aplicación activa. Destino ambiguo con opciones.",
             "en": "Same phrase under different active applications. Ambiguous target with options."}, ex03),
    Example("example_04_ood", {"es": "04 · Órdenes fuera de dominio", "en": "04 · Out-of-domain requests"},
            {"es": "Distingue órdenes conocidas de peticiones no contempladas.",
             "en": "Separates known commands from unsupported requests."}, ex04),
    Example("example_05_calibration", {"es": "05 · Calibración", "en": "05 · Calibration"},
            {"es": "ECE, Brier y NLL sin calibrar, con temperatura, isotónica y UCIL.",
             "en": "ECE, Brier and NLL uncalibrated, temperature, isotonic and UCIL."}, ex05),
    Example("example_06_risk_routing", {"es": "06 · Decisión según riesgo", "en": "06 · Risk-aware routing"},
            {"es": "La misma confianza produce decisiones distintas según el costo del error.",
             "en": "The same confidence yields different decisions depending on error cost."}, ex06),
    Example("example_07_incremental_learning", {"es": "07 · Aprendizaje incremental",
                                                "en": "07 · Incremental learning"},
            {"es": "Una corrección del usuario cambia predicciones futuras sin reentrenar.",
             "en": "A user correction changes future predictions without retraining."}, ex07),
    Example("example_08_noisy_audio", {"es": "08 · Errores de reconocimiento", "en": "08 · Recognition errors"},
            {"es": "Efecto de errores simulados del ASR sobre intención y decisión.",
             "en": "Effect of simulated ASR errors on intent and decision."}, ex08),
    Example("example_09_user_personalization", {"es": "09 · Personalización por usuario",
                                                "en": "09 · User personalisation"},
            {"es": "Usuarios simulados A, B y C con expresiones propias.",
             "en": "Simulated users A, B and C with their own expressions."}, ex09),
    Example("example_10_ucil_benchmark", {"es": "10 · Comparación UCIL vs baselines",
                                          "en": "10 · UCIL vs baselines"},
            {"es": "Comparación reducida de B1–B4 y UCIL (una semilla).",
             "en": "Reduced comparison of B1–B4 and UCIL (one seed)."}, ex10),
]}


def run_example(key: str, lang: str = "es") -> str:
    if key not in EXAMPLES:
        raise KeyError(f"unknown example {key!r}; available: {', '.join(EXAMPLES)}")
    return EXAMPLES[key].run(lang)
