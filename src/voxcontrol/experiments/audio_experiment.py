"""E5 — noisy speech: SNR -> ASR error -> intent error -> decision.

Speech is synthesised locally with the operating system's text-to-speech
(SAPI through pyttsx3) and cached; it is clean synthetic speech, not natural
speech, which is a stated limitation. Each utterance is degraded with white
noise at controlled SNRs and transcribed with faster-whisper; the word
probabilities give a real U_ASR signal.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..asr.recognizers import FasterWhisperRecognizer
from ..audio.processing import add_noise, load_audio, preprocess
from ..baselines import MethodOutput
from ..datasets.asr_noise import word_error_rate
from ..datasets.synthetic import generate
from ..text import normalize
from .common import build, method_rows

_RECOGNIZER: FasterWhisperRecognizer | None = None


def synthesise(texts: list[str], cache: Path, rate: int = 0) -> dict[str, Path]:
    """SAPI speaking rate: -10 (slow) .. 10 (fast); 0 = default."""
    cache.mkdir(parents=True, exist_ok=True)
    paths = {t: cache / (hashlib.sha1(t.encode("utf-8")).hexdigest()[:16] + ".wav") for t in texts}
    todo = [t for t, p in paths.items() if not p.exists()]
    if todo:
        import comtypes.client   # Windows SAPI, one file per utterance
        from ..paths import frozen
        if frozen():
            comtypes.client.gen_dir = None    # keep generated COM wrappers in memory (no cache on disk)
        voice = comtypes.client.CreateObject("SAPI.SpVoice")
        voice.Rate = rate
        for t in todo:
            stream = comtypes.client.CreateObject("SAPI.SpFileStream")
            stream.Open(str(paths[t].resolve()), 3)      # 3 = SSFMCreateForWrite
            voice.AudioOutputStream = stream
            voice.Speak(t)
            stream.Close()
    return paths


def _recogniser(cfg) -> FasterWhisperRecognizer:
    global _RECOGNIZER
    if _RECOGNIZER is None:
        a = cfg.get("asr", {})
        _RECOGNIZER = FasterWhisperRecognizer(model=a.get("model", "base"), device=a.get("device", "auto"),
                                              compute_type=a.get("compute_type", "int8"), language="en")
    return _RECOGNIZER


def exp_audio(seed: int, cfg: dict) -> dict:
    acfg = cfg.get("audio", {})
    snrs = acfg.get("snr_db", ["clean", 20, 10, 5, 0])
    rng = np.random.default_rng(seed)
    d = generate(seed, **{**cfg.get("dataset", {}), "languages": ("en",)})
    take = lambda xs, n: [xs[i] for i in sorted(rng.choice(len(xs), size=min(n, len(xs)), replace=False))]
    test = take(d.test, acfg.get("n_test", 120)) + d.ood_test
    cal = take(d.cal, acfg.get("n_cal", 120)) + d.ood_cal
    cache = Path(acfg.get("cache", "datasets/cache/tts_en"))
    wavs = synthesise(sorted({s.text for s in test + cal}), cache)
    rec = _recogniser(cfg)
    clean_audio = {t: preprocess(*load_audio(p)) for t, p in wavs.items()}

    def transcribe(samples, snr_for):
        out = []
        for i, s in enumerate(samples):
            snr = snr_for(i)
            x = clean_audio[s.text]
            if snr != "clean":
                x = add_noise(x, float(snr), np.random.default_rng([seed, i, snrs.index(snr)]))
            tr = rec.transcribe(x)
            out.append(replace(s, text=tr.text or "", meta={**s.meta, "asr_unc": tr.uncertainty, "snr": snr,
                                                             "reference": s.text}))
        return out

    # calibration utterances are transcribed at a random SNR so U_ASR varies during calibration
    cal_snrs = [snrs[int(k)] for k in rng.integers(len(snrs), size=len(cal))]
    cal_tr = transcribe(cal, lambda i: cal_snrs[i])
    cal_data = replace(d, cal=[s for s in cal_tr if s.in_domain], ood_cal=[s for s in cal_tr if not s.in_domain])
    b = build(seed, cfg, data=cal_data)
    variants = {"UCIL_no_asr_uncertainty": build(seed, cfg, data=cal_data, use_asr_uncertainty=False),
                "UCIL_fusion_with_asr": build(seed, cfg, data=cal_data, calibrator="fusion"),
                "UCIL_fusion_without_asr": build(seed, cfg, data=cal_data, calibrator="fusion",
                                                 use_asr_uncertainty=False)}
    rows, attribution = [], []
    for snr in snrs:
        tt = transcribe(test, lambda i: snr)
        texts = [s.text for s in tt]
        asr = [s.meta["asr_unc"] for s in tt]
        wer = float(np.mean([word_error_rate(normalize(s.meta["reference"]), normalize(s.text)) for s in tt]))
        ev = b.ucil.evidence(texts)
        ev_asr = b.ucil.evidence(texts, asr_unc=asr)
        outs = b.suite.run(tt, ev, ucil_ev=ev_asr)
        y, ind = b.suite.truth(tt)
        for vname, vb in variants.items():
            ev_v = vb.ucil.evidence(texts, asr_unc=asr)
            dec, _ = vb.ucil.decide(ev_v)
            outs[vname] = MethodOutput(dec, ev_v.pred, ind & (ev_v.pred == y), ev_v.risks, ev_v.p_correct)
        rows += method_rows(outs, tt, b.ucil.config.costs, seed=seed, snr_db=str(snr), wer=wer,
                            mean_asr_uncertainty=float(np.mean(asr)))
        # error attribution on in-domain items: intent error vs ASR-induced error
        ref_pred = b.ucil.evidence([s.meta["reference"] for s in tt]).probs_raw.argmax(1)
        hyp_pred = ev.probs_raw.argmax(1)
        ok_ref, ok_hyp = (ref_pred == y)[ind], (hyp_pred == y)[ind]
        attribution.append({"seed": seed, "snr_db": str(snr), "wer": wer,
                            "correct": float(np.mean(ok_ref & ok_hyp)),
                            "intent_error": float(np.mean(~ok_ref)),
                            "asr_induced_error": float(np.mean(ok_ref & ~ok_hyp)),
                            "asr_recovered": float(np.mean(~ok_ref & ok_hyp))})
    return {"tables": {"audio_noise": rows, "audio_error_attribution": attribution}, "curves": {}}
