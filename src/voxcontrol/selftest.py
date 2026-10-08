"""Self-test of the application (also of the packaged executable).

Checks, headless:
* the scientific core and the safety postconditions of the simulated desktop (an ambiguous target is
  never executed without the user's choice, only the chosen application changes, a high-risk action is
  confirmed, an unresolved slot never runs);
* reading WAV and FLAC files, mono and stereo;
* every example; example 02 uses the real speech path (Windows text-to-speech and the speech
  recognition model, which is downloaded on first use);
* one reduced experiment of each text type: main, ambiguity, asr_text_noise, incremental,
  combined_shift, personalization and ablation (the audio experiment E5 is not included);
* every tab, button, language and tutorial step of the interface, the option selector, the seeds loaded
  from the experiment files and the persistence of corrections. In this part the voice tab uses a
  simulated recogniser, so it checks the interface, not speech recognition.

Writes ``selftest_report.txt`` in the writable application folder and returns 0 when everything passes.
The self-test never touches the user's models, corrections or results: it works in ``selftest_results``.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

from . import SOFTWARE_NAME, __version__
from .paths import app_root, configure_caches, writable_root


def run(report_dir: Path | None = None, include_gui: bool = True, include_asr: bool = True) -> int:
    configure_caches()
    out_dir = report_dir or writable_root()
    lines: list[str] = [f"{SOFTWARE_NAME} {__version__} self-test", f"application folder: {app_root().name}", ""]
    failures = 0

    def check(name, fn):
        nonlocal failures
        t0 = time.time()
        try:
            detail = fn()
            lines.append(f"[OK]   {name} ({time.time() - t0:.1f} s){'  ' + str(detail) if detail else ''}")
        except Exception as e:  # noqa: BLE001 - every failure is reported, not raised
            failures += 1
            lines.append(f"[FAIL] {name}: {type(e).__name__}: {e}")
            lines.append("       " + traceback.format_exc().strip().splitlines()[-1])

    state: dict = {}

    def core():
        from .api import VoxModel
        m = VoxModel.train(seed=0, config={"dataset": {"variants_train": 4, "variants_cal": 3, "variants_test": 2}})
        r = m.process_text("abre spotify")
        assert r.intent == "open_app", r.intent
        if len(r.options) > 1:          # several possible intents: nothing runs until the user chooses
            assert not m.execute(r, confirmed=True).success and "spotify" not in m.sandbox.state.open_apps
        res = m.execute(r, confirmed=True, choice="open_app" if len(r.options) > 1 else None)
        assert res.success and "spotify" in m.sandbox.state.open_apps
        state["model"] = m
        return f"open_app -> {r.decision} {r.options or ''}, sandbox ok"
    check("scientific core: train, decide, sandbox", core)

    def safety():
        import copy
        from .api import VoxModel
        from .decision.policies import EXECUTE
        m = VoxModel(copy.deepcopy(state["model"].ucil))
        for app in ("whatsapp", "teams"):
            m.sandbox.execute("open_app", {"app": app})
        r = m.process_text("cierra el chat", context={})
        assert set(r.options) == {"close_app:whatsapp", "close_app:teams"}, r.options
        before = copy.deepcopy(m.sandbox.state.as_dict())
        assert not m.execute(r, confirmed=True).success and m.sandbox.state.as_dict() == before, \
            "an ambiguous target was executed without a choice"
        assert m.execute(r, confirmed=True, choice="close_app:teams").success
        assert m.sandbox.state.open_apps == ["explorer", "whatsapp"], m.sandbox.state.open_apps
        assert not m.sandbox.execute("close_app", {}).success, "closing without a target must be refused"
        r = m.process_text("borra el archivo informe.docx", context={})
        assert r.risk == "high" and r.decision != EXECUTE, "high-risk action not confirmed"
        assert not m.execute(r).success and "informe.docx" in m.sandbox.state.files
        return "ambiguous target, chosen option, missing target and high risk verified"
    check("safety postconditions of the simulated desktop", safety)

    def audio_files():
        import numpy as np
        import soundfile as sf
        from .audio.processing import TARGET_SR, load_audio, preprocess
        folder = out_dir / "selftest_results" / "audio"
        folder.mkdir(parents=True, exist_ok=True)
        sr = 22050
        tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
        for ch in (1, 2):
            for ext in ("wav", "flac"):
                path = folder / f"tone_{ch}ch.{ext}"
                sf.write(str(path), np.stack([tone] * ch, axis=1).astype(np.float32), sr)
                x, sr2 = load_audio(path)
                assert x.shape == (ch, sr) and sr2 == sr, (path.name, x.shape, sr2)
                assert abs(len(preprocess(x, sr2)) - TARGET_SR) <= 1
        return "WAV and FLAC, mono and stereo"
    check("audio files", audio_files)

    from .examples_catalog import EXAMPLES
    for key in EXAMPLES:
        if key == "example_02_voice_command" and not include_asr:
            continue
        def run_example(k=key):
            text = EXAMPLES[k].run("es")
            if k == "example_02_voice_command" and "Transcripción" not in text:
                raise RuntimeError(text.strip().splitlines()[0])
            return None
        check(f"example {key}", run_example)

    from .experiments.runner import run as run_experiment
    tiny = {"dataset": {"variants_train": 3, "variants_cal": 2, "variants_test": 2}, "checkpoints": [0, 5],
            "noise_rates": [0.0, 0.2]}
    for kind in ("main", "ambiguity", "asr_text_noise", "incremental", "combined_shift", "personalization",
                 "ablation"):
        cfg = {"experiment": {"name": f"selftest_{kind}", "type": kind, "seeds": [1]}, **tiny}
        check(f"experiment {kind} (reduced)", lambda c=cfg: run_experiment(
            c, out_dir / "selftest_results", progress=lambda m: None).id)

    if include_gui:
        check("graphical interface: every tab, button, language and tutorial step", lambda: _gui(state, out_dir))

    lines += ["", f"{'PASSED' if failures == 0 else 'FAILED'}: {failures} failure(s)"]
    (out_dir / "selftest_report.txt").write_text("\n".join(lines), encoding="utf-8")
    return 0 if failures == 0 else 1


def _gui(state, out_dir: Path) -> str:
    import copy
    import json
    import tkinter as tk

    from .gui import app as G
    from .gui.i18n import TUTORIAL
    errors: list[str] = []
    G.messagebox.showerror = lambda title, msg, **k: errors.append(str(msg))
    G.messagebox.showwarning = lambda title, msg, **k: errors.append(str(msg))
    G.messagebox.showinfo = lambda *a, **k: None
    G.messagebox.askokcancel = lambda *a, **k: True
    G.filedialog.asksaveasfilename = lambda **k: ""
    G.filedialog.askopenfilename = lambda **k: ""
    root = tk.Tk()
    root.withdraw()
    try:
        work = out_dir / "selftest_results" / "gui"          # never the user's models or corrections
        (work / "models" / "corrections.json").unlink(missing_ok=True)
        G.app_dir = lambda: work
        a = G.VoxControlApp(root, "es")
        a.model = copy.deepcopy(state["model"])
        a._after_model()

        def pump(cond, timeout=600):
            t = time.time()
            while not cond() and time.time() - t < timeout:
                root.update()
                time.sleep(0.02)
            return cond()

        for i, key in enumerate(a.tabs):
            a.show_tab(key)
            root.update()
        a.reset_sandbox()
        for app in ("whatsapp", "teams"):
            a.model.sandbox.execute("open_app", {"app": app})
        a.ctx_var.set("")
        a.analyse("cierra el chat")
        assert set(a.choice_combo["values"]) == {"close_app:whatsapp", "close_app:teams"}
        assert not a.confirm_and_execute().success, "the interface executed an ambiguous target"
        assert a.confirm_and_execute("close_app:teams").success
        assert a.model.sandbox.state.open_apps == ["explorer", "whatsapp"]
        a.analyse("cállalo")
        a.apply_correction("mute")
        assert len(json.loads((work / "models" / "corrections.json").read_text(encoding="utf-8"))) == 1
        a.reset_sandbox()
        a._refresh_corrections()
        cfgs = list(a.cfg_combo["values"])
        if "exp_main.yaml" in cfgs:
            a.cfg_var.set("exp_main.yaml")
            a._load_config_seeds()
            assert a._seeds() == list(range(100, 110)), a._seeds()
        import numpy as np
        from .asr.recognizers import Transcript

        class _Recognizer:      # exercises the voice path without a microphone or speech model
            def load(self):
                return self

            def transcribe(self, audio, sr=16000):
                return Transcript(text="abre spotify", confidence=0.9, uncertainty=0.1)
        a._get_recognizer = lambda model, lang: _Recognizer()
        a._process_audio(lambda: np.zeros(16000, dtype=np.float32))
        assert pump(lambda: not a.busy) and a.last_result.intent == "open_app"
        a.run_selected_example("example_06_risk_routing")
        assert pump(lambda: ("example", "example_06_risk_routing") in a.events)
        a._refresh_thresholds()
        a.toggle_language()
        assert a.nb.tab(0, "text") == "Home"
        a.toggle_language()
        tut = G.Tutorial(a)
        for _ in range(len(TUTORIAL)):
            tut.render()
            tut.go(1)
        tut.win.destroy()
        a.refresh_results()
        if a.table_combo["values"]:
            a._show_table()
        if a.fig_combo["values"]:
            a._show_figure()
        a.clear_corrections(ask=False)
        if errors:
            raise RuntimeError(" | ".join(errors))
        return f"{len(a.tabs)} tabs, {len(TUTORIAL)} tutorial steps (voice tab with a simulated recogniser)"
    finally:
        root.destroy()
