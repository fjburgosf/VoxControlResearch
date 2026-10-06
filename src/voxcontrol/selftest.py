"""Self-test of the application (also of the packaged executable).

Runs headless checks of the scientific core, every example, one complete reduced experiment of each
text type, and drives every button, menu and tutorial step of the interface. Writes
``selftest_report.txt`` in the writable application folder and returns 0 when everything passes.
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
        res = m.execute(r, confirmed=True)
        assert res.success and "spotify" in m.sandbox.state.open_apps
        state["model"] = m
        return f"open_app -> {r.decision}, sandbox ok"
    check("scientific core: train, decide, sandbox", core)

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
    for kind in ("main", "ambiguity", "asr_text_noise", "incremental", "combined_shift"):
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
    import tkinter as tk

    from .gui import app as G
    from .gui.i18n import TUTORIAL
    errors: list[str] = []
    G.messagebox.showerror = lambda title, msg, **k: errors.append(str(msg))
    G.messagebox.showwarning = lambda title, msg, **k: errors.append(str(msg))
    G.messagebox.showinfo = lambda *a, **k: None
    G.filedialog.asksaveasfilename = lambda **k: ""
    G.filedialog.askopenfilename = lambda **k: ""
    root = tk.Tk()
    root.withdraw()
    try:
        a = G.VoxControlApp(root, "es")
        a.base = out_dir
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
        a.analyse("cierra el chat")
        a.confirm_and_execute()
        a.analyse("cállalo")
        a.apply_correction("mute")
        a.reset_sandbox()
        a._refresh_corrections()
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
        if errors:
            raise RuntimeError("; ".join(errors))
        return f"{len(a.tabs)} tabs, {len(TUTORIAL)} tutorial steps"
    finally:
        root.destroy()
