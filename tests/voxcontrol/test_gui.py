"""Automatic GUI tests: every button/menu action, language toggle and the tutorial."""
import copy
import time
from pathlib import Path

import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture
def gui(model, tmp_path, monkeypatch):
    from voxcontrol.api import VoxModel
    from voxcontrol.gui import app as G
    (tmp_path / "configs").mkdir()
    monkeypatch.setattr(G, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(G.messagebox, "showerror", lambda *a, **k: pytest.fail(f"error dialog: {a}"))
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    app = G.VoxControlApp(root, "es")
    app.model = VoxModel(copy.deepcopy(model))
    app._after_model()
    yield app, root, G
    root.destroy()


def _pump(root, cond, timeout=120):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        root.update()
        time.sleep(0.02)
    root.update()
    return cond()


def test_tabs_and_language_toggle(gui):
    app, root, _ = gui
    assert app.nb.tab(0, "text") == "Inicio"
    app.toggle_language()
    assert app.nb.tab(0, "text") == "Home"
    assert app.example_combo.get().startswith("01 · Text intent")
    app.toggle_language()
    assert app.nb.tab(6, "text") == "Configuración"


def test_text_analysis_correction_and_sandbox(gui):
    app, root, _ = gui
    r = app.analyse("abre spotify")
    assert r.intent == "open_app" and app.fields["f_intent"].cget("text").startswith("open_app")
    assert len(app.factor_tree.get_children()) >= 10
    app.confirm_and_execute()
    assert "spotify" in app.model.sandbox.state.open_apps
    app.reset_sandbox()
    assert "spotify" not in app.model.sandbox.state.open_apps
    app.analyse("cállalo")
    app.apply_correction("mute")
    assert "corrected" in app.events and len(app.corr_tree.get_children()) == 1


def test_examples_dropdown_runs_named_example(gui):
    app, root, _ = gui
    from voxcontrol.examples_catalog import EXAMPLES
    assert len(app.example_combo["values"]) == len(EXAMPLES) == 10
    app.run_selected_example("example_06_risk_routing")
    assert _pump(root, lambda: ("example", "example_06_risk_routing") in app.events)
    assert "umbral" in app.example_out.get("1.0", "end")


def test_settings_thresholds_update(gui):
    app, root, _ = gui
    app.param_vars["p_cerr_high"].set(100.0)
    app._refresh_thresholds()
    assert "0.99" in app.thr_label.cget("text")


def test_tutorial_steps_render_and_check(gui):
    app, root, G = gui
    tut = G.Tutorial(app)
    for _ in range(len(G.TUTORIAL)):
        assert tut.title.cget("text")
        tut.go(1)
    tut.i = 2
    tut.render()
    tut.do_it()
    assert tut.done()
    app.toggle_language()
    assert tut.next_btn.cget("text") == "Next ›"
    tut.win.destroy()


def test_experiment_run_and_results_view(gui):
    app, root, _ = gui
    cfg_dir = app._config_dir()
    (cfg_dir / "exp_tiny.yaml").write_text(
        "experiment: {name: tiny, type: incremental, seeds: [1]}\n"
        "dataset: {variants_train: 3, variants_cal: 2, variants_test: 2}\n"
        "checkpoints: [0, 5]\n", encoding="utf-8")
    app.cfg_combo["values"] = app._config_files()
    app.run_experiment("exp_tiny.yaml", "1")
    assert _pump(root, lambda: "experiment" in app.events, 300)
    assert app.res_var.get().startswith("EXP-")
    assert app.table_combo["values"] and app.fig_combo["values"]
    assert len(app.res_tree.get_children()) > 0


def test_voice_processing_runs_in_worker_without_touching_tk(gui):
    """Regression: the worker thread must not read Tk variables (RuntimeError outside the main loop)."""
    import numpy as np
    from voxcontrol.asr.recognizers import Transcript
    app, root, _ = gui

    class FakeRecognizer:
        def load(self):
            return self

        def transcribe(self, audio, sr=16000):
            return Transcript(text="abre spotify", confidence=0.9, uncertainty=0.1)

    app._get_recognizer = lambda model, lang: FakeRecognizer()
    app._process_audio(lambda: np.zeros(16000, dtype=np.float32))
    assert _pump(root, lambda: not app.busy)
    assert "abre spotify" in app.voice_out.get("1.0", "end")
    assert app.last_result.intent == "open_app"
