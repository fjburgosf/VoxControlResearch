"""Regression tests for confirmation safety, persistence of corrections and reproducibility settings.

R01-R10 of the regression protocol: ambiguous targets need an explicit choice, unresolved slots never
run, mandatory confirmation of high-risk actions also applies on the simulated desktop, corrections
survive a restart and a retraining, experiment seeds come from the YAML file, a single seed gives no
confidence interval, the B3 threshold of the experiment file is effective, the context prior acts only
when enabled, and FLAC/WAV files are read.
"""
import copy
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from voxcontrol.decision import CONFIRM, EXECUTE

PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture
def vm(model):
    from voxcontrol.api import VoxModel
    return VoxModel(copy.deepcopy(model))


def _open(vm, *apps):
    for a in apps:
        assert vm.sandbox.execute("open_app", {"app": a}).success


# ---------------------------------------------------------------- R01-R03: no action without a target
def test_R01_ambiguous_editor_needs_a_choice(vm):
    r = vm.process_text("open the editor", context={})
    assert r.intent == "open_app" and r.slot_status == "ambiguous" and r.decision == CONFIRM
    assert set(r.options) == {"open_app:notepad", "open_app:vscode", "open_app:paint"}
    before = copy.deepcopy(vm.sandbox.state.as_dict())
    res = vm.execute(r, confirmed=True)
    assert not res.success and res.key == "needs_choice"
    assert vm.sandbox.state.as_dict() == before


def test_R02_close_chat_closes_only_the_chosen_application(vm):
    _open(vm, "whatsapp", "teams")
    r = vm.process_text("cierra el chat", context={})
    assert r.intent == "close_app" and r.decision == CONFIRM
    assert set(r.options) == {"close_app:whatsapp", "close_app:teams"}
    before = copy.deepcopy(vm.sandbox.state.as_dict())
    assert not vm.execute(r, confirmed=True).success
    assert vm.sandbox.state.as_dict() == before                       # explorer and both chats still open
    res = vm.execute(r, confirmed=True, choice="close_app:teams")
    assert res.success and res.slots == {"app": "teams"}
    assert vm.sandbox.state.open_apps == ["explorer", "whatsapp"]
    with pytest.raises(ValueError):
        vm.execute(r, confirmed=True, choice="close_app:explorer")   # not one of the options


def test_R03_missing_or_invalid_slots_never_run(vm):
    from voxcontrol.ucil import Result
    base = vm.process_text("abre spotify", context={})
    for status, slots in (("missing", {}), ("invalid", {"value": 150})):
        r = copy.copy(base)
        r.intent, r.slots, r.slot_status, r.decision, r.options = "set_volume", slots, status, CONFIRM, []
        assert isinstance(r, Result)
        before = copy.deepcopy(vm.sandbox.state.as_dict())
        res = vm.execute(r, confirmed=True)
        assert not res.success and res.key == "unresolved_slot"
        assert vm.sandbox.state.as_dict() == before
    # the simulated desktop itself never closes "whatever is active" when no application is given
    assert not vm.sandbox.execute("close_app", {}).success and "explorer" in vm.sandbox.state.open_apps


def test_confirm_decision_needs_explicit_confirmation(vm):
    r = vm.process_text("cierra el chat", context={})
    before = copy.deepcopy(vm.sandbox.state.as_dict())
    res = vm.execute(r, confirmed=False, choice=r.options[0])
    assert not res.success and res.key == "needs_confirmation" and vm.sandbox.state.as_dict() == before


# ------------------------------------------------------------- R04: mandatory confirmation in sandbox
def test_R04_high_risk_is_never_executed_directly_in_sandbox(model, data):
    from voxcontrol.api import VoxModel
    from voxcontrol.decision import CostModel
    cheap = copy.deepcopy(model)          # a cost model under which UCIL itself would run high-risk actions
    cheap.config.costs = CostModel(error={"low": 2.0, "medium": 5.0, "high": 0.4}, confirm=0.3, reject=1.0)
    strict, free = VoxModel(copy.deepcopy(cheap)), VoxModel(copy.deepcopy(cheap), mandatory_confirm_risk=())
    high = [s for s in data.test if model.registry.risk(s.intent) == "high"][:60]
    assert high
    executed_without_rule = 0
    for s in high:
        r = strict.process_text(s.text, context={})
        if r.risk == "high":
            assert r.decision != EXECUTE
        executed_without_rule += free.process_text(s.text, context={}).decision == EXECUTE
    assert executed_without_rule > 0          # the rule is what prevents direct execution


# ------------------------------------------------------------------- R05: corrections are persistent
def test_R05_corrections_survive_restart_and_retraining(model, tmp_path):
    from voxcontrol.api import VoxModel
    a = VoxModel(copy.deepcopy(model))
    before = a.process_text("cállalo ya", user="ana", context={}).intent
    a.correct("cállalo ya", "mute", user="ana")
    path = tmp_path / "corrections.json"
    a.save_corrections(path)
    b = VoxModel(copy.deepcopy(model))                 # a new session or a retrained global model
    assert b.load_corrections(path) == 1 and b.load_corrections(path) == 0      # no duplicates
    assert b.process_text("cállalo ya", user="ana", context={}).intent == "mute"
    assert b.process_text("cállalo ya", user="otro", context={}).intent == before
    b.clear_corrections()
    b.save_corrections(path)
    assert VoxModel(copy.deepcopy(model)).load_corrections(path) == 0


# ------------------------------------------------------------------------- R07: no fictitious CI
def test_R07_single_seed_has_no_confidence_interval(tmp_path):
    from voxcontrol.io.results import read_csv, summarise, write_csv
    from voxcontrol.metrics.stats import describe
    one = describe([0.5])
    assert math.isnan(one["sd"]) and math.isnan(one["ci95_low"]) and math.isnan(one["ci95_high"])
    ten = describe(np.linspace(0.4, 0.6, 10))
    assert ten["ci95_low"] < ten["mean"] < ten["ci95_high"]
    write_csv(tmp_path / "s.csv", summarise([{"method": "UCIL", "seed": 1, "cost": 0.5}], ["method"]))
    row = read_csv(tmp_path / "s.csv")[0]
    assert row["ci95_low"] == "" and row["ci95_high"] == "" and row["sd"] == ""      # never "nan"


# ---------------------------------------------------------------------- R08: B3 threshold is used
def test_R08_fixed_threshold_of_the_experiment_changes_B3(model, data):
    from voxcontrol.baselines import BaselineSuite
    ev = model.evidence([s.text for s in data.test[:200]])
    low = BaselineSuite(model, tau_fixed=0.2).run(data.test[:200], ev, ucil_ev=ev)["B3_fixed_threshold"]
    high = BaselineSuite(model, tau_fixed=0.99).run(data.test[:200], ev, ucil_ev=ev)["B3_fixed_threshold"]
    assert np.sum(low.decisions == EXECUTE) > np.sum(high.decisions == EXECUTE)


# ---------------------------------------------------------------------- R09: context prior switch
def test_R09_context_prior_acts_only_when_enabled(data):
    from voxcontrol.ucil import UCIL, UCILConfig
    sub = data.train[:600] + data.train[-600:]
    off = UCIL(config=UCILConfig(seed=3)).fit(sub, data.cal, data.ood_cal)
    on = UCIL(config=UCILConfig(seed=3, use_context=True)).fit(sub, data.cal, data.ood_cal)
    ctx = {"active_app": "spotify", "category": on.registry.app_category("spotify")}
    assert ctx["category"] == "music_player"
    assert off.process_text("súbelo", context=ctx).explanation["context_shift"] == 0.0
    assert on.process_text("súbelo", context=ctx).explanation["context_shift"] > 0.05
    # an unknown category leaves the distribution unchanged (only rounding noise)
    unknown = {"active_app": "spotify", "category": "not_a_category"}
    assert on.process_text("súbelo", context=unknown).explanation["context_shift"] < 1e-9


# --------------------------------------------------------------------------- R10: WAV and FLAC
@pytest.mark.parametrize("channels", [1, 2])
def test_R10_wav_and_flac_files_are_read(tmp_path, channels):
    sf = pytest.importorskip("soundfile")
    from voxcontrol.audio.processing import TARGET_SR, load_audio, preprocess
    sr = 22050
    t = np.arange(sr) / sr
    x = np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * channels, axis=1).astype(np.float32)
    for ext in ("wav", "flac"):
        path = tmp_path / f"tone_{channels}.{ext}"
        sf.write(str(path), x, sr)
        y, sr2 = load_audio(path)
        assert sr2 == sr and y.shape == (channels, sr)
        mono = preprocess(y, sr2)
        assert mono.ndim == 1 and abs(len(mono) - TARGET_SR) <= 1


# ------------------------------------------------------------------------- GUI: R01, R05, R06
@pytest.fixture
def gui(model, tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    from voxcontrol.api import VoxModel
    from voxcontrol.gui import app as G
    (tmp_path / "configs").mkdir()
    for name in ("exp_main.yaml", "exp_audio_noise.yaml"):
        shutil.copy(PROJECT / "configs" / name, tmp_path / "configs" / name)
    monkeypatch.setattr(G, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(G.messagebox, "showerror", lambda *a, **k: pytest.fail(f"error dialog: {a}"))
    monkeypatch.setattr(G.messagebox, "askokcancel", lambda *a, **k: True)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    apps = []

    def new_app():
        a = G.VoxControlApp(root, "es")
        a.model = VoxModel(copy.deepcopy(model))
        a._after_model()
        apps.append(a)
        return a
    yield new_app, root
    root.destroy()


def test_gui_confirm_requires_selected_option(gui):
    new_app, root = gui
    app = new_app()
    _open(app.model, "whatsapp", "teams")
    app.analyse("cierra el chat")
    assert set(app.choice_combo["values"]) == {"close_app:whatsapp", "close_app:teams"}
    assert app.choice_var.get() == ""
    res = app.confirm_and_execute()
    assert not res.success and set(app.model.sandbox.state.open_apps) == {"explorer", "whatsapp", "teams"}
    assert "elija" in app.action_msg.cget("text")
    res = app.confirm_and_execute("close_app:whatsapp")
    assert res.success and app.model.sandbox.state.open_apps == ["explorer", "teams"]
    app.analyse("abre spotify")
    assert app.choice_var.get() == ""                         # a new command never inherits a choice


def test_gui_corrections_persist_after_restart(gui):
    new_app, root = gui
    app = new_app()
    app.analyse("cállalo")
    app.apply_correction("mute")
    assert app._corrections_path().exists()
    again = new_app()                                         # program closed and opened again
    assert len(again.model.ucil.memory.records) == 1 and len(again.corr_tree.get_children()) == 1
    again.clear_corrections(ask=False)
    assert len(new_app().model.ucil.memory.records) == 0


def test_gui_loads_seeds_of_the_selected_configuration(gui):
    new_app, root = gui
    app = new_app()
    app.cfg_var.set("exp_main.yaml")
    app._load_config_seeds()
    assert app._seeds() == list(range(100, 110)) and "10" in app.runs_label.cget("text")
    app.cfg_var.set("exp_audio_noise.yaml")
    app._load_config_seeds()
    assert app._seeds() == [100, 101, 102]


def test_gui_single_seed_run_records_its_seed_and_shows_na(gui):
    new_app, root = gui
    app = new_app()
    (app._config_dir() / "exp_tiny.yaml").write_text(
        "experiment: {name: tiny, type: incremental, seeds: [1, 2]}\n"
        "dataset: {variants_train: 3, variants_cal: 2, variants_test: 2}\n"
        "checkpoints: [0, 5]\n", encoding="utf-8")
    app.cfg_combo["values"] = app._config_files()
    app.cfg_var.set("exp_tiny.yaml")
    app._load_config_seeds()
    assert app._seeds() == [1, 2]
    app.run_experiment("exp_tiny.yaml", "1")
    t = time.time()
    while "experiment" not in app.events and time.time() - t < 300:
        root.update()
        time.sleep(0.02)
    import json
    meta = json.loads((app._results_dir() / app.res_var.get() / "metadata.json").read_text(encoding="utf-8"))
    assert meta["seeds"] == [1]
    assert "NA" in app.exp_log.get("1.0", "end")
    app.show_result_item(table=next(t for t in app.table_combo["values"] if t.startswith("summary_")))
    rows = [app.res_tree.item(i, "values") for i in app.res_tree.get_children()]
    assert rows and all("NA" in r for r in rows)


def test_gui_action_message_follows_language(gui):
    new_app, root = gui
    app = new_app()
    _open(app.model, "whatsapp", "teams")
    app.analyse("cierra el chat")
    app.confirm_and_execute("close_app:teams")
    assert "se cerró teams" in app.action_msg.cget("text")
    app.toggle_language()
    assert "closed teams" in app.action_msg.cget("text")
    app.toggle_language()
    assert "se cerró teams" in app.action_msg.cget("text")
    app.analyse("abre spotify")                       # a new command clears the earlier message
    assert app.action_msg.cget("text") == ""
