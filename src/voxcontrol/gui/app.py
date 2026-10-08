"""VoxControlResearch graphical interface (tkinter, bilingual ES/EN)."""
from __future__ import annotations

import copy
import csv
import os
import queue
import shutil
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .. import SOFTWARE_NAME, __version__
from ..decision.policies import CostModel
from ..examples_catalog import EXAMPLES
from ..intents.registry import IntentRegistry
from .i18n import COLUMNS, FACTORS, I18N, SANDBOX, TERMS_ES, TUTORIAL, VALUES

DECISION_COLOURS = {"execute": "#1baf7a", "confirm": "#eda100", "reject": "#e34948"}


def app_dir() -> Path:
    from ..paths import writable_root
    return writable_root()


def config_dir() -> Path:
    from ..paths import app_root
    return app_root() / "configs"


class VoxControlApp:
    def __init__(self, root: tk.Tk, lang: str = "es"):
        self.root = root
        self.i18n = I18N(lang)
        self.registry = IntentRegistry.default()
        self.model = None                 # api.VoxModel
        self.last_result = None
        self.events: set = set()          # tutorial progress markers
        self.q: queue.Queue = queue.Queue()
        self.busy = False
        self.base = app_dir()
        self.settings = {"seed": 0, "predictor": "tfidf_logreg", "calibrator": "auto", "ood_detector": "fused",
                         "memory_threshold": 0.70, "use_context": False, "mandatory": True,
                         "costs": {"error": {"low": 2.0, "medium": 5.0, "high": 20.0}, "confirm": 0.3, "reject": 1.0}}
        root.title(f"{SOFTWARE_NAME} {__version__}")
        root.geometry("1180x780")
        root.minsize(980, 640)
        root.report_callback_exception = self._report_exception
        self._build()
        self.root.after(60, self._poll)
        self._try_load_cached_model()

    # ================================================================ infra
    def t(self, key, **kw):
        return self.i18n.t(key, **kw)

    def _report_exception(self, exc, val, tb):
        messagebox.showerror(self.t("error"), "".join(traceback.format_exception_only(exc, val)).strip())

    def _run_bg(self, fn, done=None, label="busy"):
        """Run ``fn`` in a worker thread; ``done(result)`` runs on the UI thread."""
        if self.busy:
            return
        self.busy = True
        self.status_var.set(self.t(label))

        def worker():
            try:
                res = fn()
                self.q.put(("ok", done, res))
            except Exception as e:  # noqa: BLE001 - shown to the user in a dialog
                self.q.put(("err", None, f"{type(e).__name__}: {e}"))
        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, cb, payload = self.q.get_nowait()
                if kind == "log":
                    cb(payload)
                    continue
                self.busy = False
                self._refresh_status()
                if kind == "err":
                    messagebox.showerror(self.t("error"), payload)
                elif cb:
                    cb(payload)
        except queue.Empty:
            pass
        self.root.after(60, self._poll)

    def _need_model(self) -> bool:
        if self.model is None:
            messagebox.showwarning(SOFTWARE_NAME, self.t("need_model"))
            return False
        return True

    # ================================================================ layout
    def _build(self):
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text=SOFTWARE_NAME, font=("Segoe UI", 14, "bold")).pack(side="left")
        self.subtitle = self.i18n.bind(ttk.Label(top, foreground="#52514e"), "app_subtitle")
        self.subtitle.pack(side="left", padx=12)
        self.i18n.bind(ttk.Button(top, command=self.toggle_language), "lang_toggle").pack(side="right")
        self.i18n.bind(ttk.Button(top, command=self.open_tutorial), "tutorial").pack(side="right", padx=6)
        self.status_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.status_var, foreground="#2a78d6", padding=(10, 0)).pack(fill="x")

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.tabs = {}
        for key, builder in (("home", self._tab_home), ("text", self._tab_text), ("voice", self._tab_voice),
                             ("adapt", self._tab_adapt), ("exp", self._tab_exp), ("results", self._tab_results),
                             ("settings", self._tab_settings)):
            f = ttk.Frame(self.nb, padding=10)
            self.nb.add(f, text=self.t(f"tab_{key}"))
            self.tabs[key] = f
            builder(f)
        self.i18n.listeners.append(self._relabel)
        self._refresh_status()

    def _relabel(self):
        for i, key in enumerate(self.tabs):
            self.nb.tab(i, text=self.t(f"tab_{key}"))
        idx = max(self.example_combo.current(), 0)
        self.example_combo["values"] = [e.title[self.i18n.lang] for e in EXAMPLES.values()]
        self.example_combo.current(idx)
        self._on_example_selected()
        self._refresh_status()
        self._refresh_thresholds()
        if self.last_result is not None:
            self._show_result(self.last_result)
        if self.table_var.get():
            self._show_table()
        if self.fig_var.get():
            self._show_figure()
        self._refresh_corrections()
        self._show_action()

    def toggle_language(self):
        self.i18n.toggle()

    def show_tab(self, key: str):
        self.nb.select(list(self.tabs).index(key))

    def _refresh_status(self):
        if self.model is None:
            self.status_var.set(self.t("model_none"))
        else:
            u = self.model.ucil
            v = lambda x: VALUES.get(x, {}).get(self.i18n.lang, x)
            self.status_var.set(self.t("model_ready", seed=u.config.seed, cal=v(u.calibrator_name), ood=v(u.ood_name)))

    # ---------------------------------------------------------------- home
    def _tab_home(self, f):
        self.i18n.bind(ttk.Label(f, wraplength=1050, justify="left"), "home_intro").pack(fill="x")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=10)
        self.i18n.bind(ttk.Button(row, command=self.train_model), "train_model").pack(side="left")
        self.i18n.bind(ttk.Button(row, command=self.load_model_dialog), "load_model").pack(side="left", padx=6)
        self.i18n.bind(ttk.Button(row, command=self.save_model_dialog), "save_model").pack(side="left")
        box = self.i18n.bind(ttk.LabelFrame(f, padding=8), "examples")
        box.pack(fill="both", expand=True)
        r = ttk.Frame(box)
        r.pack(fill="x")
        self.example_var = tk.StringVar()
        self.example_combo = ttk.Combobox(r, textvariable=self.example_var, state="readonly", width=48,
                                          values=[e.title[self.i18n.lang] for e in EXAMPLES.values()])
        self.example_combo.pack(side="left")
        self.example_combo.bind("<<ComboboxSelected>>", lambda e: self._on_example_selected())
        self.i18n.bind(ttk.Button(r, command=self.run_selected_example), "run_example").pack(side="left", padx=8)
        self.example_desc = ttk.Label(box, foreground="#52514e", wraplength=1000)
        self.example_desc.pack(fill="x", pady=4)
        self.example_out = scrolledtext.ScrolledText(box, height=18, font=("Consolas", 9), wrap="none")
        self.example_out.pack(fill="both", expand=True)
        self.example_combo.current(0)
        self._on_example_selected()

    def _selected_example_key(self) -> str:
        i = max(self.example_combo.current(), 0)
        return list(EXAMPLES)[i]

    def _on_example_selected(self):
        self.example_desc.configure(text=EXAMPLES[self._selected_example_key()].description[self.i18n.lang])

    def run_selected_example(self, key: str | None = None):
        if key is not None:
            self.example_combo.current(list(EXAMPLES).index(key))
            self._on_example_selected()
        key = self._selected_example_key()

        def done(text):
            self.example_out.delete("1.0", "end")
            self.example_out.insert("end", text)
            self.events.add(("example", key))
        self._run_bg(lambda: EXAMPLES[key].run(self.i18n.lang), done)

    # ------------------------------------------------------------ model mgmt
    def _model_path(self) -> Path:
        return self.base / "models" / "ucil_default.pkl"

    def _corrections_path(self) -> Path:
        return self.base / "models" / "corrections.json"

    def _try_load_cached_model(self):
        p = self._model_path()
        if p.exists():
            try:
                from ..api import VoxModel
                self.model = VoxModel.load(p)
                self._after_model()
            except Exception:  # noqa: BLE001 - a stale cache is simply ignored
                self.model = None
        self._refresh_status()

    def _after_model(self):
        self.model.mandatory_confirm_risk = ("high",) if self.settings["mandatory"] else ()
        self.model.load_corrections(self._corrections_path())    # corrections survive restarts and retraining
        self.events.add("model")
        self._refresh_status()
        self._refresh_corrections()
        self._refresh_sandbox()

    def train_model(self):
        s = self.settings

        def work():
            from ..api import VoxModel
            cfg = {"ucil": {"predictor": s["predictor"], "calibrator": s["calibrator"],
                            "ood_detector": s["ood_detector"], "memory_threshold": s["memory_threshold"],
                            "use_context": s["use_context"]},
                   "costs": s["costs"]}
            m = VoxModel.train(seed=int(s["seed"]), config=cfg)
            p = self._model_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            m.save(p)
            return m

        def done(m):
            self.model = m
            self._after_model()
        self._run_bg(work, done)

    def load_model_dialog(self):
        if not messagebox.askokcancel(SOFTWARE_NAME, self.t("load_warning"), icon="warning"):
            return
        path = filedialog.askopenfilename(filetypes=[("VoxControl model", "*.pkl")])
        if path:
            from ..api import VoxModel
            self.model = VoxModel.load(path)
            self._after_model()

    def save_model_dialog(self):
        if not self._need_model():
            return
        path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("VoxControl model", "*.pkl")])
        if path:
            self.model.save(path)

    # ------------------------------------------------------------- text tab
    def _tab_text(self, f):
        row = ttk.Frame(f)
        row.pack(fill="x")
        self.i18n.bind(ttk.Label(row), "command").pack(side="left")
        self.cmd_var = tk.StringVar()
        e = ttk.Entry(row, textvariable=self.cmd_var, width=60)
        e.pack(side="left", padx=6)
        e.bind("<Return>", lambda ev: self.analyse())
        self.i18n.bind(ttk.Button(row, command=self.analyse), "analyse").pack(side="left")
        self.i18n.bind(ttk.Label(row), "user").pack(side="left", padx=(16, 4))
        self.user_var = tk.StringVar(value="default")
        ttk.Entry(row, textvariable=self.user_var, width=12).pack(side="left")
        row2 = ttk.Frame(f)
        row2.pack(fill="x", pady=6)
        self.i18n.bind(ttk.Label(row2), "active_app").pack(side="left")
        self.ctx_var = tk.StringVar(value="")
        ttk.Combobox(row2, textvariable=self.ctx_var, state="readonly", width=18,
                     values=[""] + sorted(self.registry.apps)).pack(side="left", padx=6)

        body = ttk.Frame(f)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        self.fields = {}
        grid = ttk.Frame(left)
        grid.pack(fill="x", pady=4)
        for i, key in enumerate(("f_intent", "f_slots", "f_raw", "f_cal", "f_pin", "f_ood", "f_risk",
                                 "f_decision", "f_options")):
            self.i18n.bind(ttk.Label(grid, foreground="#52514e"), key).grid(row=i, column=0, sticky="w", pady=1)
            v = ttk.Label(grid, text="—", font=("Segoe UI", 10, "bold" if key == "f_decision" else "normal"))
            v.grid(row=i, column=1, sticky="w", padx=10)
            self.fields[key] = v
        fac = self.i18n.bind(ttk.LabelFrame(left, padding=6), "factors")
        fac.pack(fill="both", expand=True, pady=6)
        self.factor_tree = ttk.Treeview(fac, columns=("factor", "value"), show="headings", height=11)
        self.factor_tree.column("factor", width=260)
        self.factor_tree.column("value", width=200)
        self.factor_tree.pack(fill="both", expand=True)

        right = ttk.Frame(body, padding=(12, 0))
        right.pack(side="left", fill="both", expand=True)
        ch = ttk.Frame(right)
        ch.pack(fill="x", pady=(0, 6))
        self.i18n.bind(ttk.Label(ch), "choice").pack(side="left")
        self.choice_var = tk.StringVar()
        self.choice_combo = ttk.Combobox(ch, textvariable=self.choice_var, state="disabled", width=30)
        self.choice_combo.pack(side="left", padx=6)
        self.i18n.bind(ttk.Button(right, command=self.confirm_and_execute), "confirm_exec").pack(fill="x")
        cr = ttk.Frame(right)
        cr.pack(fill="x", pady=8)
        self.i18n.bind(ttk.Label(cr), "correct_as").pack(side="left")
        self.correct_var = tk.StringVar()
        ttk.Combobox(cr, textvariable=self.correct_var, state="readonly", values=self.registry.names,
                     width=18).pack(side="left", padx=6)
        self.i18n.bind(ttk.Button(cr, command=self.apply_correction), "apply_correction").pack(side="left")
        self.action_msg = ttk.Label(right, foreground="#2a78d6", wraplength=420)
        self.action_msg.pack(fill="x")
        sb = self.i18n.bind(ttk.LabelFrame(right, padding=6), "sandbox_state")
        sb.pack(fill="both", expand=True, pady=6)
        self.sandbox_text = scrolledtext.ScrolledText(sb, height=16, font=("Consolas", 9))
        self.sandbox_text.pack(fill="both", expand=True)
        self.i18n.bind(ttk.Button(right, command=self.reset_sandbox), "reset_sandbox").pack(anchor="e")

    def analyse(self, text: str | None = None):
        if not self._need_model():
            return None
        if text is not None:
            self.cmd_var.set(text)
        text = self.cmd_var.get().strip()
        if not text:
            return None
        ctx = self.model.sandbox.context()
        if self.ctx_var.get():
            ctx = {"active_app": self.ctx_var.get(), "category": self.registry.app_category(self.ctx_var.get())}
        r = self.model.process_text(text, user=self.user_var.get() or "default", context=ctx)
        self.last_result = r
        self._last_action = (None, None)       # the message of an earlier action no longer applies
        self._show_action()
        self._show_result(r)
        self.events.add(("analysed", text))
        return r

    def _show_result(self, r):
        f = self.fields
        f["f_intent"].configure(text=f"{r.intent}  —  {self.registry.spec(r.intent).describe(self.i18n.lang)}")
        val = lambda v: VALUES.get(v, {}).get(self.i18n.lang, v)
        f["f_slots"].configure(text=f"{r.slots or '—'}  [{val(r.slot_status)}]")
        f["f_raw"].configure(text=f"{r.confidence:.3f}")
        f["f_cal"].configure(text=f"{r.calibrated_confidence:.3f}")
        f["f_pin"].configure(text=f"{r.p_in_domain:.3f}")
        f["f_ood"].configure(text=f"{r.ood_score:+.2f}")
        f["f_risk"].configure(text=val(r.risk))
        f["f_decision"].configure(text=self.t(r.decision.upper()), foreground=DECISION_COLOURS[r.decision])
        f["f_options"].configure(text=", ".join(r.options) or "—")
        if self.choice_combo["values"] != tuple(r.options) or not r.options:
            self.choice_var.set("")                # a new result never inherits an earlier choice
        self.choice_combo.configure(values=r.options, state="readonly" if r.options else "disabled")
        self.factor_tree.delete(*self.factor_tree.get_children())
        self.factor_tree.heading("factor", text=self.t("factor"))
        self.factor_tree.heading("value", text=self.t("value"))
        lang = self.i18n.lang
        flat = {k: v for k, v in r.explanation.items() if k != "expected_costs"}
        flat.update({f"expected_cost_{d}": v for d, v in r.explanation.get("expected_costs", {}).items()})
        for k, v in flat.items():
            if isinstance(v, float):
                v = f"{v:.4f}"
            elif isinstance(v, str):
                v = val(v)
            self.factor_tree.insert("", "end", values=(FACTORS.get(k, {}).get(lang, k), v))
        if r.options and r.options[0].split(":")[0] in self.registry:
            self.correct_var.set(r.options[0].split(":")[0])

    def confirm_and_execute(self, choice: str | None = None):
        if self.last_result is None:
            return None
        if choice is not None:
            self.choice_var.set(choice)
        res = self.model.execute(self.last_result, confirmed=True, choice=self.choice_var.get() or None)
        self._last_action = ("executed", res)
        self._show_action()
        self._refresh_sandbox()
        return res

    def _show_action(self):
        """Message of the last action, rendered in the current language (also after switching language)."""
        kind, value = getattr(self, "_last_action", (None, None))
        if kind == "executed":
            es = self.i18n.lang == "es" and value.key in SANDBOX
            msg = SANDBOX[value.key].format(**value.params) if es else value.message
            text = self.t("executed" if value.success else "not_executed", msg=msg)
        elif kind == "corrected":
            text = self.t("corrected", user=value)
        else:
            text = ""
        self.action_msg.configure(text=text)

    def apply_correction(self, intent: str | None = None):
        if self.last_result is None or not self._need_model():
            return
        intent = intent or self.correct_var.get()
        if not intent:
            return
        user = self.user_var.get() or "default"
        self.model.correct(self.last_result, intent, user=user)
        self.model.save_corrections(self._corrections_path())
        self._last_action = ("corrected", user)
        self._show_action()
        self.events.add("corrected")
        self._refresh_corrections()

    def reset_sandbox(self):
        if self.model is not None:
            from ..sandbox.desktop import Sandbox
            self.model.sandbox = Sandbox(self.registry)
            self._last_action = (None, None)
            self._show_action()
            self._refresh_sandbox()

    def _refresh_sandbox(self):
        if self.model is None:
            return
        import json
        self.sandbox_text.delete("1.0", "end")
        self.sandbox_text.insert("end", json.dumps(self.model.sandbox.state.as_dict(), indent=1, ensure_ascii=False))

    # ------------------------------------------------------------ voice tab
    def _tab_voice(self, f):
        row = ttk.Frame(f)
        row.pack(fill="x")
        self.i18n.bind(ttk.Label(row), "asr_model").pack(side="left")
        self.asr_model_var = tk.StringVar(value="base")
        ttk.Combobox(row, textvariable=self.asr_model_var, state="readonly", width=8,
                     values=["tiny", "base", "small"]).pack(side="left", padx=6)
        self.i18n.bind(ttk.Label(row), "asr_lang").pack(side="left", padx=(12, 4))
        self.asr_lang_var = tk.StringVar(value="es")
        ttk.Combobox(row, textvariable=self.asr_lang_var, state="readonly", width=5,
                     values=["es", "en"]).pack(side="left")
        self.i18n.bind(ttk.Label(row), "seconds").pack(side="left", padx=(12, 4))
        self.rec_secs = tk.DoubleVar(value=3.0)
        ttk.Spinbox(row, from_=1, to=15, increment=0.5, textvariable=self.rec_secs, width=6).pack(side="left")
        self.i18n.bind(ttk.Button(row, command=self.record_and_process), "record").pack(side="left", padx=8)
        self.i18n.bind(ttk.Button(row, command=self.open_audio_and_process), "open_audio").pack(side="left")
        self.voice_out = scrolledtext.ScrolledText(f, height=24, font=("Consolas", 9))
        self.voice_out.pack(fill="both", expand=True, pady=8)
        self._recognizer = None

    def _get_recognizer(self, model: str, lang: str):
        from ..asr.recognizers import FasterWhisperRecognizer
        key = (model, lang)
        if self._recognizer is None or self._recognizer[0] != key:
            self._recognizer = (key, FasterWhisperRecognizer(model=model, language=lang))
        return self._recognizer[1]

    def _process_audio(self, audio_fn):
        """Tk variables are read here, on the UI thread; the worker only gets plain values."""
        if not self._need_model():
            return
        from ..asr.recognizers import ASRUnavailable
        rec = self._get_recognizer(self.asr_model_var.get(), self.asr_lang_var.get())
        user = self.user_var.get() or "default"
        missing = self.t("asr_missing", err="{err}")

        def work():
            try:
                rec.load()
            except ASRUnavailable as e:
                raise RuntimeError(missing.format(err=e)) from e
            tr = rec.transcribe(audio_fn())
            return tr, self.model.process_text(tr.text, user=user, asr_uncertainty=tr.uncertainty)

        def done(res):
            tr, r = res
            self.last_result = r
            self._show_result(r)
            self.voice_out.insert("end", f"{self.t('transcript')}: {tr.text!r}\n{self.t('asr_unc')}: "
                                         f"{tr.uncertainty:.3f}\n{self.t('f_intent')}: {r.intent}  ·  "
                                         f"{self.t('f_cal')}: {r.calibrated_confidence:.3f}  ·  "
                                         f"{self.t('f_decision')}: {self.t(r.decision.upper())}\n\n")
        self._run_bg(work, done)

    def record_and_process(self):
        seconds = float(self.rec_secs.get())

        def capture():
            import numpy as np
            import sounddevice as sd
            sr = 16000
            x = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            from ..audio.processing import preprocess
            return preprocess(np.asarray(x).T, sr)
        self._process_audio(capture)

    def open_audio_and_process(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.flac")])
        if path:
            from ..audio.processing import load_audio, preprocess
            self._process_audio(lambda: preprocess(*load_audio(path)))

    # ------------------------------------------------------- adaptation tab
    def _tab_adapt(self, f):
        box = self.i18n.bind(ttk.LabelFrame(f, padding=6), "corrections")
        box.pack(fill="both", expand=True)
        cols = ("user", "text", "wrong", "correct", "unc")
        self.corr_tree = ttk.Treeview(box, columns=cols, show="headings")
        for c, w in zip(cols, (90, 420, 140, 140, 100)):
            self.corr_tree.column(c, width=w)
        self.corr_tree.pack(fill="both", expand=True)
        bar = ttk.Frame(f)
        bar.pack(fill="x", pady=6)
        self.i18n.bind(ttk.Button(bar, command=self.export_corrections), "export_csv").pack(side="right")
        self.i18n.bind(ttk.Button(bar, command=self.clear_corrections), "clear_corrections").pack(side="right",
                                                                                                padx=6)

    def _refresh_corrections(self):
        for c, k in zip(("user", "text", "wrong", "correct", "unc"),
                        ("col_user", "col_text", "col_wrong", "col_correct", "col_unc")):
            self.corr_tree.heading(c, text=self.t(k))
        self.corr_tree.delete(*self.corr_tree.get_children())
        if self.model is None:
            return
        for r in self.model.ucil.memory.records:
            self.corr_tree.insert("", "end", values=(r.user, r.text, r.wrong_intent or "", r.correct_intent,
                                                     "" if r.uncertainty is None else f"{r.uncertainty:.3f}"))

    def clear_corrections(self, ask: bool = True):
        if not self._need_model():
            return
        if ask and not messagebox.askokcancel(SOFTWARE_NAME, self.t("clear_confirm"), icon="warning"):
            return
        self.model.clear_corrections()
        self.model.save_corrections(self._corrections_path())
        self._refresh_corrections()

    def export_corrections(self):
        if not self._need_model():
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            recs = [r.as_dict() for r in self.model.ucil.memory.records]
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["user", "text", "wrong_intent", "correct_intent", "timestamp",
                                                   "uncertainty", "model_version"])
                w.writeheader()
                w.writerows(recs)

    # ------------------------------------------------------ experiments tab
    def _tab_exp(self, f):
        row = ttk.Frame(f)
        row.pack(fill="x")
        self.i18n.bind(ttk.Label(row), "config").pack(side="left")
        self.cfg_var = tk.StringVar()
        self.cfg_combo = ttk.Combobox(row, textvariable=self.cfg_var, state="readonly", width=30,
                                      values=self._config_files())
        self.cfg_combo.pack(side="left", padx=6)
        self.cfg_combo.bind("<<ComboboxSelected>>", lambda e: self._load_config_seeds())
        self.i18n.bind(ttk.Label(row), "seeds").pack(side="left", padx=(12, 4))
        self.seeds_var = tk.StringVar(value="")
        self.seeds_var.trace_add("write", lambda *a: self._refresh_runs())
        ttk.Entry(row, textvariable=self.seeds_var, width=34).pack(side="left")
        self.runs_label = ttk.Label(row, foreground="#52514e")
        self.runs_label.pack(side="left", padx=6)
        self.i18n.bind(ttk.Button(row, command=self.run_experiment), "run").pack(side="left", padx=8)
        self.i18n.bind(ttk.Label(f, foreground="#52514e", wraplength=1050, justify="left"), "exp_note").pack(
            fill="x", pady=(6, 0))
        if self.cfg_combo["values"]:
            self.cfg_combo.current(0)
            self._load_config_seeds()
        self.exp_log = scrolledtext.ScrolledText(f, height=24, font=("Consolas", 9))
        self.exp_log.pack(fill="both", expand=True, pady=8)

    def _config_dir(self) -> Path:
        local = self.base / "configs"
        return local if local.exists() else config_dir()

    def _config_files(self) -> list[str]:
        d = self._config_dir()
        return sorted(p.name for p in d.glob("*.yaml")) if d.exists() else []

    def _seeds(self) -> list[int]:
        try:
            return [int(x) for x in self.seeds_var.get().split()]
        except ValueError:
            return []

    def _refresh_runs(self):
        self.runs_label.configure(text=self.t("n_runs", n=len(self._seeds())))

    def _load_config_seeds(self):
        from ..experiments.runner import load_config
        name = self.cfg_var.get()
        if name:
            seeds = load_config(self._config_dir() / name)["experiment"].get("seeds", [])
            self.seeds_var.set(" ".join(str(x) for x in seeds))

    def run_experiment(self, config: str | None = None, seeds: str | None = None):
        from ..experiments.runner import load_config, run
        if config:
            self.cfg_var.set(config)
            self._load_config_seeds()
        if seeds:
            self.seeds_var.set(seeds)
        name = self.cfg_var.get()
        if not name:
            return
        cfg = load_config(self._config_dir() / name)
        chosen = self._seeds()
        if not chosen:
            raise ValueError(self.t("seeds"))
        cfg["experiment"]["seeds"] = chosen
        log = lambda msg: self.q.put(("log", lambda m: (self.exp_log.insert("end", m + "\n"),
                                                        self.exp_log.see("end")), msg))
        log(f"{name}  seeds={cfg['experiment']['seeds']}  ({self.t('n_runs', n=len(chosen))})")
        if len(chosen) < 2:
            log(self.t("one_seed"))

        def done(rec):
            self.exp_log.insert("end", self.t("exp_done", id=rec.id) + "\n")
            self.events.add("experiment")
            self.refresh_results(select=rec.id)
        self._run_bg(lambda: run(cfg, self.base / "results", progress=log), done)

    # ---------------------------------------------------------- results tab
    def _tab_results(self, f):
        row = ttk.Frame(f)
        row.pack(fill="x")
        self.i18n.bind(ttk.Label(row), "experiment").pack(side="left")
        self.res_var = tk.StringVar()
        self.res_combo = ttk.Combobox(row, textvariable=self.res_var, state="readonly", width=22)
        self.res_combo.pack(side="left", padx=6)
        self.res_combo.bind("<<ComboboxSelected>>", lambda e: self._load_experiment())
        self.i18n.bind(ttk.Label(row), "table").pack(side="left", padx=(12, 4))
        self.table_var = tk.StringVar()
        self.table_combo = ttk.Combobox(row, textvariable=self.table_var, state="readonly", width=34)
        self.table_combo.pack(side="left")
        self.table_combo.bind("<<ComboboxSelected>>", lambda e: self._show_table())
        self.i18n.bind(ttk.Label(row), "figure").pack(side="left", padx=(12, 4))
        self.fig_var = tk.StringVar()
        self.fig_combo = ttk.Combobox(row, textvariable=self.fig_var, state="readonly", width=34)
        self.fig_combo.pack(side="left")
        self.fig_combo.bind("<<ComboboxSelected>>", lambda e: self._show_figure())
        row2 = ttk.Frame(f)
        row2.pack(fill="x", pady=4)
        self.i18n.bind(ttk.Button(row2, command=self.refresh_results), "refresh").pack(side="left")
        self.i18n.bind(ttk.Button(row2, command=self.open_results_folder), "open_folder").pack(side="left", padx=6)
        self.i18n.bind(ttk.Button(row2, command=self.export_zip), "export_zip").pack(side="left")
        pan = ttk.PanedWindow(f, orient="vertical")
        pan.pack(fill="both", expand=True)
        tf = ttk.Frame(pan)
        self.res_tree = ttk.Treeview(tf, show="headings", height=9)
        xs = ttk.Scrollbar(tf, orient="horizontal", command=self.res_tree.xview)
        self.res_tree.configure(xscrollcommand=xs.set)
        self.res_tree.pack(fill="both", expand=True)
        xs.pack(fill="x")
        pan.add(tf, weight=1)
        self.fig_label = ttk.Label(pan, anchor="center")
        pan.add(self.fig_label, weight=2)
        self._fig_image = None
        self.refresh_results()

    def _results_dir(self) -> Path:
        return self.base / "results"

    def refresh_results(self, select: str | None = None):
        d = self._results_dir()
        exps = sorted((p.name for p in d.glob("EXP-*") if p.is_dir()), reverse=True) if d.exists() else []
        self.res_combo["values"] = exps
        if exps:
            self.res_var.set(select if select in exps else exps[0])
            self._load_experiment()

    def _current_exp(self) -> Path | None:
        return self._results_dir() / self.res_var.get() if self.res_var.get() else None

    def _load_experiment(self):
        d = self._current_exp()
        if d is None:
            return
        tables = sorted(p.stem for p in d.glob("summary_*.csv")) + sorted(
            p.stem for p in d.glob("*.csv") if not p.stem.startswith("summary_"))
        figs = sorted(p.stem for p in (d / "figures").glob("*.png") if not p.stem.endswith("_es"))
        self.table_combo["values"] = tables
        self.fig_combo["values"] = figs
        if tables:
            self.table_var.set(tables[0])
            self._show_table()
        if figs:
            pref = [f for f in figs if "tradeoff" in f]
            self.fig_var.set(pref[0] if pref else figs[0])
            self._show_figure()
        self.events.add("results")

    def show_result_item(self, table: str | None = None, figure: str | None = None):
        if table and table in self.table_combo["values"]:
            self.table_var.set(table)
            self._show_table()
        if figure and figure in self.fig_combo["values"]:
            self.fig_var.set(figure)
            self._show_figure()

    def _show_table(self):
        path = self._current_exp() / f"{self.table_var.get()}.csv"
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            return
        cols = rows[0]
        lang = self.i18n.lang
        self.res_tree.configure(columns=cols)
        def term(v):
            return TERMS_ES.get(v, v) if lang == "es" else v

        def head(c):
            return COLUMNS.get(c, {}).get(lang) or term(c)

        na = {j for j, c in enumerate(cols) if c in ("sd", "ci95_low", "ci95_high")}
        body = [[self.t("not_estimable") if j in na and v == "" else term(_fmt(v)) for j, v in enumerate(r)]
                for r in rows[1:500]]
        for j, c in enumerate(cols):
            width = max([len(head(c))] + [len(r[j]) for r in body[:60] if j < len(r)])
            self.res_tree.heading(c, text=head(c))
            self.res_tree.column(c, width=min(300, 8 * width + 16), stretch=False)
        self.res_tree.delete(*self.res_tree.get_children())
        for r in body:
            self.res_tree.insert("", "end", values=r)

    def _show_figure(self):
        base = self._current_exp() / "figures" / self.fig_var.get()
        path = base.with_name(base.name + "_es.png") if self.i18n.lang == "es" else base.with_suffix(".png")
        if not path.exists():
            path = base.with_suffix(".png")
        if not path.exists():
            return
        img = tk.PhotoImage(file=str(path))
        factor = max(1, int(max(img.width() / 1100, img.height() / 430) + 0.999))
        self._fig_image = img.subsample(factor, factor) if factor > 1 else img
        self.fig_label.configure(image=self._fig_image)

    def open_results_folder(self):
        d = self._current_exp() or self._results_dir()
        if d.exists() and sys.platform == "win32":
            os.startfile(str(d))  # noqa: S606 - opens the user's own results folder

    def export_zip(self):
        d = self._current_exp()
        if d is None or not d.exists():
            messagebox.showinfo(SOFTWARE_NAME, self.t("no_results"))
            return
        path = filedialog.asksaveasfilename(defaultextension=".zip", initialfile=f"{d.name}.zip",
                                            filetypes=[("ZIP", "*.zip")])
        if path:
            shutil.make_archive(str(Path(path).with_suffix("")), "zip", d)

    # --------------------------------------------------------- settings tab
    def _tab_settings(self, f):
        box = self.i18n.bind(ttk.LabelFrame(f, padding=8), "costs")
        box.pack(fill="x")
        self.param_vars = {}
        c = self.settings["costs"]
        entries = [("p_cerr_low", c["error"]["low"]), ("p_cerr_medium", c["error"]["medium"]),
                   ("p_cerr_high", c["error"]["high"]), ("p_confirm", c["confirm"]), ("p_reject", c["reject"])]
        for i, (k, v) in enumerate(entries):
            self.i18n.bind(ttk.Label(box), k).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.DoubleVar(value=v)
            ttk.Spinbox(box, from_=0.0, to=1000.0, increment=0.1, textvariable=var, width=10,
                        command=self._refresh_thresholds).grid(row=i, column=1, padx=8)
            self.param_vars[k] = var
        self.thr_label = ttk.Label(box, foreground="#2a78d6")
        self.thr_label.grid(row=len(entries), column=0, columnspan=3, sticky="w", pady=6)
        mb = self.i18n.bind(ttk.LabelFrame(f, padding=8), "model_params")
        mb.pack(fill="x", pady=8)
        rows = [("p_seed", "seed", None), ("p_predictor", "predictor",
                                           ["tfidf_logreg", "embedding_softmax", "prototype", "ensemble"]),
                ("p_calibrator", "calibrator", ["auto", "fusion", "product_temperature", "product_isotonic"]),
                ("p_ood", "ood_detector", ["fused", "auto", "msp", "energy", "knn", "prototype", "mahalanobis",
                                           "lexical"]),
                ("p_mem", "memory_threshold", None)]
        for i, (label, key, choices) in enumerate(rows):
            self.i18n.bind(ttk.Label(mb), label).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(self.settings[key]))
            w = (ttk.Combobox(mb, textvariable=var, state="readonly", values=choices, width=22) if choices
                 else ttk.Entry(mb, textvariable=var, width=10))
            w.grid(row=i, column=1, sticky="w", padx=8)
            self.param_vars[key] = var
        self.context_var = tk.BooleanVar(value=self.settings["use_context"])
        self.i18n.bind(ttk.Checkbutton(mb, variable=self.context_var), "p_context").grid(
            row=len(rows), column=0, columnspan=2, sticky="w", pady=2)
        eb = self.i18n.bind(ttk.LabelFrame(f, padding=8), "exec_mode")
        eb.pack(fill="x")
        self.i18n.bind(ttk.Label(eb, wraplength=1000, justify="left"), "mode_sandbox").pack(anchor="w")
        self.mandatory_var = tk.BooleanVar(value=self.settings["mandatory"])
        self.i18n.bind(ttk.Checkbutton(eb, variable=self.mandatory_var), "mandatory").pack(anchor="w")
        self.i18n.bind(ttk.Button(f, command=self.apply_settings), "apply_retrain").pack(anchor="e", pady=8)
        self._refresh_thresholds()

    def _cost_model(self) -> CostModel:
        v = self.param_vars
        return CostModel(error={"low": float(v["p_cerr_low"].get()), "medium": float(v["p_cerr_medium"].get()),
                                "high": float(v["p_cerr_high"].get())},
                         confirm=float(v["p_confirm"].get()), reject=float(v["p_reject"].get()))

    def _refresh_thresholds(self):
        try:
            c = self._cost_model()
            self.thr_label.configure(text=self.t("thresholds", **{r: c.execution_threshold(r)
                                                                  for r in ("low", "medium", "high")}))
        except (tk.TclError, ValueError):
            pass

    def apply_settings(self):
        c = self._cost_model()
        if min(c.error.values()) <= 0 or c.confirm < 0 or c.reject < 0:
            raise ValueError("costs must be positive")
        v = self.param_vars
        self.settings.update({"seed": int(v["seed"].get()), "predictor": v["predictor"].get(),
                              "calibrator": v["calibrator"].get(), "ood_detector": v["ood_detector"].get(),
                              "memory_threshold": float(v["memory_threshold"].get()),
                              "use_context": bool(self.context_var.get()),
                              "mandatory": bool(self.mandatory_var.get()),
                              "costs": {"error": dict(c.error), "confirm": c.confirm, "reject": c.reject}})
        self.train_model()

    # ------------------------------------------------------------- tutorial
    def open_tutorial(self):
        Tutorial(self)


def _fmt(v: str) -> str:
    try:
        x = float(v)
        return f"{x:.4f}" if not x.is_integer() else str(int(x))
    except ValueError:
        return v


class Tutorial:
    """Step-by-step guide that can also perform each step on the live interface."""

    def __init__(self, app: VoxControlApp):
        self.app = app
        self.i = 0
        self.win = tk.Toplevel(app.root)
        self.win.title(app.t("tut_title"))
        root = app.root
        self.win.geometry("460x320")
        self.win.after(30, self._place)
        self.win.attributes("-topmost", True)
        self.step_lbl = ttk.Label(self.win, foreground="#52514e", padding=(12, 8, 12, 0))
        self.step_lbl.pack(fill="x")
        self.title = ttk.Label(self.win, font=("Segoe UI", 12, "bold"), padding=(12, 2))
        self.title.pack(fill="x")
        self.body = ttk.Label(self.win, wraplength=430, justify="left", padding=12, anchor="nw")
        self.body.pack(fill="both", expand=True)
        self.state = ttk.Label(self.win, padding=(12, 0))
        self.state.pack(fill="x")
        bar = ttk.Frame(self.win, padding=10)
        bar.pack(fill="x")
        self.back_btn = ttk.Button(bar, command=lambda: self.go(-1))
        self.back_btn.pack(side="left")
        self.do_btn = ttk.Button(bar, command=self.do_it)
        self.do_btn.pack(side="left", padx=6)
        self.close_btn = ttk.Button(bar, command=self.win.destroy)
        self.close_btn.pack(side="right")
        self.next_btn = ttk.Button(bar, command=lambda: self.go(1))
        self.next_btn.pack(side="right", padx=6)
        app.i18n.listeners.append(self.render)
        self.render()
        self._tick()

    def _place(self):
        """Dock the guide in the bottom-right corner of the main window (geometry known once mapped)."""
        import re
        m = re.match(r"(\d+)x(\d+)([+-]\d+)([+-]\d+)", self.app.root.geometry())
        if not m or not self.win.winfo_exists():
            return
        w, h, x, y = (int(g) for g in m.groups())
        self.win.geometry(f"460x320+{max(x + w - 470, 0)}+{max(y + h - 400, 0)}")

    def step(self) -> dict:
        return TUTORIAL[self.i]

    def go(self, d: int):
        self.i = min(max(self.i + d, 0), len(TUTORIAL) - 1)
        self.render()

    def render(self):
        if not self.win.winfo_exists():
            return
        a, s, lang = self.app, self.step(), self.app.i18n.lang
        self.win.title(a.t("tut_title"))
        self.step_lbl.configure(text=a.t("tut_step", i=self.i + 1, n=len(TUTORIAL)))
        self.title.configure(text=s["title"][lang])
        self.body.configure(text=s["body"][lang])
        self.back_btn.configure(text=a.t("tut_back"))
        self.next_btn.configure(text=a.t("tut_next"))
        self.close_btn.configure(text=a.t("tut_close"))
        self.do_btn.configure(text=a.t("tut_do"), state="normal" if s["action"] else "disabled")
        a.show_tab(s["tab"])
        self._update_state()

    def done(self) -> bool:
        chk = self.step()["check"]
        return chk is None or chk in self.app.events

    def _update_state(self):
        if self.step()["check"] is None:
            self.state.configure(text="")
        else:
            ok = self.done()
            self.state.configure(text=self.app.t("tut_done" if ok else "tut_pending"),
                                 foreground="#1baf7a" if ok else "#52514e")

    def _tick(self):
        if self.win.winfo_exists():
            self._update_state()
            self.win.after(400, self._tick)

    def do_it(self):
        a, act = self.app, self.step()["action"]
        if act == "train":
            a.train_model()
        elif act == "results":
            a.show_tab("results")
            a.refresh_results()
            a.show_result_item("summary_decision_metrics", "fig_tradeoff_ier_vs_intervention")
        elif act[0] == "analyse":
            a.show_tab("text")
            a.analyse(act[1])
        elif act[0] == "example":
            a.show_tab("home")
            a.run_selected_example(act[1])
        elif act[0] == "correct":
            a.show_tab("text")
            if a.analyse(act[1]) is not None:
                a.apply_correction(act[2])
                a.analyse(act[1])
        elif act[0] == "experiment":
            a.show_tab("exp")
            a.run_experiment(act[1], act[2])


def main(lang: str = "es") -> None:
    from ..paths import configure_caches
    configure_caches()
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista" if sys.platform == "win32" else "clam")
    except tk.TclError:
        pass
    VoxControlApp(root, lang)
    root.mainloop()


if __name__ == "__main__":
    main()
