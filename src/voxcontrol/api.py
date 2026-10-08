"""High-level Python API.

    from voxcontrol import VoxModel
    model = VoxModel.train(seed=0)                 # or VoxModel.load("model.pkl")
    result = model.process_text("abre el navegador")
    result.intent, result.slots, result.confidence, result.uncertainty, result.ood_score, result.decision
    model.execute(result, confirmed=True, choice=None)   # always on the simulated desktop
    model.correct(result, correct_intent="open_app")

Actions only ever run on the simulated desktop (``Sandbox``); there is no operating-system actuator.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .adaptation.memory import CorrectionRecord
from .datasets.synthetic import generate
from .decision.policies import CONFIRM, EXECUTE, REJECT
from .sandbox.desktop import ActionResult, Sandbox
from .ucil import UCIL, Result, UCILConfig

DEFAULT_MANDATORY_CONFIRM = ("high",)


class VoxModel:
    def __init__(self, ucil: UCIL, mandatory_confirm_risk: tuple[str, ...] = DEFAULT_MANDATORY_CONFIRM):
        self.ucil = ucil
        self.mandatory_confirm_risk = tuple(mandatory_confirm_risk)
        self.sandbox = Sandbox(ucil.registry)
        self.recognizer = None

    # ------------------------------------------------------------- creation
    @classmethod
    def train(cls, seed: int = 0, config: dict | None = None, **kwargs) -> "VoxModel":
        cfg = dict(config or {})
        data = generate(seed, **cfg.get("dataset", {}))
        ucfg = UCILConfig.from_dict({**cfg.get("ucil", {}), "seed": seed, "costs": cfg.get("costs")})
        return cls(UCIL(config=ucfg).fit(data.train, data.cal, data.ood_cal), **kwargs)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "VoxModel":
        """Only load model files you created yourself: they are pickle files and can run code."""
        return cls(UCIL.load(path), **kwargs)

    def save(self, path: str | Path) -> None:
        self.ucil.save(path)

    # ------------------------------------------------------------ inference
    def process_text(self, text: str, user: str = "default", context: dict | None = None,
                     asr_uncertainty: float = 0.0) -> Result:
        context = context if context is not None else self.sandbox.context()
        result = self.ucil.process_text(text, context=context, user=user, asr_uncertainty=asr_uncertainty)
        if result.decision == EXECUTE and result.risk in self.mandatory_confirm_risk:
            result.decision = CONFIRM
            result.explanation["mandatory_confirmation"] = True
        return result

    def process_audio(self, path: str | Path, user: str = "default", context: dict | None = None) -> Result:
        if self.recognizer is None:
            from .asr.recognizers import FasterWhisperRecognizer
            self.recognizer = FasterWhisperRecognizer()
        tr = self.recognizer.transcribe_file(path)
        result = self.process_text(tr.text, user=user, context=context, asr_uncertainty=tr.uncertainty)
        result.explanation["transcript"] = tr.text
        result.explanation["asr_confidence"] = tr.confidence
        return result

    # ------------------------------------------------------------ execution
    def resolve(self, result: Result, choice: str | None = None) -> tuple[str, dict, str | None, dict]:
        """Intent and slots that a confirmation would run.

        ``choice`` is one of ``result.options`` ("intent" or "intent:app"). Returns
        ``(intent, slots, problem_key, params)``; ``problem_key`` is None when the action is fully specified.
        """
        intent, slots = result.intent, dict(result.slots)
        if choice is None:
            if len(result.options) > 1:
                return intent, slots, "needs_choice", {"options": ", ".join(result.options)}
            status = result.slot_status
        else:
            if choice not in result.options:
                raise ValueError(f"{choice!r} is not one of the options {result.options}")
            name, _, app = choice.partition(":")
            if name != intent:
                intent = name
                slots = dict(self.ucil.slots.extract(result.text, intent, self.sandbox.context()).values)
            if app:
                slots["app"] = app
            status = "ok"
        required = self.ucil.registry.spec(intent).slots
        missing = [s for s in required if s not in slots]
        if status != "ok" or missing:
            return intent, slots, "unresolved_slot", {"slots": ", ".join(missing) or result.slot_status}
        return intent, slots, None, {}

    def execute(self, result: Result, confirmed: bool = False, choice: str | None = None) -> ActionResult:
        """Run the decided action on the simulated desktop.

        EXECUTE runs directly. CONFIRM needs ``confirmed=True`` and, when there are several options, the
        user's ``choice``; an action with ambiguous, missing or invalid slots never runs. REJECT never runs.
        A refused action leaves the desktop unchanged and returns ``success=False``.
        """
        if result.decision == REJECT:
            return self.sandbox.refuse(result.intent, result.slots, "rejected")
        if result.decision == CONFIRM and not confirmed:
            return self.sandbox.refuse(result.intent, result.slots, "needs_confirmation")
        intent, slots, problem, params = self.resolve(result, choice)
        if problem:
            return self.sandbox.refuse(intent, slots, problem, **params)
        return self.sandbox.execute(intent, slots)

    # ----------------------------------------------------------- adaptation
    def correct(self, sample: Result | str, correct_intent: str, user: str = "default") -> None:
        if isinstance(sample, Result):
            self.ucil.correct(sample.text, correct_intent, user=user, wrong_intent=sample.intent,
                              uncertainty=1.0 - sample.calibrated_confidence)
        else:
            self.ucil.correct(sample, correct_intent, user=user)

    def save_corrections(self, path: str | Path) -> None:
        """Write every stored correction to a JSON file (atomic replace, never a half-written file)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps([r.as_dict() for r in self.ucil.memory.records], ensure_ascii=False, indent=1),
                       encoding="utf-8")
        os.replace(tmp, path)

    def load_corrections(self, path: str | Path) -> int:
        """Add the corrections stored in ``path`` that the model does not have yet; returns how many."""
        path = Path(path)
        if not path.exists():
            return 0
        records = [CorrectionRecord(**r) for r in json.loads(path.read_text(encoding="utf-8"))]
        return self.ucil.memory.restore(records)

    def clear_corrections(self) -> None:
        self.ucil.memory.clear()
