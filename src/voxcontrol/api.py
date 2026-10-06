"""High-level Python API.

    from voxcontrol import VoxModel
    model = VoxModel.train(seed=0)                 # or VoxModel.load("model.pkl")
    result = model.process_text("abre el navegador")
    result.intent, result.slots, result.confidence, result.uncertainty, result.ood_score, result.decision
    model.correct(result, correct_intent="open_app")
"""
from __future__ import annotations

from pathlib import Path

from .datasets.synthetic import generate
from .decision.policies import CONFIRM, EXECUTE
from .sandbox.desktop import Sandbox
from .ucil import UCIL, Result, UCILConfig

DEFAULT_MANDATORY_CONFIRM = ("high",)


class VoxModel:
    def __init__(self, ucil: UCIL, execution: str = "sandbox",
                 mandatory_confirm_risk: tuple[str, ...] = DEFAULT_MANDATORY_CONFIRM):
        self.ucil = ucil
        self.execution = execution
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
        return cls(UCIL.load(path), **kwargs)

    def save(self, path: str | Path) -> None:
        self.ucil.save(path)

    # ------------------------------------------------------------ inference
    def process_text(self, text: str, user: str = "default", context: dict | None = None,
                     asr_uncertainty: float = 0.0) -> Result:
        context = context if context is not None else self.sandbox.context()
        result = self.ucil.process_text(text, context=context, user=user, asr_uncertainty=asr_uncertainty)
        if self.execution == "real" and result.decision == EXECUTE and result.risk in self.mandatory_confirm_risk:
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

    def execute(self, result: Result, confirmed: bool = False):
        """Run the decided action in the sandbox. CONFIRM decisions need ``confirmed=True``."""
        if result.decision == EXECUTE or (result.decision == CONFIRM and confirmed):
            return self.sandbox.execute(result.intent, result.slots)
        return None

    # ----------------------------------------------------------- adaptation
    def correct(self, sample: Result | str, correct_intent: str, user: str = "default") -> None:
        if isinstance(sample, Result):
            self.ucil.correct(sample.text, correct_intent, user=user, wrong_intent=sample.intent,
                              uncertainty=1.0 - sample.calibrated_confidence)
        else:
            self.ucil.correct(sample, correct_intent, user=user)
