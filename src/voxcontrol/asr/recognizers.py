"""Speech recogniser interface and adapters.

    recognizer.transcribe(audio, sr) -> Transcript(text, confidence, uncertainty, alternatives)

``FasterWhisperRecognizer`` derives U_ASR from per-word probabilities and,
optionally, from disagreement between decodings at different temperatures.
``TextRecognizer`` passes text through (text mode separates ASR error from
intent error).
"""
from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..audio.processing import TARGET_SR, load_audio, preprocess
from ..uncertainty.measures import asr_uncertainty_from_agreement, asr_uncertainty_from_words


class ASRUnavailable(RuntimeError):
    pass


@dataclass
class Transcript:
    text: str
    confidence: float = 1.0
    uncertainty: float = 0.0
    alternatives: list = field(default_factory=list)
    words: list = field(default_factory=list)


class SpeechRecognizer:
    name = "base"

    def transcribe(self, audio: np.ndarray, sr: int = TARGET_SR) -> Transcript:
        raise NotImplementedError

    def transcribe_file(self, path: str | Path) -> Transcript:
        x, sr = load_audio(path)
        return self.transcribe(preprocess(x, sr), TARGET_SR)

    def get_confidence(self, t: Transcript) -> float:
        return t.confidence

    def get_alternatives(self, t: Transcript) -> list:
        return t.alternatives


class TextRecognizer(SpeechRecognizer):
    name = "text"

    def transcribe(self, audio, sr=TARGET_SR) -> Transcript:
        return Transcript(str(audio))


def _add_cuda_dll_dirs() -> None:
    """Make pip-installed NVIDIA libraries (cuBLAS/cuDNN wheels) visible on Windows."""
    if sys.platform != "win32":
        return
    for base in {os.path.dirname(os.path.dirname(np.__file__))}:
        for d in glob.glob(os.path.join(base, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


class FasterWhisperRecognizer(SpeechRecognizer):
    name = "faster_whisper"

    def __init__(self, model: str = "base", device: str = "auto", compute_type: str = "int8",
                 language: str | None = None, alternatives: int = 0):
        self.model_name, self.device, self.compute_type = model, device, compute_type
        self.language = language
        self.n_alternatives = alternatives
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ASRUnavailable("Speech recognition model not available: install 'faster-whisper'.") from e
        _add_cuda_dll_dirs()
        source = self._model_source()
        from ..paths import frozen
        device = "cpu" if (frozen() and self.device == "auto") else self.device   # CUDA libraries are not bundled
        attempts = [(device, self.compute_type)]
        if device in ("auto", "cuda"):
            attempts.append(("cpu", "int8"))
        last = None
        for dev, ct in attempts:
            try:
                self._model = WhisperModel(source, device=dev, compute_type=ct)
                self.active_device = dev
                return self._model
            except Exception as e:  # noqa: BLE001 - try the next configuration
                last = e
        raise ASRUnavailable(f"Speech recognition model not available ({self.model_name}): {last}")

    def _model_source(self) -> str:
        """Packaged application: download once into a short, flat folder inside the application
        (cache/models/whisper-<size>) so no path approaches the Windows 260-character limit."""
        from ..paths import frozen, writable_root
        if not frozen() or Path(self.model_name).exists():
            return self.model_name
        target = writable_root() / "cache" / "models" / f"whisper-{self.model_name}"
        if (target / "model.bin").exists():
            return str(target)
        try:
            from faster_whisper import download_model
            return download_model(self.model_name, output_dir=str(target))
        except Exception as e:  # noqa: BLE001 - reported to the user as a clear message
            raise ASRUnavailable(f"Speech recognition model not available ({self.model_name}): "
                                 f"download failed ({type(e).__name__}: {e})") from e

    def describe(self) -> dict:
        """Identity of the recogniser actually used: model files (SHA-256 of model.bin), device, versions."""
        import hashlib
        info = {"asr_engine": "faster-whisper", "asr_model": self.model_name,
                "asr_device": getattr(getattr(self._model, "model", None), "device", None)
                or getattr(self, "active_device", self.device), "asr_compute_type": self.compute_type}
        try:
            import ctranslate2
            import faster_whisper
            info.update(faster_whisper_version=faster_whisper.__version__, ctranslate2_version=ctranslate2.__version__)
        except Exception:  # noqa: BLE001 - versions are informative only
            pass
        folder = Path(self.model_name) if Path(self.model_name).exists() else None
        if folder is None:
            try:
                from ..paths import frozen, writable_root
                if frozen():
                    folder = writable_root() / "cache" / "models" / f"whisper-{self.model_name}"
                else:
                    from faster_whisper import download_model
                    folder = Path(download_model(self.model_name, local_files_only=True))
            except Exception:  # noqa: BLE001
                folder = None
        if folder is not None and (folder / "model.bin").exists():
            h = hashlib.sha256()
            with open(folder / "model.bin", "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            info["asr_model_sha256"] = h.hexdigest()
        return info

    def _decode(self, audio, temperature: float = 0.0):
        try:
            return self._decode_once(audio, temperature)
        except RuntimeError as e:
            # GPU selected but CUDA runtime libraries missing: fall back to the CPU once.
            if getattr(self, "active_device", "cpu") == "cpu" or not any(
                    k in str(e).lower() for k in ("cublas", "cudnn", "cuda")):
                raise
            self._model, self.device, self.compute_type = None, "cpu", "int8"
            return self._decode_once(audio, temperature)

    def _decode_once(self, audio, temperature: float):
        segs, _ = self.load().transcribe(audio, language=self.language, beam_size=1, temperature=temperature,
                                         word_timestamps=True, vad_filter=False, condition_on_previous_text=False)
        words, texts = [], []
        for s in segs:
            texts.append(s.text)
            words.extend((w.word, float(w.probability)) for w in (s.words or []))
        return " ".join(t.strip() for t in texts).strip(), words

    def transcribe(self, audio, sr=TARGET_SR) -> Transcript:
        audio = np.asarray(audio, dtype=np.float32)
        text, words = self._decode(audio)
        u_words = asr_uncertainty_from_words([p for _, p in words]) if words else 1.0
        alts = [text]
        for i in range(self.n_alternatives):
            alts.append(self._decode(audio, temperature=0.4 + 0.2 * i)[0])
        u = u_words if self.n_alternatives == 0 else 0.5 * (u_words + asr_uncertainty_from_agreement(alts))
        return Transcript(text=text, confidence=1.0 - u_words, uncertainty=u, alternatives=alts[1:], words=words)
