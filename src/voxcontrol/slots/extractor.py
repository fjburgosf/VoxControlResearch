"""Slot extraction and validation.

Primary strategy: the intent's registry templates are compiled into regular
expressions whose placeholders become capture groups ("template grammar").
The set of usable templates can be restricted (e.g. to training templates) so
that slot accuracy is not measured on phrasings the system was built from.
Fallback: generic patterns (numbers, file names, gazetteer look-up).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..intents.registry import IntentRegistry
from ..text import normalize

_FILE_RE = re.compile(r"\b[\w\-]+\.(?:docx|txt|png|xlsx|csv|zip|pptx|pdf|log)\b")
_NUM_RE = re.compile(r"\b(\d{1,3})\b")
_PRONOUN_HINTS = ("lo", "la", "eso", "esto", "esa", "ese", "it", "this", "that")
_SLOT_RANGES = {"value": (0, 100)}


@dataclass
class SlotResult:
    values: dict = field(default_factory=dict)
    status: str = "ok"                  # ok | ambiguous | missing | invalid
    candidates: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"values": self.values, "status": self.status, "candidates": self.candidates, "errors": self.errors}


class SlotExtractor:
    def __init__(self, registry: IntentRegistry, allowed_templates: set[str] | None = None):
        self.registry = registry
        self.patterns: dict[str, list[re.Pattern]] = {}
        for name in registry.names:
            spec = registry.spec(name)
            pats = []
            for lang, levels in spec.templates.items():
                for lvl, tpls in levels.items():
                    for i, tpl in enumerate(tpls):
                        if allowed_templates is not None and f"{name}/{lang}/{lvl}/{i}" not in allowed_templates:
                            continue
                        if "{" in tpl:
                            pats.append(self._compile(tpl))
            self.patterns[name] = pats
        self.surface: list[tuple[str, str]] = sorted(
            ((normalize(f), app) for app, d in registry.apps.items()
             for lang in ("es", "en") for f in d.get(lang, [])),
            key=lambda x: -len(x[0]))
        self.ambiguous: list[tuple[str, list[str]]] = sorted(
            ((normalize(a), apps) for lang in registry.ambiguous_app_aliases.values() for a, apps in lang.items()),
            key=lambda x: -len(x[0]))

    @staticmethod
    def _compile(template: str) -> re.Pattern:
        parts = re.split(r"(\{\w+\})", template)
        rx = ""
        for p in parts:
            m = re.fullmatch(r"\{(\w+)\}", p)
            if m:
                rx += f"(?P<{m.group(1)}>.+?)"
            elif p:
                rx += re.escape(normalize(p) if p.strip() else " ").replace(r"\ ", r"\s*")
        return re.compile(rf"(?:^|\s){rx}(?:\s|$)")

    def resolve_app(self, phrase: str) -> tuple[str | None, list[str]]:
        n = f" {normalize(phrase)} "
        for form, app in self.surface:
            if f" {form} " in n:
                return app, [app]
        for alias, apps in self.ambiguous:
            if f" {alias} " in n:
                return None, list(apps)
        return None, []

    def extract(self, text: str, intent: str, context: dict | None = None) -> SlotResult:
        spec = self.registry.spec(intent)
        res = SlotResult()
        if not spec.slots:
            return res
        nt = normalize(text)
        captured: dict[str, str] = {}
        for pat in self.patterns.get(intent, []):
            m = pat.search(nt)
            if m:
                captured = {k: v.strip() for k, v in m.groupdict().items()}
                break
        for slot in spec.slots:
            if slot == "app":
                app, cands = self.resolve_app(captured.get("app", nt))
                if app is None and not cands:
                    app, cands = self.resolve_app(nt)
                if app:
                    res.values["app"] = app
                elif len(cands) > 1:
                    res.candidates["app"] = cands
                elif context and context.get("active_app") and any(f" {h} " in f" {nt} " or nt.endswith(h)
                                                                    for h in _PRONOUN_HINTS):
                    res.values["app"] = context["active_app"]
            elif slot == "value":
                m = _NUM_RE.search(captured.get("value", nt))
                if m:
                    res.values["value"] = int(m.group(1))
            elif slot == "file":
                m = _FILE_RE.search(nt)
                if m:
                    res.values["file"] = m.group(0)
                elif captured.get("file"):
                    res.values["file"] = captured["file"]
            elif captured.get(slot):
                res.values[slot] = captured[slot]
        self._validate(spec.slots, res)
        return res

    @staticmethod
    def _validate(required: tuple[str, ...], res: SlotResult) -> None:
        for slot, (lo, hi) in _SLOT_RANGES.items():
            if slot in res.values and not lo <= res.values[slot] <= hi:
                res.errors.append(f"{slot}={res.values[slot]} outside [{lo}, {hi}]")
        missing = [s for s in required if s not in res.values]
        if res.errors:
            res.status = "invalid"
        elif any(s in res.candidates for s in missing):
            res.status = "ambiguous"
        elif missing:
            res.status = "missing"
            res.errors.append("missing: " + ", ".join(missing))
