"""Extensible intent registry loaded from YAML (no intent is hard-coded in the models)."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

RESOURCES = Path(__file__).resolve().parents[1] / "resources"
RISK_LEVELS = ("low", "medium", "high")


def load_resource(name: str) -> dict:
    with open(RESOURCES / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class IntentSpec:
    name: str
    risk: str
    slots: tuple[str, ...]
    description: dict
    keywords: dict
    templates: dict = field(repr=False)

    def describe(self, lang: str) -> str:
        return self.description.get(lang) or self.description.get("en") or self.name


class IntentRegistry:
    def __init__(self, data: dict):
        self.raw = data
        self.specs: dict[str, IntentSpec] = {}
        for name, d in data["intents"].items():
            risk = d.get("risk", "low")
            if risk not in RISK_LEVELS:
                raise ValueError(f"intent {name}: unknown risk level {risk!r}")
            templates = {lang: {int(k): list(v) for k, v in levels.items()}
                         for lang, levels in d.get("templates", {}).items()}
            self.specs[name] = IntentSpec(
                name=name, risk=risk, slots=tuple(d.get("slots", [])),
                description=d.get("description", {}), keywords=d.get("keywords", {}),
                templates=templates,
            )
        self.names: list[str] = sorted(self.specs)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.apps: dict = data.get("apps", {})
        self.ambiguous_app_aliases: dict = data.get("ambiguous_app_aliases", {})
        self.fillers: dict = data.get("fillers", {})
        self.wrappers: dict = data.get("wrappers", {})

    @classmethod
    def default(cls) -> "IntentRegistry":
        return _default_registry()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IntentRegistry":
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self.specs

    def spec(self, name: str) -> IntentSpec:
        return self.specs[name]

    def risk(self, name: str) -> str:
        return self.specs[name].risk

    def languages(self) -> list[str]:
        langs = set()
        for s in self.specs.values():
            langs.update(s.templates)
        return sorted(langs)

    def app_category(self, app_id: str) -> str | None:
        a = self.apps.get(app_id)
        return a.get("category") if a else None


@lru_cache(maxsize=1)
def _default_registry() -> IntentRegistry:
    return IntentRegistry(load_resource("intents.yaml"))
