"""Reproducible synthetic intent corpus.

Splits are disjoint *by template*: a template used to generate training
utterances never generates calibration or test utterances, so test accuracy
measures paraphrase generalisation rather than memorisation. Difficulty
levels 2-3 (syntactic reformulation, implicit request) never appear in the
training split. OOD topics are split between calibration and test so that
OOD evaluation is topic-disjoint.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np

from ..intents.registry import IntentRegistry, load_resource
from ..text import normalize

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


@dataclass
class Sample:
    text: str
    intent: str | None            # None => out-of-distribution
    lang: str
    kind: str = "id"              # id | ood_near | ood_far | ambiguous_intent | ambiguous_slot | user
    level: int = -1
    template_id: str = ""
    slots: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    user: str | None = None
    noise: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def in_domain(self) -> bool:
        return self.intent is not None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GeneratorConfig:
    seed: int = 0
    languages: tuple[str, ...] = ("es", "en")
    variants_train: int = 8
    variants_cal: int = 5
    variants_test: int = 5
    easy_fractions: tuple[float, float, float] = (0.6, 0.15, 0.25)   # levels 0-1: train/cal/test
    hard_fractions: tuple[float, float] = (0.4, 0.6)                 # levels 2-3: cal/test
    ood_variants: int = 2
    user_variants: int = 6
    user_adapt_fraction: float = 0.5
    context_consistency: float = 0.5


@dataclass
class IntentDataset:
    config: GeneratorConfig
    train: list[Sample]
    cal: list[Sample]
    test: list[Sample]
    ood_cal: list[Sample]
    ood_test: list[Sample]
    ambiguous_intent: list[Sample]
    ambiguous_slot: list[Sample]
    user_adapt: dict[str, list[Sample]]
    user_eval: dict[str, list[Sample]]

    def summary(self) -> dict:
        return {
            "train": len(self.train), "cal": len(self.cal), "test": len(self.test),
            "ood_cal": len(self.ood_cal), "ood_test": len(self.ood_test),
            "ambiguous_intent": len(self.ambiguous_intent), "ambiguous_slot": len(self.ambiguous_slot),
            "users": {u: [len(self.user_adapt[u]), len(self.user_eval[u])] for u in self.user_adapt},
        }


def _partition(items: list, fractions: tuple[float, ...], rng: np.random.Generator) -> list[list]:
    """Shuffle and split; every part gets at least one item when possible."""
    items = list(items)
    rng.shuffle(items)
    n, k = len(items), len(fractions)
    if n == 0:
        return [[] for _ in fractions]
    counts = [max(1, int(round(f * n))) if n >= k else 0 for f in fractions]
    if n < k:
        counts = [1 if i < n else 0 for i in range(k)]
    while sum(counts) > n:
        counts[int(np.argmax(counts))] -= 1
    while sum(counts) < n:
        counts[0] += 1
    out, start = [], 0
    for c in counts:
        out.append(items[start:start + c])
        start += c
    return out


class SyntheticGenerator:
    def __init__(self, registry: IntentRegistry | None = None, config: GeneratorConfig | None = None):
        self.registry = registry or IntentRegistry.default()
        self.config = config or GeneratorConfig()
        self.ood = load_resource("ood.yaml")["topics"]
        self.ambiguous = load_resource("ambiguous.yaml")
        self.users = load_resource("users.yaml")["users"]
        self.priors = load_resource("context_priors.yaml")["categories"]

    # ----------------------------------------------------------------- utils
    def _apps_by_category(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for app_id, a in self.registry.apps.items():
            out.setdefault(a["category"], []).append(app_id)
        return out

    def _context(self, intent: str | None, rng: np.random.Generator) -> dict:
        cats = sorted(self.priors)
        apps = self._apps_by_category()
        if intent is not None and rng.random() < self.config.context_consistency:
            supportive = [c for c in cats if self.priors[c].get(intent, 1.0) >= 1.5]
            cat = str(rng.choice(supportive)) if supportive else str(rng.choice(cats))
        else:
            cat = str(rng.choice(cats))
        app = str(rng.choice(apps[cat])) if apps.get(cat) else ""
        return {"category": cat, "active_app": app}

    def _fill(self, template: str, lang: str, rng: np.random.Generator) -> tuple[str, dict]:
        slots: dict = {}
        fillers = self.registry.fillers.get(lang, {})

        def repl(m: re.Match) -> str:
            name = m.group(1)
            if name == "app":
                app_id = str(rng.choice(sorted(self.registry.apps)))
                forms = self.registry.apps[app_id].get(lang) or self.registry.apps[app_id]["en"]
                slots["app"] = app_id
                return str(rng.choice(forms))
            value = str(rng.choice(fillers[name]))
            slots[name] = int(value) if name == "value" else value
            return value

        return _PLACEHOLDER.sub(repl, template), slots

    def _wrap(self, text: str, lang: str, rng: np.random.Generator, plain: bool) -> str:
        if plain:
            return text
        w = self.registry.wrappers.get(lang, {})
        pre = str(rng.choice(w.get("prefix", [""])))
        suf = str(rng.choice(w.get("suffix", [""])))
        return f"{pre}{text}{suf}".strip()

    def _variants(self, template: str, lang: str, n: int, rng: np.random.Generator) -> list[tuple[str, dict]]:
        out, seen = [], set()
        for attempt in range(n * 6):
            text, slots = self._fill(template, lang, rng)
            text = self._wrap(text, lang, rng, plain=(attempt == 0))
            key = normalize(text)
            if key in seen:
                continue
            seen.add(key)
            out.append((text, slots))
            if len(out) >= n:
                break
        return out

    # ------------------------------------------------------------- generate
    def generate(self) -> IntentDataset:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        train, cal, test = [], [], []
        for intent in self.registry.names:
            spec = self.registry.spec(intent)
            for lang in cfg.languages:
                levels = spec.templates.get(lang, {})
                easy = [(lvl, i, t) for lvl in (0, 1) for i, t in enumerate(levels.get(lvl, []))]
                hard = [(lvl, i, t) for lvl in (2, 3) for i, t in enumerate(levels.get(lvl, []))]
                e_tr, e_ca, e_te = _partition(easy, cfg.easy_fractions, rng)
                h_ca, h_te = _partition(hard, cfg.hard_fractions, rng)
                for split, items, n in ((train, e_tr, cfg.variants_train), (cal, e_ca + h_ca, cfg.variants_cal),
                                        (test, e_te + h_te, cfg.variants_test)):
                    for lvl, i, tpl in items:
                        for text, slots in self._variants(tpl, lang, n, rng):
                            split.append(Sample(
                                text=text, intent=intent, lang=lang, kind="id", level=lvl,
                                template_id=f"{intent}/{lang}/{lvl}/{i}", slots=slots,
                                context=self._context(intent, rng),
                            ))
        train_texts = {normalize(s.text) for s in train}
        cal = [s for s in cal if normalize(s.text) not in train_texts]
        test = [s for s in test if normalize(s.text) not in train_texts]

        ood_cal, ood_test = self._ood(rng)
        amb_intent, amb_slot = self._ambiguous(rng)
        user_adapt, user_eval = self._users(rng)
        return IntentDataset(cfg, train, cal, test, ood_cal, ood_test, amb_intent, amb_slot, user_adapt, user_eval)

    def _ood(self, rng: np.random.Generator) -> tuple[list[Sample], list[Sample]]:
        near = sorted(t for t, d in self.ood.items() if d["kind"] == "near")
        far = sorted(t for t, d in self.ood.items() if d["kind"] == "far")
        near_cal, near_test = _partition(near, (1 / 3, 2 / 3), rng)
        far_cal, far_test = _partition(far, (0.5, 0.5), rng)
        out = {"cal": [], "test": []}
        for split, topics in (("cal", near_cal + far_cal), ("test", near_test + far_test)):
            for topic in topics:
                d = self.ood[topic]
                for lang in self.config.languages:
                    for j, base in enumerate(d.get(lang, [])):
                        seen = set()
                        for v in range(self.config.ood_variants):
                            text = self._wrap(base, lang, rng, plain=(v == 0))
                            if normalize(text) in seen:
                                continue
                            seen.add(normalize(text))
                            out[split].append(Sample(
                                text=text, intent=None, lang=lang, kind=f"ood_{d['kind']}",
                                template_id=f"ood/{topic}/{lang}/{j}", context=self._context(None, rng),
                                meta={"topic": topic},
                            ))
        return out["cal"], out["test"]

    def _ambiguous(self, rng: np.random.Generator) -> tuple[list[Sample], list[Sample]]:
        apps = self._apps_by_category()
        amb_intent, amb_slot = [], []
        for lang in self.config.languages:
            for j, item in enumerate(self.ambiguous["intent_ambiguous"].get(lang, [])):
                for cat, intent in sorted(item["resolve"].items()):
                    amb_intent.append(Sample(
                        text=item["text"], intent=intent, lang=lang, kind="ambiguous_intent",
                        template_id=f"amb_intent/{lang}/{j}",
                        context={"category": cat, "active_app": str(rng.choice(apps[cat]))},
                        meta={"candidates": sorted(set(item["resolve"].values()))},
                    ))
            aliases = self.registry.ambiguous_app_aliases.get(lang, {})
            for j, item in enumerate(self.ambiguous["slot_ambiguous"].get(lang, [])):
                alias = next((a for a in aliases if a in item["text"]), None)
                amb_slot.append(Sample(
                    text=item["text"], intent=item["intent"], lang=lang, kind="ambiguous_slot",
                    template_id=f"amb_slot/{lang}/{j}",
                    slots={"app": None}, context=self._context(item["intent"], rng),
                    meta={"app_candidates": aliases.get(alias, [])},
                ))
        return amb_intent, amb_slot

    def _users(self, rng: np.random.Generator) -> tuple[dict, dict]:
        adapt, evaluate = {}, {}
        for user, langs in sorted(self.users.items()):
            adapt[user], evaluate[user] = [], []
            for lang, expressions in langs.items():
                for j, e in enumerate(expressions):
                    variants = self._variants(e["template"], lang, self.config.user_variants, rng)
                    n_adapt = max(1, int(round(self.config.user_adapt_fraction * len(variants))))
                    for k, (text, slots) in enumerate(variants):
                        s = Sample(text=text, intent=e["intent"], lang=lang, kind="user",
                                   template_id=f"user/{user}/{lang}/{j}", slots=slots, user=user,
                                   context=self._context(e["intent"], rng))
                        (adapt if k < n_adapt else evaluate)[user].append(s)
        return adapt, evaluate


def generate(seed: int = 0, **overrides) -> IntentDataset:
    cfg = GeneratorConfig(seed=seed, **overrides)
    return SyntheticGenerator(config=cfg).generate()
