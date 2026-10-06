"""Explicit, reproducible interaction context c_t and the context prior w(y | c)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .intents.registry import IntentRegistry, load_resource


@dataclass
class ContextState:
    active_app: str | None = None
    category: str | None = None
    previous_intent: str | None = None
    previous_target: str | None = None
    dialog_state: str = "idle"            # idle | awaiting_confirmation | awaiting_clarification
    history: list = field(default_factory=list)   # compact: last intents only

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None, registry: IntentRegistry | None = None) -> "ContextState":
        d = dict(d or {})
        cat = d.get("category")
        if cat is None and d.get("active_app") and registry is not None:
            cat = registry.app_category(d["active_app"])
        return cls(active_app=d.get("active_app"), category=cat, previous_intent=d.get("previous_intent"),
                   previous_target=d.get("previous_target"), dialog_state=d.get("dialog_state", "idle"),
                   history=list(d.get("history", []))[-5:])

    def remember(self, intent: str, target: str | None) -> None:
        self.previous_intent, self.previous_target = intent, target
        self.history = (self.history + [intent])[-5:]


class ContextPrior:
    """p(y | x, c) proportional to p(y | x) * w(y | c) ** alpha."""

    def __init__(self, classes: list[str], weights: dict | None = None, alpha: float = 1.0):
        self.classes = list(classes)
        self.alpha = alpha
        self.weights = weights if weights is not None else load_resource("context_priors.yaml")["categories"]

    def vector(self, category: str | None) -> np.ndarray:
        w = self.weights.get(category or "", {})
        return np.array([float(w.get(c, 1.0)) for c in self.classes]) ** self.alpha

    def apply(self, probs: np.ndarray, categories: list[str | None]) -> tuple[np.ndarray, np.ndarray]:
        """Return reweighted probabilities and the total-variation shift U_context."""
        probs = np.asarray(probs, dtype=np.float64)
        W = np.stack([self.vector(c) for c in categories])
        post = probs * W
        post /= post.sum(axis=1, keepdims=True)
        shift = 0.5 * np.abs(post - probs).sum(axis=1)
        return post, shift
