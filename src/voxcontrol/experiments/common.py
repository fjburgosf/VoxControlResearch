"""Shared building blocks for experiments."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..baselines import BaselineSuite, MethodOutput
from ..datasets.asr_noise import perturb
from ..datasets.synthetic import IntentDataset, Sample, generate
from ..decision.policies import decision_metrics
from ..ucil import UCIL, UCILConfig

COVERAGE_GRID = np.linspace(0.02, 1.0, 50)


@dataclass
class Bundle:
    data: IntentDataset
    ucil: UCIL
    suite: BaselineSuite


def ucil_config(cfg: dict, seed: int, **overrides) -> UCILConfig:
    d = dict(cfg.get("ucil", {}))
    d.update(overrides)
    d["seed"] = seed
    if "costs" not in d and "costs" in cfg:
        d["costs"] = cfg["costs"]
    return UCILConfig.from_dict(d)


def build(seed: int, cfg: dict, data: IntentDataset | None = None, **ucil_overrides) -> Bundle:
    data = data or generate(seed, **cfg.get("dataset", {}))
    ucil = UCIL(config=ucil_config(cfg, seed, **ucil_overrides)).fit(data.train, data.cal, data.ood_cal)
    suite = BaselineSuite(ucil, tau_fixed=float(cfg.get("baselines", {}).get("tau_fixed", 0.7)))
    suite.fit(data.cal + data.ood_cal)
    return Bundle(data, ucil, suite)


def noisy(samples: list[Sample], rate: float, seed: int) -> list[Sample]:
    rng = np.random.default_rng(seed)
    return [replace(s, text=perturb(s.text, rate, rng, s.lang), noise=rate) for s in samples]


def method_rows(outs: dict[str, MethodOutput], samples, costs, **tags) -> list[dict]:
    ind = np.array([s.in_domain for s in samples])
    rows = []
    for name, o in outs.items():
        m = decision_metrics(o.decisions, ind, o.correct, o.risks, costs)
        acc = float(np.mean(o.correct[ind])) if ind.any() else float("nan")
        rows.append({**tags, "method": name, "intent_accuracy": acc, **m})
    return rows


def interp_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    return np.interp(grid, np.asarray(x)[order], np.asarray(y)[order])
