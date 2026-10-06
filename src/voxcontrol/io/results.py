"""Reproducible experiment records.

results/EXP-<year>-<NNNNNN>/
    config.yaml, metadata.json, <table>.csv, summary_<table>.csv, figures/, logs/run.log
"""
from __future__ import annotations

import csv
import json
import logging
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from .. import __version__
from ..metrics.stats import describe


def environment() -> dict:
    import matplotlib
    import scipy
    import sklearn
    return {"software_version": __version__, "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__}


def next_experiment_id(root: Path) -> str:
    year = time.strftime("%Y")
    root.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name.split("-")[-1]) for p in root.glob(f"EXP-{year}-*") if p.name.split("-")[-1].isdigit()]
    return f"EXP-{year}-{(max(nums) + 1 if nums else 1):06d}"


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return "" if not np.isfinite(v) else round(float(v), 6)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return v


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _clean(r.get(k, "")) for k in keys})


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if v not in ("", None) and not v.isalpha() else v
            except (ValueError, AttributeError):
                pass
    return rows


def summarise(rows: list[dict], group_keys: list[str], exclude: tuple[str, ...] = ("seed",)) -> list[dict]:
    """Mean, SD, median, IQR and 95% CI of every numeric column, per group (across seeds)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in group_keys)].append(r)
    out = []
    for key, rs in groups.items():
        base = dict(zip(group_keys, key))
        numeric = [k for k in rs[0] if k not in group_keys and k not in exclude
                   and isinstance(rs[0][k], (int, float, np.integer, np.floating)) and not isinstance(rs[0][k], bool)]
        for k in numeric:
            d = describe([r[k] for r in rs if isinstance(r.get(k), (int, float, np.integer, np.floating))])
            if d.get("n", 0) == 0:
                continue
            out.append({**base, "metric": k, **d})
    return out


class ExperimentRecorder:
    def __init__(self, name: str, config: dict, root: str | Path = "results", experiment_id: str | None = None):
        self.root = Path(root)
        self.id = experiment_id or next_experiment_id(self.root)
        self.dir = self.root / self.id
        (self.dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        self.name = name
        self.config = config
        self.started = time.time()
        with open(self.dir / "config.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=False)
        self.logger = logging.getLogger(f"voxcontrol.{self.id}")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.dir / "logs" / "run.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)
        self._handler = handler

    def table(self, name: str, rows: list[dict]) -> Path:
        path = self.dir / f"{name}.csv"
        write_csv(path, rows)
        return path

    def figure_path(self, name: str) -> Path:
        return self.dir / "figures" / name

    def finish(self, extra: dict | None = None) -> None:
        meta = {"experiment_id": self.id, "name": self.name, "started": time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(self.started)), "duration_s": round(time.time() - self.started, 1),
            "environment": environment(), **(extra or {})}
        with open(self.dir / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
        self.logger.removeHandler(self._handler)
        self._handler.close()
