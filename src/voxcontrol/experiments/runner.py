"""Run an experiment over several seeds, aggregate and export."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path

import yaml

from ..intents.registry import IntentRegistry
from ..io.results import ExperimentRecorder, summarise
from ..visualization.charts import make_figures
from .text_experiments import EXPERIMENTS

log = logging.getLogger("voxcontrol.runner")

GROUPS = {
    "decision_metrics": ["method"], "classifier_metrics": ["model"], "accuracy_by_level": ["model", "level"],
    "decisions_by_level": ["method", "level"], "error_detection": ["measure", "errors_of"],
    "calibration": ["calibrator", "population"], "ucil_components": [], "risk_coverage": ["confidence"],
    "tradeoff_curve": ["method", "param"], "ood_detection": ["detector", "subset"],
    "ood_decisions": ["method", "subset"], "risk_sensitivity": ["method", "risk"], "execution_thresholds": ["risk"],
    "confidence_matched_execution": ["method", "predicted_risk", "conf_bin"],
    "ambiguity_decisions": ["method", "set"], "ambiguity_uncertainty": ["set"],
    "slot_ambiguity_summary": ["method"], "asr_noise": ["method", "noise_rate"],
    "incremental": ["strategy", "n_corrections"], "personalization": ["strategy", "n_corrections"],
    "combined_shift": ["method"], "combined_by_subset": ["method", "subset"], "ablation": ["method", "setting"],
    "audio_noise": ["method", "snr_db"], "audio_error_attribution": ["snr_db"],
}


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run(cfg: dict, root: str | Path = "results", progress=print) -> ExperimentRecorder:
    name = cfg["experiment"]["name"]
    kind = cfg["experiment"]["type"]
    seeds = list(cfg["experiment"].get("seeds", [0]))
    if kind == "audio_noise":
        from .audio_experiment import exp_audio as fn
    else:
        fn = EXPERIMENTS[kind]
    rec = ExperimentRecorder(name, cfg, root)
    rec.logger.info("experiment %s (%s) seeds=%s", name, kind, seeds)
    tables: dict[str, list] = defaultdict(list)
    curves = []
    for seed in seeds:
        t0 = time.time()
        res = fn(seed, cfg)
        for k, rows in res["tables"].items():
            tables[k].extend(rows)
        curves.append(res.get("curves", {}))
        msg = f"[{rec.id}] {kind} seed {seed} done in {time.time() - t0:.1f}s"
        rec.logger.info(msg)
        progress(msg)
    for k, rows in tables.items():
        rec.table(k, rows)
        if k in GROUPS:
            rec.table(f"summary_{k}", summarise(rows, GROUPS[k]))
    if kind == "personalization":
        rec.table("summary_personalization_by_user", summarise(tables["personalization"],
                                                               ["user", "strategy", "n_corrections"]))
    try:
        make_figures(kind, tables, curves, rec.dir / "figures", classes=IntentRegistry.default().names)
    except Exception:  # noqa: BLE001 - figures must not lose computed results
        rec.logger.exception("figure generation failed")
        raise
    rec.finish({"seeds": seeds, "type": kind, "tables": sorted(tables)})
    return rec
