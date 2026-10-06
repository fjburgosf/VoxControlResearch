"""Paired statistical comparisons between methods, computed from an experiment's CSV tables.

For every table with ``method`` and ``seed`` columns, each method is compared with the reference
(UCIL) on the same seeds and conditions: mean paired difference with a 95% t-interval, Cohen's d_z
and the Wilcoxon signed-rank p-value (secondary).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

from .io.results import read_csv, write_csv
from .metrics.stats import cohens_dz, wilcoxon_paired

METRICS = ("mean_cost", "incorrect_execution_rate", "wrong_execution_cost", "high_risk_incorrect_execution_rate",
           "clarification_rate", "rejection_rate", "false_rejection_rate", "intent_accuracy",
           "user_burden_per_command")
CONDITION_KEYS = ("noise_rate", "snr_db", "setting", "set", "level", "subset", "risk")


def paired_comparisons(rows: list[dict], reference: str = "UCIL") -> list[dict]:
    conds = [k for k in CONDITION_KEYS if rows and k in rows[0]]
    by: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by[tuple(r.get(k) for k in conds) + (r["seed"],)][r["method"]] = r
    methods = sorted({r["method"] for r in rows})
    if reference not in methods:
        reference = next((m for m in methods if m.startswith("UCIL")), methods[0])
    out = []
    groups = sorted({k[:-1] for k in by}, key=str)
    for g in groups:
        cells = [v for k, v in by.items() if k[:-1] == g]
        for other in methods:
            if other == reference:
                continue
            for metric in METRICS:
                pairs = [(c[reference][metric], c[other][metric]) for c in cells
                         if reference in c and other in c and isinstance(c[reference].get(metric), float)
                         and isinstance(c[other].get(metric), float)]
                if len(pairs) < 2:
                    continue
                a, b = np.array(pairs).T
                d = a - b
                half = sps.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                out.append({**dict(zip(conds, g)), "reference": reference, "compared_with": other,
                            "metric": metric, "n_seeds": len(d), "reference_mean": float(a.mean()),
                            "other_mean": float(b.mean()), "mean_difference": float(d.mean()),
                            "ci95_low": float(d.mean() - half), "ci95_high": float(d.mean() + half),
                            "cohens_dz": cohens_dz(a, b), "wilcoxon_p": wilcoxon_paired(a, b),
                            "reference_better_in_seeds": int(np.sum(d < 0) if metric != "intent_accuracy"
                                                             else np.sum(d > 0))})
    return out


def analyze(exp_dir: str | Path, reference: str = "UCIL") -> list[Path]:
    exp_dir = Path(exp_dir)
    written = []
    for path in sorted(exp_dir.glob("*.csv")):
        if path.stem.startswith(("summary_", "comparisons_")) or path.stem == "tradeoff_curve":
            continue
        rows = read_csv(path)
        if not rows or "method" not in rows[0] or "seed" not in rows[0]:
            continue
        comp = paired_comparisons(rows, reference)
        if comp:
            out = exp_dir / f"comparisons_{path.stem}.csv"
            write_csv(out, comp)
            written.append(out)
    return written
