"""Publication figures (PNG + SVG + PDF) in English and Spanish.

Every figure is written twice: ``fig_x.png`` (English) and ``fig_x_es.png`` (Spanish).
Colour follows the method, never its rank.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METHOD_COLOURS = {
    "UCIL": PALETTE[0], "B4_calibrated_threshold": PALETTE[1], "B3_fixed_threshold": PALETTE[2],
    "B2_argmax": PALETTE[3], "B1_rules": PALETTE[4], "UCIL_context": PALETTE[5],
    "UCIL_noise_calibrated": PALETTE[6], "UCIL_no_incremental": PALETTE[7],
    "UCIL_fusion_with_asr": PALETTE[5], "UCIL_fusion_without_asr": PALETTE[6], "UCIL_no_asr_uncertainty": PALETTE[7],
}
LABELS = {"UCIL": "UCIL", "B4_calibrated_threshold": "B4 calibrated threshold",
          "B3_fixed_threshold": "B3 fixed threshold", "B2_argmax": "B2 argmax", "B1_rules": "B1 rules",
          "UCIL_context": "UCIL + context", "UCIL_noise_calibrated": "UCIL (noise-aware calibration)",
          "UCIL_no_incremental": "UCIL without adaptation"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LANGUAGES = ("en", "es")

ES = {
    "B4 calibrated threshold": "B4 umbral calibrado", "B3 fixed threshold": "B3 umbral fijo",
    "B1 rules": "B1 reglas", "UCIL + context": "UCIL + contexto",
    "UCIL (noise-aware calibration)": "UCIL (calibración con ruido)", "UCIL without adaptation": "UCIL sin adaptación",
    " (sweep)": " (barrido)", " (operating point)": " (punto de operación)",
    "Incorrect execution rate": "Tasa de ejecuciones incorrectas",
    "Wrong-execution cost per command": "Costo de ejecuciones erróneas por orden",
    "Intervention rate (confirm + reject/clarify)": "Tasa de intervención (confirmar + rechazar/aclarar)",
    "Coverage (fraction executed)": "Cobertura (fracción ejecutada)", "Risk = P(error | executed)": "Riesgo = P(error | ejecutada)",
    "B2 raw confidence": "B2 confianza bruta", "B4 temperature confidence": "B4 confianza con temperatura",
    "UCIL p correct": "UCIL P(correcta)",
    "Uncalibrated (in-domain)": "Sin calibrar (en dominio)", "Temperature scaling (in-domain)": "Escalado por temperatura (en dominio)",
    "UCIL P(correct execution) (ID + OOD)": "UCIL P(ejecución correcta) (ID + OOD)",
    "Confidence": "Confianza", "Observed accuracy": "Exactitud observada",
    "Predicted intent": "Intención predicha", "True intent": "Intención verdadera", "Row-normalised rate": "Tasa normalizada por fila",
    "Paraphrase difficulty level": "Nivel de dificultad de la paráfrasis", "Intent accuracy": "Exactitud de intención",
    "AUROC (in-domain vs OOD)": "AUROC (en dominio vs OOD)", "Clarification rate": "Tasa de confirmación",
    "Mean realised cost per command": "Costo medio realizado por orden", "Mean realised cost": "Costo medio realizado",
    "Simulated ASR error rate (per word)": "Tasa de error ASR simulada (por palabra)",
    "SNR (dB); 40 = clean": "SNR (dB); 40 = limpio", "Number of user corrections": "Número de correcciones del usuario",
    "Accuracy on new expressions": "Exactitud en expresiones nuevas",
    "Forgetting (old-knowledge accuracy drop)": "Olvido (caída de exactitud en lo ya aprendido)",
    "Corrections from this user": "Correcciones de este usuario",
    "Accuracy on this user's phrasing": "Exactitud en la forma de hablar del usuario",
    "Accuracy for other users": "Exactitud para otros usuarios",
    "Accuracy on context-dependent commands": "Exactitud en órdenes dependientes del contexto",
    "far": "lejano", "near": "cercano", "fused": "fusión", "UCIL 1-P(in)": "UCIL 1-P(dominio)",
    "low": "bajo", "medium": "medio", "high": "alto", "id": "en dominio", "ood far": "OOD lejano",
    "ood near": "OOD cercano", "user": "usuario", "clean": "limpio", "combined shift": "cambio combinado",
    "memory": "memoria", "none": "ninguna", "prototype": "prototipos", "finetune naive": "ajuste ingenuo",
    "replay fifo": "replay FIFO", "replay class balanced": "replay balanceado", "replay uncertainty": "replay por incertidumbre",
    "replay diversity": "replay por diversidad", "ucil memory": "memoria UCIL", "rules": "reglas",
    "tfidf logreg": "TF-IDF + regresión logística", "embedding softmax": "embeddings + softmax", "ensemble": "ensamble",
    "UCIL full": "UCIL completo", "no ood": "sin OOD", "no calibration": "sin calibración",
    "no risk routing": "sin enrutamiento por riesgo", "no incremental": "sin aprendizaje incremental",
    "lexical": "léxico", "energy": "energía", "knn": "kNN", "msp": "MSP", "Mahalanobis": "Mahalanobis",
    "UCIL no asr uncertainty": "UCIL sin U_ASR", "UCIL fusion with asr": "UCIL fusión con U_ASR",
    "UCIL fusion without asr": "UCIL fusión sin U_ASR",
}
_lang = "en"


def tr(text: str) -> str:
    return ES.get(text, text) if _lang == "es" else text


def label(name) -> str:
    return tr(LABELS.get(name, str(name).replace("_", " ")))


def _style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "legend.frameon": False,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    })


def colour(name: str, i: int = 0) -> str:
    return METHOD_COLOURS.get(name, PALETTE[i % len(PALETTE)])


def save(fig, path: Path, pdf: bool = False) -> None:
    if _lang != "en":
        path = path.with_name(f"{path.name}_{_lang}")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".svg"))
    if pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _mean_by(rows, keys, value):
    g = defaultdict(list)
    for r in rows:
        v = r.get(value)
        if isinstance(v, (int, float)) and np.isfinite(v):
            g[tuple(r[k] for k in keys)].append(v)
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) for k, v in g.items()}


# ------------------------------------------------------------------ main
def fig_tradeoff(tables, out: Path):
    """Left: unweighted incorrect-execution rate. Right: cost-weighted wrong executions."""
    panels = (("incorrect_execution_rate", "Incorrect execution rate"),
              ("wrong_execution_cost", "Wrong-execution cost per command"))
    xs = _mean_by(tables["tradeoff_curve"], ["method", "param"], "intervention_rate")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, (metric, ylabel) in zip(axes, panels):
        curve = _mean_by(tables["tradeoff_curve"], ["method", "param"], metric)
        for m in ("B3_fixed_threshold", "B4_calibrated_threshold", "UCIL"):
            keys = [k for k in curve if k[0] == m]
            x = np.array([xs[k][0] for k in keys]); y = np.array([curve[k][0] for k in keys])
            o = np.argsort(x)
            ax.plot(x[o], y[o], color=colour(m), label=label(m) + tr(" (sweep)"))
        pts = _mean_by(tables["decision_metrics"], ["method"], metric)
        for (m,), (y, sd) in sorted(pts.items(), key=lambda kv: kv[0][0] != "UCIL"):
            x = float(np.mean([r["clarification_rate"] + r["rejection_rate"]
                               for r in tables["decision_metrics"] if r["method"] == m]))
            ax.errorbar(x, y, yerr=sd, fmt="o", ms=7, color=colour(m), mec="#fcfcfb", mew=1.5, capsize=2,
                        label=label(m) + tr(" (operating point)"))
        ax.set_xlabel(tr("Intervention rate (confirm + reject/clarify)"))
        ax.set_ylabel(tr(ylabel))
        ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
    axes[1].legend(loc="upper right", fontsize=7)
    save(fig, out / "fig_tradeoff_ier_vs_intervention", pdf=True)


def fig_risk_coverage(curves_list, out: Path):
    from ..experiments.common import COVERAGE_GRID
    names = list(curves_list[0]["risk_coverage"])
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for i, k in enumerate(names):
        Y = np.stack([c["risk_coverage"][k] for c in curves_list])
        m = Y.mean(0)
        col = colour({"UCIL_p_correct": "UCIL", "B4_temperature_confidence": "B4_calibrated_threshold",
                      "B2_raw_confidence": "B2_argmax"}.get(k, k), i)
        ax.plot(COVERAGE_GRID, m, color=col, label=tr(k.replace("_", " ")))
        ax.fill_between(COVERAGE_GRID, Y.min(0), Y.max(0), color=col, alpha=0.12, linewidth=0)
    ax.set_xlabel(tr("Coverage (fraction executed)")); ax.set_ylabel(tr("Risk = P(error | executed)"))
    ax.set_xlim(0, 1); ax.set_ylim(bottom=0); ax.legend(fontsize=7.5)
    save(fig, out / "fig_risk_coverage", pdf=True)


def fig_reliability(curves_list, out: Path):
    keys = ["none", "temperature", "ucil"]
    titles = {"none": "Uncalibrated (in-domain)", "temperature": "Temperature scaling (in-domain)",
              "ucil": "UCIL P(correct execution) (ID + OOD)"}
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), sharey=True)
    for ax, k in zip(axes, keys):
        cnt = sum(c["reliability"][k]["count"] for c in curves_list)
        acc = sum(c["reliability"][k]["accuracy"] * c["reliability"][k]["count"] for c in curves_list) / np.maximum(cnt, 1)
        conf = sum(c["reliability"][k]["confidence"] * c["reliability"][k]["count"] for c in curves_list) / np.maximum(cnt, 1)
        edges = curves_list[0]["reliability"][k]["edges"]
        centres = (edges[:-1] + edges[1:]) / 2
        mask = cnt > 0
        ax.bar(centres[mask], acc[mask], width=(edges[1] - edges[0]) * 0.9, color=PALETTE[0], alpha=0.85)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
        ax.plot(conf[mask], acc[mask], "o", color=PALETTE[1], ms=3)
        ax.set_title(tr(titles[k]), fontsize=8.5)
        ax.set_xlabel(tr("Confidence"))
    axes[0].set_ylabel(tr("Observed accuracy"))
    save(fig, out / "fig_reliability_diagrams", pdf=True)


def fig_confusion(curves_list, classes, out: Path):
    C = sum(c["confusion"] for c in curves_list).astype(float)
    C = C / np.maximum(C.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel(tr("Predicted intent")); ax.set_ylabel(tr("True intent")); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label=tr("Row-normalised rate"))
    save(fig, out / "fig_confusion_matrix")


def fig_lines(rows, x, y, series, out: Path, name: str, xlabel: str, ylabel: str, ylim=(0, None)):
    m = _mean_by(rows, [series, x], y)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for i, s in enumerate(sorted({k[0] for k in m}, key=lambda s: (s != "UCIL", str(s)))):
        keys = sorted((k for k in m if k[0] == s), key=lambda k: k[1])
        xs = np.array([k[1] for k in keys]); ys = np.array([m[k][0] for k in keys]); sd = np.array([m[k][1] for k in keys])
        c = colour(str(s), i)
        ax.plot(xs, ys, marker="o", ms=4, color=c, label=label(s))
        ax.fill_between(xs, ys - sd, ys + sd, color=c, alpha=0.12, linewidth=0)
    xs_all = sorted({k[1] for k in m})
    if len(xs_all) <= 12 and all(float(x).is_integer() for x in xs_all):
        ax.set_xticks(xs_all)
    ax.set_xlabel(tr(xlabel)); ax.set_ylabel(tr(ylabel))
    ax.set_ylim(*ylim)
    ax.legend(fontsize=7, ncol=1)
    save(fig, out / name, pdf=True)


def fig_grouped_bars(rows, group, series, value, out: Path, name: str, ylabel: str):
    m = _mean_by(rows, [group, series], value)
    groups = sorted({k[0] for k in m}, key=str)
    sers = sorted({k[1] for k in m}, key=lambda s: (s != "UCIL", str(s)))
    w = 0.8 / max(len(sers), 1)
    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * len(groups) * len(sers) * 0.35 + 2), 3.6))
    for i, s in enumerate(sers):
        xs = np.arange(len(groups)) + (i - (len(sers) - 1) / 2) * w
        ys = [m.get((g, s), (np.nan, 0))[0] for g in groups]
        sd = [m.get((g, s), (np.nan, 0))[1] for g in groups]
        ax.bar(xs, ys, width=w * 0.92, yerr=sd, capsize=2, color=colour(str(s), i),
               label=label(s), error_kw={"elinewidth": 0.8, "ecolor": MUTED})
    nice = {"ucil_1_minus_p_in": "UCIL 1-P(in)", "fused": "fused", "mahalanobis": "Mahalanobis"}
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([tr(nice.get(str(g), str(g).replace("_", " "))) for g in groups], fontsize=8,
                       rotation=20 if len(groups) > 5 else 0, ha="right" if len(groups) > 5 else "center")
    ax.set_ylabel(tr(ylabel)); ax.legend(fontsize=7)
    save(fig, out / name)


def fig_bars(rows, key, value, out: Path, name: str, ylabel: str):
    m = _mean_by(rows, [key], value)
    names = sorted((k[0] for k in m), key=lambda s: (s != "UCIL", str(s)))
    fig, ax = plt.subplots(figsize=(max(4.5, 0.9 * len(names) + 1.5), 3.4))
    for i, n in enumerate(names):
        y, sd = m[(n,)]
        ax.bar(i, y, yerr=sd, width=0.6, capsize=2, color=colour(n, i), error_kw={"elinewidth": 0.8, "ecolor": MUTED})
        ax.annotate(f"{y:.3f}", (i, y + sd), textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=7.5, color=INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([label(n) for n in names], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(tr(ylabel))
    save(fig, out / name)


def _figures(experiment: str, tables: dict, curves_list: list, out: Path, classes=None) -> None:
    if experiment == "main":
        if curves_list and curves_list[0]:
            fig_risk_coverage(curves_list, out)
            fig_reliability(curves_list, out)
            if classes:
                fig_confusion(curves_list, classes, out)
        fig_tradeoff(tables, out)
        fig_lines(tables["accuracy_by_level"], "level", "accuracy", "model", out, "fig_accuracy_by_difficulty",
                  "Paraphrase difficulty level", "Intent accuracy", (0, 1.02))
        fig_grouped_bars([r for r in tables["ood_detection"] if r["subset"] != "all"], "detector", "subset", "auroc",
                         out, "fig_ood_auroc", "AUROC (in-domain vs OOD)")
        fig_grouped_bars(tables["risk_sensitivity"], "risk", "method", "clarification_rate", out,
                         "fig_risk_clarification", "Clarification rate")
        fig_bars(tables["decision_metrics"], "method", "mean_cost", out, "fig_mean_cost",
                 "Mean realised cost per command")
        fig_bars(tables["decision_metrics"], "method", "incorrect_execution_rate", out, "fig_incorrect_execution",
                 "Incorrect execution rate")
    elif experiment == "asr_text_noise":
        for y, lab in (("incorrect_execution_rate", "Incorrect execution rate"), ("intent_accuracy", "Intent accuracy"),
                       ("clarification_rate", "Clarification rate"), ("mean_cost", "Mean realised cost")):
            rows = [r for r in tables["asr_noise"] if not (y.endswith("cost") and r["method"] == "B2_argmax")]
            fig_lines(rows, "noise_rate", y, "method", out, f"fig_noise_{y}",
                      "Simulated ASR error rate (per word)", lab)
    elif experiment == "audio_noise":
        rows = []
        for r in tables["audio_noise"]:
            rows.append({**r, "snr_x": 40.0 if str(r["snr_db"]) == "clean" else float(r["snr_db"])})
        for y, lab in (("incorrect_execution_rate", "Incorrect execution rate"), ("intent_accuracy", "Intent accuracy"),
                       ("wrong_execution_cost", "Wrong-execution cost per command"), ("mean_cost", "Mean realised cost")):
            sel = [r for r in rows if not (y.endswith("cost") and r["method"] == "B2_argmax")]
            fig_lines(sel, "snr_x", y, "method", out, f"fig_audio_{y}", "SNR (dB); 40 = clean", lab)
    elif experiment == "incremental":
        fig_lines(tables["incremental"], "n_corrections", "acc_new_expressions", "strategy", out,
                  "fig_adaptation_curve", "Number of user corrections", "Accuracy on new expressions", (0, 1.02))
        fig_lines(tables["incremental"], "n_corrections", "forgetting", "strategy", out, "fig_forgetting_curve",
                  "Number of user corrections", "Forgetting (old-knowledge accuracy drop)", (None, None))
    elif experiment == "personalization":
        fig_lines(tables["personalization"], "n_corrections", "user_accuracy", "strategy", out,
                  "fig_personalization_user", "Corrections from this user", "Accuracy on this user's phrasing", (0, 1.02))
        fig_lines(tables["personalization"], "n_corrections", "other_users_accuracy", "strategy", out,
                  "fig_personalization_other_users", "Corrections from this user", "Accuracy for other users", (0, 1.02))
    elif experiment == "combined_shift":
        fig_grouped_bars(tables["combined_by_subset"], "subset", "method", "mean_cost", out, "fig_combined_cost",
                         "Mean realised cost")
        fig_grouped_bars(tables["combined_by_subset"], "subset", "method", "incorrect_execution_rate", out,
                         "fig_combined_ier", "Incorrect execution rate")
    elif experiment == "ablation":
        for v, lab in (("mean_cost", "Mean realised cost"), ("incorrect_execution_rate", "Incorrect execution rate")):
            fig_grouped_bars(tables["ablation"], "setting", "method", v, out, f"fig_ablation_{v}", lab)
    elif experiment == "ambiguity":
        fig_bars(tables["ambiguity_decisions"], "method", "intent_accuracy", out,
                 "fig_ambiguity_accuracy", "Accuracy on context-dependent commands")


def make_figures(experiment: str, tables: dict, curves_list: list, out: Path, classes=None) -> None:
    global _lang
    _style()
    out.mkdir(parents=True, exist_ok=True)
    try:
        for lang in LANGUAGES:
            _lang = lang
            _figures(experiment, tables, curves_list, out, classes)
    finally:
        _lang = "en"


def regenerate(exp_dir: str | Path) -> None:
    """Rebuild the table-based figures of an experiment folder from its CSV files."""
    import yaml

    from ..io.results import read_csv
    exp_dir = Path(exp_dir)
    kind = yaml.safe_load((exp_dir / "config.yaml").read_text(encoding="utf-8"))["experiment"]["type"]
    tables = {p.stem: read_csv(p) for p in exp_dir.glob("*.csv")
              if not p.stem.startswith(("summary_", "comparisons_"))}
    make_figures(kind, tables, [{}], exp_dir / "figures")
