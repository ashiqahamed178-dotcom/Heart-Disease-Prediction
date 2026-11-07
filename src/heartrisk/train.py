"""End-to-end training: data -> split -> tuned CV comparison -> threshold -> test -> save.

Run with ``python -m heartrisk.train``. Outputs go to ``models/`` and ``reports/``.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from heartrisk.config import (
    CV_FOLDS,
    FEATURE_LABELS,
    FEATURES,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SITES,
    TARGET,
    TARGET_RECALL,
    TEST_SIZE,
)
from heartrisk.data import data_report, load_dataset
from heartrisk.evaluate import (
    bootstrap_ci,
    choose_threshold,
    ranking_metrics,
    threshold_metrics,
)
from heartrisk.explain import explain, make_explainer
from heartrisk.models import ModelSpec, build_pipeline, model_specs

log = logging.getLogger("heartrisk.train")


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratify on site and label together, so every hospital is in both halves."""
    strata = df["site"] + "_" + df[TARGET].astype(str)
    train, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=strata, random_state=RANDOM_STATE
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def tune(spec: ModelSpec, x: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> Pipeline:
    pipe = build_pipeline(spec.estimator)
    if not spec.grid:
        return pipe.fit(x, y)
    search = GridSearchCV(pipe, spec.grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(x, y)
    log.info("%s best params %s", spec.name, search.best_params_)
    best: Pipeline = search.best_estimator_
    return best


def select_model(cv_results: dict[str, dict[str, Any]], specs: list[ModelSpec]) -> str:
    """One-standard-error rule: the simplest model within 1 SE of the best CV ROC-AUC."""
    candidates = [s for s in specs if s.name != "baseline_prior"]
    best = max(candidates, key=lambda s: cv_results[s.name]["cv_roc_auc_mean"])
    bar = cv_results[best.name]["cv_roc_auc_mean"] - cv_results[best.name]["cv_roc_auc_se"]
    within = [s for s in candidates if cv_results[s.name]["cv_roc_auc_mean"] >= bar]
    return min(within, key=lambda s: s.rank).name


def leave_one_site_out(
    template: Pipeline, df: pd.DataFrame, threshold: float
) -> dict[str, dict[str, Any]]:
    """Train on three hospitals, test on the fourth. Shows how well it travels."""
    out: dict[str, dict[str, Any]] = {}
    for site in SITES:
        tr, te = df[df["site"] != site], df[df["site"] == site]
        model = clone(template).fit(tr[FEATURES], tr[TARGET])
        proba = model.predict_proba(te[FEATURES])[:, 1]
        entry: dict[str, Any] = {
            "n": len(te),
            "positives": int(te[TARGET].sum()),
            **threshold_metrics(te[TARGET], proba, threshold),
        }
        if te[TARGET].nunique() == 2:
            entry.update(ranking_metrics(te[TARGET], proba))
        out[site] = entry
    return out


def by_sex(test: pd.DataFrame, proba: np.ndarray, threshold: float) -> dict[str, dict[str, Any]]:
    """Subgroup check. Women are about 21% of the data, so their numbers are noisy."""
    out: dict[str, dict[str, Any]] = {}
    for label, code in (("female", 0), ("male", 1)):
        mask = (test["sex"] == code).to_numpy()
        y = test.loc[mask, TARGET]
        entry: dict[str, Any] = {
            "n": int(mask.sum()),
            "positives": int(y.sum()),
            **threshold_metrics(y, proba[mask], threshold),
        }
        if y.nunique() == 2:
            entry.update(ranking_metrics(y, proba[mask]))
        out[label] = entry
    return out


def plot_roc(test_probs: dict[str, np.ndarray], y: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in test_probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = ranking_metrics(y, p)["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate (recall)",
        title="ROC curves, held-out test set",
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_test.png", dpi=150)
    plt.close(fig)


def plot_calibration(test_probs: dict[str, np.ndarray], y: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in test_probs.items():
        if name == "baseline_prior":
            continue
        frac, mean = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ax.set(
        xlabel="Predicted probability",
        ylabel="Observed disease rate",
        title="Calibration, held-out test set",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_test.png", dpi=150)
    plt.close(fig)


def plot_threshold(y_oof: pd.Series, p_oof: np.ndarray, threshold: float, name: str) -> None:
    grid = np.linspace(0.05, 0.95, 91)
    rec = [threshold_metrics(y_oof, p_oof, t)["recall"] for t in grid]
    prec = [threshold_metrics(y_oof, p_oof, t)["precision"] for t in grid]
    spec = [threshold_metrics(y_oof, p_oof, t)["specificity"] for t in grid]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(grid, rec, label="recall")
    ax.plot(grid, prec, label="precision")
    ax.plot(grid, spec, label="specificity")
    ax.axvline(threshold, color="k", ls="--", lw=1, label=f"chosen {threshold:.2f}")
    ax.axhline(TARGET_RECALL, color="grey", ls=":", lw=1)
    ax.set(
        xlabel="Threshold",
        ylabel="Score",
        title=f"Threshold choice ({name}, out-of-fold on training data)",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_oof.png", dpi=150)
    plt.close(fig)


def plot_shap(exp: shap.Explanation) -> dict[str, float]:
    exp.feature_names = [FEATURE_LABELS[f] for f in FEATURES]
    plt.figure()
    shap.plots.beeswarm(exp, show=False, max_display=len(FEATURES))
    plt.title("SHAP values on the test set (log-odds)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    mean_abs = np.abs(exp.values).mean(axis=0)
    return {
        f: round(float(v), 4)
        for f, v in sorted(zip(FEATURES, mean_abs, strict=True), key=lambda kv: -kv[1])
    }


def write_markdown(metrics: dict[str, Any]) -> None:
    lines = ["# Results", "", "Generated by `python -m heartrisk.train`. Do not edit by hand.", ""]
    lines += [
        "## Cross-validation (training set, 5 folds, tuned)",
        "",
        "| Model | CV ROC-AUC (mean ± SE) |",
        "|---|---|",
    ]
    for name, r in metrics["cv"].items():
        lines.append(f"| {name} | {r['cv_roc_auc_mean']:.3f} ± {r['cv_roc_auc_se']:.3f} |")
    lines += [
        "",
        f"Selected: **{metrics['selected_model']}** "
        f"(threshold {metrics['threshold']:.3f}, chosen for out-of-fold recall "
        f"≥ {TARGET_RECALL:.0%})",
        "",
        "## Held-out test set",
        "",
        "| Model | ROC-AUC | PR-AUC | Brier | ECE | Recall | Precision | Specificity |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in metrics["test"].items():
        t = r["at_chosen_threshold"]
        lines.append(
            f"| {name} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['brier']:.3f} | "
            f"{r['ece']:.3f} | {t['recall']:.3f} | {t['precision']:.3f} | "
            f"{t['specificity']:.3f} |"
        )
    ci = metrics["test_ci_selected"]
    lines += [
        "",
        "Recall, precision and specificity are at each model's own threshold, picked "
        f"the same way (out-of-fold recall ≥ {TARGET_RECALL:.0%}).",
        "",
        f"95% bootstrap CI for the selected model: ROC-AUC {ci['roc_auc']}, "
        f"recall {ci['recall']}, precision {ci['precision']}.",
        "",
        "## Leave one hospital out (selected model)",
        "",
        "| Held-out site | n | Positives | ROC-AUC | Recall | Precision |",
        "|---|---|---|---|---|---|",
    ]
    for site, r in metrics["leave_one_site_out"].items():
        auc = f"{r['roc_auc']:.3f}" if "roc_auc" in r else "n/a"
        lines.append(
            f"| {site} | {r['n']} | {r['positives']} | {auc} | "
            f"{r['recall']:.3f} | {r['precision']:.3f} |"
        )
    lines += [
        "",
        "## Test set by sex (selected model)",
        "",
        "| Group | n | Positives | ROC-AUC | Recall | Precision |",
        "|---|---|---|---|---|---|",
    ]
    for group, r in metrics["test_by_sex"].items():
        auc = f"{r['roc_auc']:.3f}" if "roc_auc" in r else "n/a"
        lines.append(
            f"| {group} | {r['n']} | {r['positives']} | {auc} | "
            f"{r['recall']:.3f} | {r['precision']:.3f} |"
        )
    lines += ["", "## Mean |SHAP| (log-odds), test set", ""]
    for f, v in metrics["shap_mean_abs"].items():
        lines.append(f"- {FEATURE_LABELS[f]}: {v:.3f}")
    (REPORTS_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    start = time.time()
    for d in (MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    train, test = split(df)
    x_tr, y_tr = train[FEATURES], train[TARGET]
    x_te, y_te = test[FEATURES], test[TARGET]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    specs = model_specs()
    fitted: dict[str, Pipeline] = {}
    cv_results: dict[str, dict[str, Any]] = {}
    oof: dict[str, np.ndarray] = {}
    for spec in specs:
        model = tune(spec, x_tr, y_tr, cv)
        # Fresh CV with the tuned settings. Slightly optimistic because the grid was
        # searched on the same folds; the untouched test set is the honest number.
        scores = cross_val_score(clone(model), x_tr, y_tr, cv=cv, scoring="roc_auc")
        oof[spec.name] = cross_val_predict(clone(model), x_tr, y_tr, cv=cv, method="predict_proba")[
            :, 1
        ]
        cv_results[spec.name] = {
            "cv_roc_auc_mean": round(float(scores.mean()), 4),
            "cv_roc_auc_se": round(float(scores.std(ddof=1) / np.sqrt(len(scores))), 4),
            "cv_roc_auc_folds": [round(float(s), 4) for s in scores],
            "params": {k: v for k, v in model.get_params().items() if k in spec.grid},
        }
        fitted[spec.name] = model
        log.info("%s CV ROC-AUC %.3f", spec.name, scores.mean())

    selected = select_model(cv_results, specs)
    thresholds = {
        name: (0.5 if name == "baseline_prior" else choose_threshold(y_tr, p, TARGET_RECALL))
        for name, p in oof.items()
    }
    threshold = thresholds[selected]
    log.info("Selected %s, threshold %.3f", selected, threshold)

    test_probs = {name: m.predict_proba(x_te)[:, 1] for name, m in fitted.items()}
    test_metrics = {
        name: {
            **ranking_metrics(y_te, p),
            "at_chosen_threshold": threshold_metrics(y_te, p, thresholds[name]),
            "at_0.5": threshold_metrics(y_te, p, 0.5),
        }
        for name, p in test_probs.items()
    }

    model = fitted[selected]
    background = x_tr.sample(n=min(100, len(x_tr)), random_state=RANDOM_STATE)
    explainer = make_explainer(model, background)
    shap_exp = explain(explainer, x_te)

    plot_roc(test_probs, y_te)
    plot_calibration(test_probs, y_te)
    plot_threshold(y_tr, oof[selected], threshold, selected)
    shap_mean_abs = plot_shap(shap_exp)

    metrics: dict[str, Any] = {
        "data": data_report(df),
        "split": {
            "train": len(train),
            "test": len(test),
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
        },
        "cv": cv_results,
        "selected_model": selected,
        "selection_rule": "simplest model within 1 SE of the best mean CV ROC-AUC",
        "threshold": round(threshold, 4),
        "threshold_rule": f"highest threshold with out-of-fold recall >= {TARGET_RECALL}",
        "oof_at_threshold": threshold_metrics(y_tr, oof[selected], threshold),
        "test": test_metrics,
        "test_ci_selected": bootstrap_ci(y_te, test_probs[selected], threshold),
        "leave_one_site_out": leave_one_site_out(model, df, threshold),
        "test_by_sex": by_sex(test, test_probs[selected], threshold),
        "shap_mean_abs": shap_mean_abs,
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
        },
    }

    artifact = {
        "pipeline": model,
        "threshold": threshold,
        "model_name": selected,
        "features": FEATURES,
        "background": background,
        "versions": metrics["versions"],
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_markdown(metrics)
    log.info("Done in %.1fs. Model: %s", time.time() - start, MODEL_PATH)
    return metrics


if __name__ == "__main__":
    main()
