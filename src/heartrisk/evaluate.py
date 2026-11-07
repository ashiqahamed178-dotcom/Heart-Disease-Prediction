"""Metrics, threshold choice and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: ArrayLike, proba: ArrayLike, n_bins: int = 10) -> float:
    """Weighted mean gap between predicted and observed rates over equal-width bins."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def threshold_metrics(y_true: ArrayLike, proba: ArrayLike, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def ranking_metrics(y_true: ArrayLike, proba: ArrayLike) -> dict[str, float]:
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "ece": round(expected_calibration_error(y_true, proba), 4),
    }


def choose_threshold(y_true: ArrayLike, proba: ArrayLike, target_recall: float) -> float:
    """Highest threshold whose recall is still at least ``target_recall``.

    Highest, because among thresholds that catch enough cases it raises the fewest
    false alarms. Run on out-of-fold training predictions, never on the test set.
    """
    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    positives = y.sum()
    if positives == 0:
        raise ValueError("Need at least one positive case to pick a threshold")
    # Recall at threshold t is (# positives with p >= t) / positives.
    pos_scores = np.sort(p[y == 1])[::-1]
    needed = int(np.ceil(target_recall * positives))
    return float(pos_scores[needed - 1])


def bootstrap_ci(
    y_true: ArrayLike,
    proba: ArrayLike,
    threshold: float,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, list[float]]:
    """95% percentile intervals for ROC-AUC, recall and precision on a fixed set."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    stats: dict[str, list[float]] = {"roc_auc": [], "recall": [], "precision": []}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        pred = (pb >= threshold).astype(int)
        stats["roc_auc"].append(roc_auc_score(yb, pb))
        stats["recall"].append(recall_score(yb, pred, zero_division=0))
        stats["precision"].append(precision_score(yb, pred, zero_division=0))
    return {
        k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
        for k, v in stats.items()
    }


def logit(p: NDArray[np.float64], eps: float = 1e-6) -> NDArray[np.float64]:
    p = np.clip(p, eps, 1 - eps)
    out: NDArray[np.float64] = np.log(p / (1 - p))
    return out
