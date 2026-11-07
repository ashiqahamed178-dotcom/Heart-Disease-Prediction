"""SHAP explanations in the original feature space, in log-odds units.

The explainer wraps the whole pipeline (imputation, encoding, model), so each of the
ten input fields gets one contribution, whichever model was chosen. Contributions add
up to the patient's log-odds minus the average log-odds of the background sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from heartrisk.config import FEATURES
from heartrisk.evaluate import logit


def _log_odds_fn(pipeline: Pipeline) -> Any:
    def f(x: Any) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(x, dtype=float), columns=FEATURES)
        proba = pipeline.predict_proba(frame)[:, 1]
        return logit(proba)

    return f


def make_explainer(pipeline: Pipeline, background: pd.DataFrame) -> shap.Explainer:
    masker = shap.maskers.Independent(background[FEATURES].to_numpy(dtype=float))
    return shap.explainers.Permutation(
        _log_odds_fn(pipeline), masker, feature_names=FEATURES, seed=0
    )


def explain(
    explainer: shap.Explainer, rows: pd.DataFrame, max_evals: int = 500
) -> shap.Explanation:
    return explainer(rows[FEATURES].to_numpy(dtype=float), max_evals=max_evals, silent=True)


def contributions(explanation: shap.Explanation, index: int = 0) -> pd.DataFrame:
    """One row per feature, sorted by absolute effect, for display."""
    values = np.asarray(explanation.values[index], dtype=float)
    data = np.asarray(explanation.data[index], dtype=float)
    out = pd.DataFrame({"feature": FEATURES, "value": data, "shap_log_odds": values})
    return out.reindex(out["shap_log_odds"].abs().sort_values(ascending=False).index)
