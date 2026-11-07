from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from heartrisk.config import FEATURES, TARGET
from heartrisk.evaluate import logit
from heartrisk.explain import contributions, explain, make_explainer
from heartrisk.models import build_pipeline, model_specs
from heartrisk.train import select_model, split


def test_every_model_fits_with_missing_values(synthetic: pd.DataFrame) -> None:
    x, y = synthetic[FEATURES], synthetic[TARGET]
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    for spec in model_specs():
        scores = cross_val_score(build_pipeline(spec.estimator), x, y, cv=cv, scoring="roc_auc")
        floor = 0.45 if spec.name == "baseline_prior" else 0.65
        assert scores.mean() > floor, spec.name


def test_split_keeps_every_site_in_both_halves(synthetic: pd.DataFrame) -> None:
    train, test = split(synthetic)
    assert set(train.site) == set(test.site)
    assert len(test) == round(0.2 * len(synthetic))


def test_site_is_not_a_feature() -> None:
    assert "site" not in FEATURES


def test_select_model_prefers_simpler_within_one_se() -> None:
    specs = model_specs()
    cv = {
        "baseline_prior": {"cv_roc_auc_mean": 0.5, "cv_roc_auc_se": 0.0},
        "logistic_regression": {"cv_roc_auc_mean": 0.870, "cv_roc_auc_se": 0.01},
        "random_forest": {"cv_roc_auc_mean": 0.880, "cv_roc_auc_se": 0.015},
        "xgboost": {"cv_roc_auc_mean": 0.875, "cv_roc_auc_se": 0.01},
    }
    assert select_model(cv, specs) == "logistic_regression"
    cv["logistic_regression"]["cv_roc_auc_mean"] = 0.80
    assert select_model(cv, specs) == "random_forest"


def test_shap_values_add_up_to_log_odds(synthetic: pd.DataFrame) -> None:
    x, y = synthetic[FEATURES], synthetic[TARGET]
    pipe = build_pipeline(model_specs()[1].estimator).fit(x, y)
    explainer = make_explainer(pipe, x.sample(40, random_state=0))
    rows = x.head(3)
    exp = explain(explainer, rows)
    total = exp.base_values + exp.values.sum(axis=1)
    np.testing.assert_allclose(total, logit(pipe.predict_proba(rows)[:, 1]), atol=1e-6)
    table = contributions(exp)
    assert len(table) == len(FEATURES)
    assert table.shap_log_odds.abs().is_monotonic_decreasing
