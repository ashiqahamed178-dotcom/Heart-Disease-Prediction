"""Model pipelines and their small hyperparameter grids."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from heartrisk.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)


def build_preprocessor() -> ColumnTransformer:
    """Impute and encode. Fitted inside each CV fold, so no test data leaks in."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("bin", binary, BINARY_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


@dataclass(frozen=True)
class ModelSpec:
    """A named estimator and the grid searched for it. ``rank`` orders by simplicity."""

    name: str
    estimator: Any
    grid: dict[str, list[Any]] = field(default_factory=dict)
    rank: int = 0


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="baseline_prior",
            estimator=DummyClassifier(strategy="prior"),
            rank=0,
        ),
        ModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
            rank=1,
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1
            ),
            grid={
                "model__max_depth": [3, 5, None],
                "model__min_samples_leaf": [1, 5, 10],
            },
            rank=2,
        ),
        ModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                n_estimators=300,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            grid={
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
            },
            rank=3,
        ),
    ]


def build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])
