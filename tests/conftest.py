from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heartrisk.config import FEATURES, TARGET


@pytest.fixture
def synthetic() -> pd.DataFrame:
    """Fake patients with a known signal and some missing values. No network needed."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "age": rng.integers(30, 77, n).astype(float),
            "trestbps": rng.normal(130, 18, n),
            "chol": rng.normal(230, 50, n),
            "thalach": rng.normal(140, 25, n),
            "oldpeak": rng.gamma(1.0, 1.0, n),
            "sex": rng.integers(0, 2, n).astype(float),
            "fbs": rng.integers(0, 2, n).astype(float),
            "exang": rng.integers(0, 2, n).astype(float),
            "cp": rng.integers(1, 5, n).astype(float),
            "restecg": rng.integers(0, 3, n).astype(float),
        }
    )
    signal = 0.08 * (df.age - 54) + 2.0 * df.exang + 1.5 * (df.cp == 4) + 1.0 * df.oldpeak - 1.8
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-signal))).astype(int)
    df.loc[rng.choice(n, 30, replace=False), "chol"] = np.nan
    df.loc[rng.choice(n, 10, replace=False), "thalach"] = np.nan
    df["site"] = rng.choice(["cleveland", "hungarian", "switzerland", "va"], n)
    return df[[*FEATURES, TARGET, "site"]]


@pytest.fixture
def patient_kwargs() -> dict[str, object]:
    return {
        "age": 61,
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "chol": 260,
        "fbs": 0,
        "restecg": 0,
        "thalach": 120,
        "exang": 1,
        "oldpeak": 2.0,
    }
