"""Load the saved model and score one patient, with a per-feature explanation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from heartrisk.config import FEATURE_LABELS, FEATURES, MODEL_PATH
from heartrisk.explain import contributions, explain, make_explainer

DISCLAIMER = (
    "Educational project only. Not a medical device and not clinically validated. "
    "Do not use it to diagnose, treat or make any health decision. "
    "If you have symptoms, see a doctor."
)


class PatientInput(BaseModel):
    """One patient's values. Ranges are wide but rule out obvious typos."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        description="Chest pain: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic"
    )
    trestbps: float | None = Field(
        default=None, ge=60, le=250, description="Resting blood pressure, mm Hg"
    )
    chol: float | None = Field(default=None, ge=80, le=700, description="Cholesterol, mg/dl")
    fbs: Literal[0, 1] | None = Field(default=None, description="Fasting sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] | None = Field(
        default=None, description="Resting ECG: 0 normal, 1 ST-T abnormality, 2 LVH"
    )
    thalach: float | None = Field(
        default=None, ge=50, le=230, description="Maximum heart rate reached"
    )
    exang: Literal[0, 1] | None = Field(default=None, description="Exercise-induced angina")
    oldpeak: float | None = Field(
        default=None, ge=-3, le=7, description="ST depression induced by exercise"
    )

    def to_frame(self) -> pd.DataFrame:
        row = {f: (np.nan if getattr(self, f) is None else getattr(self, f)) for f in FEATURES}
        return pd.DataFrame([row], columns=FEATURES).astype(float)


@dataclass(frozen=True)
class Prediction:
    probability: float
    high_risk: bool
    threshold: float
    model_name: str
    contributions: list[dict[str, Any]]
    base_log_odds: float
    disclaimer: str = DISCLAIMER


class RiskModel:
    def __init__(self, artifact: dict[str, Any]) -> None:
        self.pipeline = artifact["pipeline"]
        self.threshold: float = float(artifact["threshold"])
        self.model_name: str = artifact["model_name"]
        self.background: pd.DataFrame = artifact["background"]
        self._explainer = make_explainer(self.pipeline, self.background)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> RiskModel:
        if not path.exists():
            raise FileNotFoundError(f"No model at {path}. Run `python -m heartrisk.train`.")
        return cls(joblib.load(path))

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        proba: np.ndarray = self.pipeline.predict_proba(frame[FEATURES])[:, 1]
        return proba

    def predict(self, patient: PatientInput) -> Prediction:
        frame = patient.to_frame()
        prob = float(self.probability(frame)[0])
        exp = explain(self._explainer, frame)
        table = contributions(exp)
        return Prediction(
            probability=round(prob, 4),
            high_risk=prob >= self.threshold,
            threshold=round(self.threshold, 4),
            model_name=self.model_name,
            base_log_odds=round(float(np.asarray(exp.base_values).ravel()[0]), 4),
            contributions=[
                {
                    "feature": r.feature,
                    "label": FEATURE_LABELS[r.feature],
                    "value": None if np.isnan(r.value) else float(r.value),
                    "shap_log_odds": round(float(r.shap_log_odds), 4),
                }
                for r in table.itertuples()
            ],
        )


@lru_cache(maxsize=1)
def get_model() -> RiskModel:
    return RiskModel.load()
