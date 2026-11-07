"""Tests against the committed model in models/. Skipped if it has not been trained."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from heartrisk.config import METRICS_PATH, MODEL_PATH
from heartrisk.predict import PatientInput, RiskModel

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained")


@pytest.fixture(scope="module")
def model() -> RiskModel:
    return RiskModel.load()


def test_high_risk_profile_scores_higher(model: RiskModel, patient_kwargs: dict[str, Any]) -> None:
    risky = model.predict(PatientInput(**patient_kwargs))
    healthy = model.predict(
        PatientInput(
            age=35,
            sex=0,
            cp=3,
            trestbps=115,
            chol=180,
            fbs=0,
            restecg=0,
            thalach=180,
            exang=0,
            oldpeak=0.0,
        )
    )
    assert 0 <= healthy.probability < risky.probability <= 1
    assert risky.high_risk
    assert not healthy.high_risk
    assert len(risky.contributions) == 10


def test_missing_optional_fields_still_predict(model: RiskModel) -> None:
    pred = model.predict(PatientInput(age=50, sex=1, cp=2))
    assert 0 <= pred.probability <= 1
    assert any(c["value"] is None for c in pred.contributions)


@pytest.mark.parametrize("bad", [{"age": 5}, {"cp": 7}, {"chol": 0}, {"site": "va"}])
def test_input_validation_rejects_bad_values(
    patient_kwargs: dict[str, Any], bad: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        PatientInput(**{**patient_kwargs, **bad})


def test_saved_metrics_beat_baseline() -> None:
    m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    selected = m["test"][m["selected_model"]]
    assert selected["roc_auc"] > m["test"]["baseline_prior"]["roc_auc"] + 0.2
    assert m["oof_at_threshold"]["recall"] >= 0.9


def test_api_health_and_predict(patient_kwargs: dict[str, Any]) -> None:
    from heartrisk.api import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/predict", json=patient_kwargs)
    assert r.status_code == 200
    body = r.json()
    assert {"probability", "high_risk", "threshold", "contributions", "disclaimer"} <= set(body)
    assert client.post("/predict", json={**patient_kwargs, "age": 500}).status_code == 422
