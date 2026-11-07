"""Small FastAPI service. Run with ``uvicorn heartrisk.api:app``."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException

from heartrisk import __version__
from heartrisk.predict import DISCLAIMER, PatientInput, get_model

app = FastAPI(
    title="Heart disease risk (educational)",
    version=__version__,
    description=DISCLAIMER,
)


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "model": model.model_name, "threshold": round(model.threshold, 4)}


@app.post("/predict")
def predict(patient: PatientInput) -> dict[str, Any]:
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return asdict(model.predict(patient))
