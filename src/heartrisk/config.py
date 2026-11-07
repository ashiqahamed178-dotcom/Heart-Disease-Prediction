"""Paths, column definitions and fixed settings used across the project."""

from __future__ import annotations

from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
DATA_DIR: Final = ROOT / "data" / "raw"
MODELS_DIR: Final = ROOT / "models"
REPORTS_DIR: Final = ROOT / "reports"
FIGURES_DIR: Final = REPORTS_DIR / "figures"

MODEL_PATH: Final = MODELS_DIR / "model.joblib"
METRICS_PATH: Final = MODELS_DIR / "metrics.json"

RANDOM_STATE: Final = 42
TEST_SIZE: Final = 0.2
CV_FOLDS: Final = 5

# Recall we want the chosen threshold to reach on out-of-fold training predictions.
# Missing a sick patient costs more than a false alarm, so recall comes first.
TARGET_RECALL: Final = 0.90

# The 14 columns of the UCI "processed" files, in file order.
RAW_COLUMNS: Final = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

SITES: Final = ("cleveland", "hungarian", "switzerland", "va")

# Dropped: mostly missing outside Cleveland (see README "Data decisions").
DROPPED_FEATURES: Final = ("slope", "ca", "thal")

NUMERIC_FEATURES: Final = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_FEATURES: Final = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES: Final = ["cp", "restecg"]
FEATURES: Final = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
TARGET: Final = "target"

FEATURE_LABELS: Final = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate reached",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
}

CP_LABELS: Final = {
    1: "Typical angina",
    2: "Atypical angina",
    3: "Non-anginal pain",
    4: "Asymptomatic",
}
RESTECG_LABELS: Final = {
    0: "Normal",
    1: "ST-T wave abnormality",
    2: "Left ventricular hypertrophy",
}
