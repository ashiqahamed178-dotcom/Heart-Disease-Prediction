"""Streamlit front end. Run with ``streamlit run app/streamlit_app.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # works without `pip install -e .` too
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from heartrisk.config import (  # noqa: E402
    CP_LABELS,
    FIGURES_DIR,
    METRICS_PATH,
    RESTECG_LABELS,
)
from heartrisk.predict import DISCLAIMER, PatientInput, Prediction, RiskModel  # noqa: E402

st.set_page_config(page_title="Heart disease risk (educational)", page_icon="🫀", layout="wide")


@st.cache_resource
def load_model() -> RiskModel:
    return RiskModel.load()


@st.cache_data
def load_metrics() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return data


def yes_no(label: str, key: str) -> int | None:
    choice = st.selectbox(label, ["No", "Yes", "Unknown"], key=key)
    return None if choice == "Unknown" else int(choice == "Yes")


def optional_number(
    label: str, key: str, lo: float, hi: float, default: float, step: float
) -> float | None:
    unknown = st.checkbox("Unknown", key=f"{key}_unknown")
    value = st.number_input(label, lo, hi, default, step, key=key, disabled=unknown)
    return None if unknown else float(value)


def patient_form() -> PatientInput:
    """Inputs update the estimate live, so there is no submit button."""
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", 18, 100, 55)
            sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
            cp = st.selectbox(
                "Chest pain type", list(CP_LABELS), format_func=CP_LABELS.__getitem__, index=3
            )
            exang = yes_no("Chest pain brought on by exercise?", "exang")
        with c2:
            trestbps = optional_number("Resting blood pressure (mm Hg)", "bp", 60, 250, 130, 1)
            chol = optional_number("Cholesterol (mg/dl)", "chol", 80, 700, 230, 1)
            fbs = yes_no("Fasting blood sugar above 120 mg/dl?", "fbs")
        with c3:
            thalach = optional_number("Max heart rate on exercise test", "hr", 50, 230, 140, 1)
            oldpeak = optional_number(
                "ST depression on exercise (oldpeak)", "op", -3.0, 7.0, 1.0, 0.1
            )
            ecg = st.selectbox(
                "Resting ECG",
                [*RESTECG_LABELS, "Unknown"],
                format_func=lambda v: v if v == "Unknown" else RESTECG_LABELS[v],
            )
    return PatientInput.model_validate(
        {
            "age": int(age),
            "sex": 1 if sex == "Male" else 0,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": None if ecg == "Unknown" else ecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
        }
    )


def display_value(feature: str, value: float | None) -> str:
    if value is None:
        return "unknown"
    if feature == "cp":
        return CP_LABELS[int(value)]
    if feature == "restecg":
        return RESTECG_LABELS[int(value)]
    if feature == "sex":
        return "male" if value == 1 else "female"
    if feature in ("fbs", "exang"):
        return "yes" if value == 1 else "no"
    return f"{value:g}"


def contribution_chart(pred: Prediction) -> Figure:
    rows = list(reversed(pred.contributions))
    labels = [f"{r['label']}: {display_value(r['feature'], r['value'])}" for r in rows]
    values = [r["shap_log_odds"] for r in rows]
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in values]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Effect on risk (log-odds). Red raises it, blue lowers it.")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def show_prediction(pred: Prediction) -> None:
    pct = pred.probability * 100
    left, right = st.columns([1, 2])
    with left:
        st.metric("Estimated probability of heart disease", f"{pct:.0f}%")
        if pred.high_risk:
            st.error(f"Above the screening threshold ({pred.threshold:.0%}): higher risk.")
        else:
            st.success(f"Below the screening threshold ({pred.threshold:.0%}).")
        st.caption(
            "The threshold is set low on purpose so the model misses fewer sick patients. "
            "That means more false alarms. Unknown fields are filled with typical training values."
        )
    with right:
        st.markdown("**Why this estimate** (SHAP, compared with an average training patient)")
        st.pyplot(contribution_chart(pred), clear_figure=True)


def about_page() -> None:
    st.title("How good is the model?")
    st.warning(f"**Not for medical use.** {DISCLAIMER}")
    m: dict[str, Any] = load_metrics()
    rows = [
        {
            "model": name,
            "ROC-AUC": r["roc_auc"],
            "Brier": r["brier"],
            "Recall": r["at_chosen_threshold"]["recall"],
            "Precision": r["at_chosen_threshold"]["precision"],
            "Specificity": r["at_chosen_threshold"]["specificity"],
        }
        for name, r in m["test"].items()
    ]
    st.markdown(
        f"Selected model: **{m['selected_model']}** (simplest model within one standard error "
        f"of the best cross-validated ROC-AUC). Threshold **{m['threshold']:.2f}**, chosen so "
        f"out-of-fold recall on the training data is at least 90%. "
        f"Held-out test set: {m['split']['test']} patients."
    )
    st.table(pd.DataFrame(rows).set_index("model").style.format(precision=3))
    st.markdown("**Leave one hospital out** (train on three hospitals, test on the fourth)")
    loso = [
        {
            "held-out site": site,
            "patients": r["n"],
            "ROC-AUC": r.get("roc_auc"),
            "Recall": r["recall"],
            "Precision": r["precision"],
        }
        for site, r in m["leave_one_site_out"].items()
    ]
    st.table(pd.DataFrame(loso).set_index("held-out site").style.format(precision=3, na_rep="n/a"))
    st.markdown("**Test set by sex** (small groups, so treat these as rough)")
    sex_rows = [
        {
            "group": group,
            "patients": r["n"],
            "ROC-AUC": r.get("roc_auc"),
            "Recall": r["recall"],
            "Precision": r["precision"],
        }
        for group, r in m.get("test_by_sex", {}).items()
    ]
    if sex_rows:
        st.table(pd.DataFrame(sex_rows).set_index("group").style.format(precision=3, na_rep="n/a"))
    c1, c2 = st.columns(2)
    c1.image(str(FIGURES_DIR / "roc_test.png"))
    c2.image(str(FIGURES_DIR / "calibration_test.png"))
    c1.image(str(FIGURES_DIR / "threshold_oof.png"))
    c2.image(str(FIGURES_DIR / "shap_beeswarm.png"))
    st.markdown(
        "**Limits.** 918 patients from four hospitals, collected in the 1980s, 79% men. "
        "Disease rates differ a lot by hospital (36% to 94%). Recall is much lower for women. "
        "Accuracy drops when a whole "
        "hospital is held out. Not validated on any current population."
    )


def estimate_page() -> None:
    st.title("Heart disease risk estimator")
    st.warning(f"**Not for medical use.** {DISCLAIMER}")
    show_prediction(load_model().predict(patient_form()))


st.navigation(
    [
        st.Page(estimate_page, title="Estimate risk", url_path="estimate", default=True),
        st.Page(about_page, title="Model performance", url_path="model"),
    ],
    position="top",
).run()
