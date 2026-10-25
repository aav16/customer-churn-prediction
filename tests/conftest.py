"""Fixtures compartidas: un dataset sintético con el mismo esquema que Telco.

Los tests no dependen del CSV real (que no se versiona), así funcionan en un clon limpio.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.train import build_model_pipeline  # noqa: E402

CATEGORY_LEVELS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"],
}


def make_raw_data(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Genera datos crudos con los mismos problemas que el original (TotalCharges como texto)."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    tenure[:3] = 0  # clientes nuevos -> TotalCharges en blanco
    monthly = rng.uniform(18, 120, n).round(2)
    total = (tenure * monthly).round(2).astype(str).astype(object)
    total[tenure == 0] = " "
    df = pd.DataFrame({
        "customerID": [f"{i:04d}-TEST" for i in range(n)],
        "SeniorCitizen": rng.integers(0, 2, n),
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        **{col: rng.choice(levels, n) for col, levels in CATEGORY_LEVELS.items()},
    })
    # Objetivo con señal (contrato mensual y poca antigüedad -> más abandono).
    logit = -1.0 + 1.5 * (df["Contract"] == "Month-to-month") - 0.03 * df["tenure"]
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df[[config.ID_COL, *config.FEATURES, config.TARGET_COL]]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_data()


@pytest.fixture
def trained_pipelines(raw_df):
    """Pipelines pequeños entrenados sobre datos sintéticos (LR y XGBoost)."""
    from src.data_preprocessing import clean_data, split_features_target

    X, y = split_features_target(clean_data(raw_df))
    pipelines = {}
    for key in ["logreg", "xgboost"]:
        pipe = build_model_pipeline(key, "none", pos_weight=1.0)
        if key == "xgboost":
            pipe.set_params(model__n_estimators=20, model__max_depth=2)
        pipelines[key] = pipe.fit(X, y)
    return pipelines
