"""Tests de modelo, predicción y explicabilidad."""

import joblib
import numpy as np
import pytest
from scipy.special import logit

from src import config
from src.data_preprocessing import clean_data, split_features_target
from src.explain import compute_shap, top_drivers
from src.predict import assign_risk_level, load_model, predict_customers
from src.train import build_model_pipeline, build_risk_bands, select_threshold

METADATA = {"threshold": 0.4, "risk_bands": build_risk_bands(0.4)}


@pytest.mark.parametrize("key", ["logreg", "xgboost"])
def test_probabilities_are_valid(trained_pipelines, raw_df, key):
    X, _ = split_features_target(clean_data(raw_df))
    proba = trained_pipelines[key].predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_predictions_keep_customer_ids(trained_pipelines, raw_df):
    pipe = trained_pipelines["logreg"]
    preds = predict_customers(raw_df, pipe, METADATA)
    assert len(preds) == len(raw_df)
    assert set(preds[config.ID_COL]) == set(raw_df[config.ID_COL])
    # Cada ID conserva SU probabilidad (independiente de la reordenación).
    X, _ = split_features_target(clean_data(raw_df))
    expected = dict(zip(raw_df[config.ID_COL], pipe.predict_proba(X)[:, 1].round(4)))
    assert all(expected[i] == p for i, p in zip(preds[config.ID_COL], preds["churn_probability"]))


def test_predictions_sorted_and_thresholded(trained_pipelines, raw_df):
    preds = predict_customers(raw_df, trained_pipelines["xgboost"], METADATA)
    assert preds["churn_probability"].is_monotonic_decreasing
    assert list(preds["risk_rank"]) == list(range(1, len(preds) + 1))
    flagged = preds["churn_probability"] >= METADATA["threshold"]
    assert (preds.loc[flagged, "predicted_churn"] == "Yes").all()
    assert (preds.loc[~flagged, "predicted_churn"] == "No").all()


def test_target_never_used_as_feature(trained_pipelines, raw_df):
    """Cambiar la etiqueta real no debe alterar ninguna probabilidad."""
    pipe = trained_pipelines["logreg"]
    flipped = raw_df.assign(Churn=raw_df["Churn"].map({"Yes": "No", "No": "Yes"}))
    a = predict_customers(raw_df, pipe, METADATA).set_index(config.ID_COL)["churn_probability"]
    b = predict_customers(flipped, pipe, METADATA).set_index(config.ID_COL)["churn_probability"]
    assert a.sort_index().equals(b.sort_index())


def test_predict_new_customers_without_target(trained_pipelines, raw_df):
    preds = predict_customers(raw_df.drop(columns="Churn"), trained_pipelines["logreg"], METADATA)
    assert "actual_churn" not in preds.columns


def test_risk_levels_aligned_with_threshold():
    levels = assign_risk_level(np.array([0.0, 0.32, 0.33, 0.59, 0.60, 1.0]), build_risk_bands(0.33))
    assert list(levels) == ["Low", "Low", "Medium", "Medium", "High", "High"]


def test_risk_level_consistent_with_predicted_churn(trained_pipelines, raw_df):
    preds = predict_customers(raw_df, trained_pipelines["logreg"], METADATA)
    assert ((preds["risk_level"] != "Low") == (preds["predicted_churn"] == "Yes")).all()


def test_select_threshold_in_range():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    t, score = select_threshold(y, proba, beta=1.0)
    assert 0.05 <= t <= 0.95 and 0 < score <= 1


def test_smote_pipeline_trains(raw_df):
    X, y = split_features_target(clean_data(raw_df))
    pipe = build_model_pipeline("logreg", "smote", pos_weight=1.0).fit(X, y)
    assert "smote" in pipe.named_steps
    assert pipe.predict_proba(X).shape == (len(X), 2)


@pytest.mark.parametrize("key", ["logreg", "xgboost"])
def test_shap_additivity_and_drivers(trained_pipelines, raw_df, key):
    pipe = trained_pipelines[key]
    X, _ = split_features_target(clean_data(raw_df))
    result = compute_shap(pipe, X.head(30), background=X.sample(50, random_state=0))
    proba = pipe.predict_proba(X.head(30))[:, 1]
    # Las contribuciones agregadas por variable original suman lo mismo que las transformadas.
    np.testing.assert_allclose(result.original.sum(axis=1), result.values.sum(axis=1), atol=1e-8)
    np.testing.assert_allclose(result.base_value + result.values.sum(axis=1), logit(proba), atol=1e-3)
    inc, dec = top_drivers(result, X.head(30))
    assert len(inc) == len(dec) == 30


def test_saved_model_roundtrip(trained_pipelines, raw_df, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(trained_pipelines["xgboost"], path)
    loaded = joblib.load(path)
    X, _ = split_features_target(clean_data(raw_df))
    np.testing.assert_allclose(loaded.predict_proba(X), trained_pipelines["xgboost"].predict_proba(X))


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="main.py"):
        load_model(tmp_path / "missing.joblib")


def test_project_model_loads_if_trained(raw_df):
    if not config.MODEL_PATH.exists():
        pytest.skip("No trained model yet (run `python main.py`).")
    pipe, metadata = load_model()
    assert 0 < metadata["threshold"] < 1
    X, _ = split_features_target(clean_data(raw_df))
    proba = pipe.predict_proba(X)[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))
