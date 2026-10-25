"""Tests de carga, limpieza y preprocesamiento."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_preprocessing import (
    build_preprocessor,
    clean_data,
    get_feature_mapping,
    load_raw_data,
    split_features_target,
    train_test_split_stratified,
)


def test_total_charges_converted_to_float(raw_df):
    assert not pd.api.types.is_numeric_dtype(raw_df["TotalCharges"])  # llega como texto
    clean = clean_data(raw_df)
    assert clean["TotalCharges"].dtype == float


def test_blank_total_charges_for_new_customers_become_zero(raw_df):
    clean = clean_data(raw_df)
    new = clean["tenure"] == 0
    assert new.sum() >= 3
    assert (clean.loc[new, "TotalCharges"] == 0).all()
    assert clean[config.FEATURES].isna().sum().sum() == 0


def test_target_is_binary_int(raw_df):
    clean = clean_data(raw_df)
    assert set(clean[config.TARGET_COL].unique()) <= {0, 1}
    assert clean[config.TARGET_COL].dtype.kind == "i"


def test_invalid_target_raises(raw_df):
    raw_df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Churn"):
        clean_data(raw_df)


def test_missing_columns_raise(raw_df):
    with pytest.raises(ValueError, match="missing required columns"):
        clean_data(raw_df.drop(columns=["Contract", "tenure"]))


def test_clean_data_does_not_modify_input(raw_df):
    before = raw_df.copy()
    clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_clean_data_without_target_for_new_customers(raw_df):
    clean = clean_data(raw_df.drop(columns="Churn"), require_target=False)
    assert config.TARGET_COL not in clean.columns
    assert len(clean) == len(raw_df)


def test_split_excludes_id_and_target(raw_df):
    X, y = split_features_target(clean_data(raw_df))
    assert list(X.columns) == config.FEATURES
    assert config.ID_COL not in X.columns and config.TARGET_COL not in X.columns
    assert len(X) == len(y)


def test_stratified_split_preserves_churn_rate(raw_df):
    df = clean_data(pd.concat([raw_df] * 5, ignore_index=True))
    df[config.ID_COL] = [f"id{i}" for i in range(len(df))]
    train, test = train_test_split_stratified(df)
    assert len(test) == pytest.approx(config.TEST_SIZE * len(df), abs=1)
    assert set(train[config.ID_COL]).isdisjoint(test[config.ID_COL])
    assert abs(train.Churn.mean() - test.Churn.mean()) < 0.02


def test_preprocessor_output_and_mapping(raw_df):
    X, _ = split_features_target(clean_data(raw_df))
    pre = build_preprocessor().fit(X)
    Xt = pre.transform(X)
    mapping = get_feature_mapping(pre)
    assert Xt.shape == (len(X), len(mapping))
    assert not np.isnan(Xt).any()
    assert set(mapping["original"]) == set(config.FEATURES)
    # Las numéricas quedan estandarizadas (media ~0) con parámetros aprendidos solo de X.
    num_idx = [i for i, o in enumerate(mapping["original"]) if o in config.NUMERIC_FEATURES]
    assert np.allclose(Xt[:, num_idx].mean(axis=0), 0, atol=1e-8)


def test_preprocessor_handles_unseen_category_and_nulls(raw_df):
    X, _ = split_features_target(clean_data(raw_df))
    pre = build_preprocessor().fit(X)
    new = X.head(2).copy()
    new.loc[new.index[0], "PaymentMethod"] = "Crypto"  # categoría no vista
    new.loc[new.index[1], "MonthlyCharges"] = np.nan  # nulo inesperado
    out = pre.transform(new)
    assert not np.isnan(out).any()


def test_real_dataset_schema_if_available():
    if not config.RAW_DATA_PATH.exists():
        pytest.skip("Real dataset not available (it is not versioned).")
    clean = clean_data(load_raw_data())
    assert clean.shape == (7043, 21)
    assert clean[config.ID_COL].is_unique
    assert clean.isna().sum().sum() == 0
