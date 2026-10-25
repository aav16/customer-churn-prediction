"""Carga, limpieza, división y preprocesamiento de los datos.

Principio clave: la limpieza determinista (conversión de tipos, reglas de negocio)
se hace aquí con pandas, y todo lo que *aprende parámetros* (imputación, escalado,
codificación) vive dentro de un `ColumnTransformer` que solo se ajusta con datos de
entrenamiento. Así se evita el data leakage.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def load_raw_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Lee el CSV original sin modificarlo."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for download instructions."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d rows x %d columns from %s", *df.shape, path.name)
    return df


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    """Comprueba que están todas las columnas necesarias antes de procesar."""
    required = [config.ID_COL, *config.FEATURES]
    if require_target:
        required.append(config.TARGET_COL)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")


def clean_data(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Limpieza determinista (no aprende nada de los datos, por lo que no hay leakage).

    - `TotalCharges` llega como texto; los valores en blanco corresponden a clientes con
      `tenure == 0` que aún no han recibido factura, por lo que su importe acumulado es 0.
    - `Churn` se convierte de "Yes"/"No" a 1/0.
    """
    validate_schema(df, require_target=require_target)
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    new_customers = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[new_customers, "TotalCharges"] = 0.0
    # Otros nulos (no esperados) se dejan como NaN: los imputa el pipeline.

    for col in config.NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in config.CATEGORICAL_FEATURES:
        # Convertimos a object (con NaN estándar) y eliminamos espacios sobrantes.
        df[col] = df[col].astype("object").map(lambda v: v.strip() if isinstance(v, str) else v)

    if require_target and config.TARGET_COL in df.columns:
        target = df[config.TARGET_COL].astype(str).str.strip().map({"Yes": 1, "No": 0})
        if target.isna().any():
            raise ValueError("Target column 'Churn' must only contain 'Yes'/'No'.")
        df[config.TARGET_COL] = target.astype(int)

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa predictores y objetivo. El identificador nunca es un predictor."""
    return df[config.FEATURES], df[config.TARGET_COL]


def train_test_split_stratified(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """División estratificada train/test. El test se reserva para la evaluación final."""
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df[config.TARGET_COL],
        random_state=config.RANDOM_STATE,
    )
    logger.info(
        "Train: %d rows (churn %.1f%%) | Test: %d rows (churn %.1f%%)",
        len(train_df),
        100 * train_df[config.TARGET_COL].mean(),
        len(test_df),
        100 * test_df[config.TARGET_COL].mean(),
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: imputación + escalado (numéricas) y one-hot (categóricas).

    El escalado beneficia a la regresión logística; para XGBoost es irrelevante pero
    inocuo, y usar el mismo preprocesador simplifica la comparación entre modelos.
    """
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def get_feature_mapping(preprocessor: ColumnTransformer) -> pd.DataFrame:
    """Relaciona cada columna transformada con su variable original.

    Ejemplo: `Contract_Month-to-month` -> original `Contract`, etiqueta legible
    `Contract = Month-to-month`. Imprescindible para agregar valores SHAP.
    """
    rows = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            rows += [(col, col, col) for col in columns]
        elif name == "cat":
            encoder = transformer.named_steps["onehot"]
            for col, categories in zip(columns, encoder.categories_):
                rows += [(f"{col}_{cat}", col, f"{col} = {cat}") for cat in categories]
    mapping = pd.DataFrame(rows, columns=["transformed", "original", "label"])

    expected = list(preprocessor.get_feature_names_out())
    if mapping["transformed"].tolist() != expected:
        raise RuntimeError("Feature mapping does not match the preprocessor output order.")
    return mapping
