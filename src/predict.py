"""Predicción de abandono: "Which customers are most likely to churn?".

Genera una tabla por cliente con probabilidad, clasificación según el umbral
seleccionado, nivel de riesgo y principales factores SHAP, ordenada de mayor a
menor riesgo.

Uso:
    python -m src.predict                       # clientes del conjunto de test (histórico)
    python -m src.predict --input nuevos.csv --output reports/nuevos_pred.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline

from src import config
from src.data_preprocessing import clean_data
from src.evaluate import load_test_set
from src.explain import compute_shap, top_drivers
from src.utils import get_logger, load_json

logger = get_logger(__name__)


def load_model(path: Path = config.MODEL_PATH) -> tuple[Pipeline, dict]:
    """Carga el pipeline serializado y sus metadatos (umbral, bandas de riesgo...)."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `python main.py` first.")
    return joblib.load(path), load_json(config.METADATA_PATH)


def assign_risk_level(proba: np.ndarray, bands: list) -> np.ndarray:
    """Asigna 'Low' / 'Medium' / 'High' según bandas [(límite inferior inclusivo, etiqueta)]."""
    lower_bounds = [b for b, _ in bands]
    labels = np.array([label for _, label in bands])
    return labels[np.searchsorted(lower_bounds, proba, side="right") - 1]


def predict_customers(
    raw_df: pd.DataFrame,
    pipeline: Pipeline,
    metadata: dict,
    background: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Devuelve las predicciones ordenadas por probabilidad de abandono (descendente).

    La columna `Churn`, si existe, se conserva solo como `actual_churn` para análisis:
    nunca entra en el modelo (el pipeline únicamente usa `config.FEATURES`).
    """
    has_target = config.TARGET_COL in raw_df.columns
    df = clean_data(raw_df, require_target=has_target)
    X = df[config.FEATURES]

    proba = pipeline.predict_proba(X)[:, 1]
    threshold = metadata["threshold"]
    out = pd.DataFrame({
        config.ID_COL: df[config.ID_COL].to_numpy(),
        "churn_probability": proba.round(4),
        "predicted_churn": np.where(proba >= threshold, "Yes", "No"),
        "risk_level": assign_risk_level(proba, metadata["risk_bands"]),
    })

    if background is not None:
        result = compute_shap(pipeline, X, background)
        out["top_risk_drivers"], out["top_protective_factors"] = top_drivers(result, X)
    if has_target:
        out["actual_churn"] = df[config.TARGET_COL].map({1: "Yes", 0: "No"}).to_numpy()

    out = out.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    out.insert(0, "risk_rank", np.arange(1, len(out) + 1))
    return out


def run_test_set_predictions(output: Path = config.PREDICTIONS_PATH) -> pd.DataFrame:
    """Predicciones sobre el conjunto de test (clientes HISTÓRICOS no vistos en entrenamiento)."""
    pipeline, metadata = load_model()
    background = joblib.load(config.MODELS_DIR / "shap_background.joblib")
    test_df = load_test_set(metadata)
    # Devolvemos Churn a su formato original para reutilizar el mismo flujo que con datos nuevos.
    raw = test_df.assign(**{config.TARGET_COL: test_df[config.TARGET_COL].map({1: "Yes", 0: "No"})})

    preds = predict_customers(raw, pipeline, metadata, background)
    preds.insert(1, "data_split", "test (held-out historical customers)")
    output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output, index=False)
    logger.info("Saved %d test-set predictions to %s (%d flagged as churn at threshold %.2f)",
                len(preds), output.name, (preds["predicted_churn"] == "Yes").sum(), metadata["threshold"])
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Score customers by churn risk.")
    parser.add_argument("--input", type=Path, help="CSV with the Telco schema (Churn optional).")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    parser.add_argument("--no-explain", action="store_true", help="Skip SHAP drivers (faster).")
    args = parser.parse_args()

    if args.input is None:
        run_test_set_predictions(args.output or config.PREDICTIONS_PATH)
        return

    pipeline, metadata = load_model()
    background = None if args.no_explain else joblib.load(config.MODELS_DIR / "shap_background.joblib")
    preds = predict_customers(pd.read_csv(args.input), pipeline, metadata, background)
    output = args.output or config.REPORTS_DIR / f"{args.input.stem}_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output, index=False)
    logger.info("Saved %d predictions to %s", len(preds), output)


if __name__ == "__main__":
    main()
