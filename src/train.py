"""Entrenamiento, comparación de estrategias de desbalanceo y selección de modelo/umbral.

Flujo (todo sobre el conjunto de ENTRENAMIENTO; el test no se toca aquí):
1. Para cada modelo (Logistic Regression, XGBoost) y cada estrategia de desbalanceo
   (ninguna, ponderación de clases, SMOTE) -> GridSearchCV con validación cruzada estratificada.
2. Predicciones out-of-fold (OOF) del mejor estimador de cada combinación para medir
   precision/recall/falsos positivos a umbral 0.5.
3. Modelo final = combinación con mayor PR-AUC media en validación cruzada.
4. Umbral = el que maximiza F1 sobre las predicciones OOF (datos de validación, no test).
"""

import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, fbeta_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from src import config
from src.data_preprocessing import (
    build_preprocessor,
    clean_data,
    load_raw_data,
    split_features_target,
    train_test_split_stratified,
)
from src.utils import get_logger, save_json

logger = get_logger(__name__)

MODEL_NAMES = {"logreg": "Logistic Regression", "xgboost": "XGBoost"}
STRATEGIES = ["none", "weighted", "smote"]

# Búsqueda de hiperparámetros deliberadamente acotada: reproducible y rápida en CPU.
PARAM_GRIDS = {
    "logreg": {"model__C": [0.01, 0.1, 1.0, 10.0]},
    "xgboost": {
        "model__n_estimators": [100, 200, 400],
        "model__max_depth": [2, 3, 5],
        "model__learning_rate": [0.03, 0.1],
        "model__min_child_weight": [1, 5],
    },
}
SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


def build_model_pipeline(model_key: str, strategy: str, pos_weight: float) -> Pipeline:
    """Construye preprocesador + (SMOTE opcional) + clasificador.

    Se usa el Pipeline de imbalanced-learn: SMOTE solo se aplica durante `fit`,
    es decir, únicamente sobre los pliegues de entrenamiento de cada partición.
    """
    if model_key == "logreg":
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if strategy == "weighted" else None,
            random_state=config.RANDOM_STATE,
        )
    elif model_key == "xgboost":
        model = XGBClassifier(
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=pos_weight if strategy == "weighted" else 1.0,
            n_jobs=1,  # el paralelismo lo gestiona GridSearchCV
            random_state=config.RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model: {model_key}")

    steps = [("preprocessor", build_preprocessor())]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)


def select_threshold(y_true: np.ndarray, proba: np.ndarray, beta: float) -> tuple[float, float]:
    """Devuelve el umbral que maximiza F-beta y su valor (sobre datos de validación)."""
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [fbeta_score(y_true, proba >= t, beta=beta, zero_division=0) for t in thresholds]
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def build_risk_bands(threshold: float) -> list[tuple[float, str]]:
    """Bandas de riesgo alineadas con el umbral: Low (no marcado), Medium, High."""
    medium = min(threshold, config.HIGH_RISK_PROBA)
    return [(0.0, "Low"), (medium, "Medium"), (config.HIGH_RISK_PROBA, "High")]


def oof_summary(y: np.ndarray, proba: np.ndarray) -> dict:
    """Métricas OOF a umbral 0.5: muestran el efecto de cada estrategia de desbalanceo."""
    tn, fp, fn, tp = confusion_matrix(y, proba >= 0.5).ravel()
    return {
        "oof_precision@0.5": tp / (tp + fp) if tp + fp else 0.0,
        "oof_recall@0.5": tp / (tp + fn),
        "oof_false_positives@0.5": int(fp),
        "oof_false_negatives@0.5": int(fn),
        "oof_brier": brier_score_loss(y, proba),
        "oof_mean_predicted_proba": float(proba.mean()),
    }


def run_training() -> dict:
    """Ejecuta todo el proceso de entrenamiento y guarda modelos y metadatos."""
    df = clean_data(load_raw_data())
    train_df, test_df = train_test_split_stratified(df)
    X_train, y_train = split_features_target(train_df)
    y_arr = y_train.to_numpy()

    pos_weight = float((y_arr == 0).sum() / (y_arr == 1).sum())
    logger.info("scale_pos_weight (neg/pos in train) = %.3f", pos_weight)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    rows, searches, oof_probas = [], {}, {}
    for model_key in MODEL_NAMES:
        for strategy in STRATEGIES:
            logger.info("Grid search: %s | imbalance strategy = %s", model_key, strategy)
            search = GridSearchCV(
                build_model_pipeline(model_key, strategy, pos_weight),
                PARAM_GRIDS[model_key],
                scoring=SCORING,
                refit=config.PRIMARY_METRIC,
                cv=cv,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best = search.best_index_
            res = search.cv_results_

            # Predicciones out-of-fold del mejor estimador (mismos pliegues).
            proba = cross_val_predict(
                clone(search.best_estimator_), X_train, y_train, cv=cv,
                method="predict_proba", n_jobs=-1,
            )[:, 1]
            oof_probas[(model_key, strategy)] = proba
            searches[(model_key, strategy)] = search

            row = {
                "model": MODEL_NAMES[model_key],
                "model_key": model_key,
                "strategy": strategy,
                "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            }
            for metric in SCORING:
                row[f"cv_{metric}_mean"] = res[f"mean_test_{metric}"][best]
                row[f"cv_{metric}_std"] = res[f"std_test_{metric}"][best]
            row.update(oof_summary(y_arr, proba))
            rows.append(row)
            logger.info(
                "  -> CV PR-AUC %.4f | ROC-AUC %.4f | params %s",
                row["cv_average_precision_mean"], row["cv_roc_auc_mean"], row["best_params"],
            )

    results = pd.DataFrame(rows)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.CV_RESULTS_PATH, index=False)

    # --- Mejor estrategia por familia de modelo y modelo final ---
    metric_col = f"cv_{config.PRIMARY_METRIC}_mean"
    best_per_model = results.loc[results.groupby("model_key")[metric_col].idxmax()]
    final_row = best_per_model.loc[best_per_model[metric_col].idxmax()]

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    candidates, oof_frame = {}, pd.DataFrame({config.ID_COL: train_df[config.ID_COL], "y_true": y_arr})
    for _, row in best_per_model.iterrows():
        key = (row["model_key"], row["strategy"])
        proba = oof_probas[key]
        threshold, f_beta = select_threshold(y_arr, proba, config.THRESHOLD_BETA)
        alt_threshold, _ = select_threshold(y_arr, proba, config.ALT_THRESHOLD_BETA)
        path = config.MODELS_DIR / f"{row['model_key']}.joblib"
        joblib.dump(searches[key].best_estimator_, path)
        oof_frame[f"proba_{row['model_key']}"] = proba
        candidates[row["model_key"]] = {
            "model": row["model"],
            "strategy": row["strategy"],
            "best_params": row["best_params"],
            "model_path": path.name,
            "threshold": threshold,
            "threshold_oof_fbeta": f_beta,
            f"f{config.ALT_THRESHOLD_BETA:g}_optimal_threshold_reference": alt_threshold,
            "cv_metrics": {m: float(row[f"cv_{m}_mean"]) for m in SCORING},
        }
        logger.info("%s: best strategy=%s, F%g threshold=%.2f",
                    row["model"], row["strategy"], config.THRESHOLD_BETA, threshold)

    oof_frame.to_csv(config.MODELS_DIR / "oof_predictions.csv", index=False)

    final_key = final_row["model_key"]
    final_model = searches[(final_key, final_row["strategy"])].best_estimator_
    joblib.dump(final_model, config.MODEL_PATH)

    # Muestra de fondo (datos de TRAIN sin transformar) para SHAP en inferencia.
    background = X_train.sample(n=200, random_state=config.RANDOM_STATE)
    joblib.dump(background, config.MODELS_DIR / "shap_background.joblib")

    metadata = {
        "final_model_key": final_key,
        "final_model": final_row["model"],
        "imbalance_strategy": final_row["strategy"],
        "threshold": candidates[final_key]["threshold"],
        "threshold_criterion": f"max F{config.THRESHOLD_BETA:g} on out-of-fold training predictions",
        "selection_metric": f"mean {config.CV_FOLDS}-fold CV {config.PRIMARY_METRIC}",
        "candidates": candidates,
        "risk_bands": build_risk_bands(candidates[final_key]["threshold"]),
        "features": {"numeric": config.NUMERIC_FEATURES, "categorical": config.CATEGORICAL_FEATURES},
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_churn_rate": float(y_arr.mean()),
        "test_customer_ids": test_df[config.ID_COL].tolist(),
        "random_state": config.RANDOM_STATE,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    save_json(metadata, config.METADATA_PATH)
    logger.info("Final model: %s (%s) saved to %s", final_row["model"], final_row["strategy"],
                config.MODEL_PATH.name)
    return metadata


if __name__ == "__main__":
    run_training()
