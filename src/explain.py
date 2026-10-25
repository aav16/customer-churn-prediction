"""Explicabilidad con SHAP: global (qué variables importan) e individual (por qué este cliente).

Escala: los valores SHAP se calculan sobre la salida en LOG-ODDS del modelo
(margen bruto de XGBoost / función lineal de la regresión logística), no sobre
la probabilidad. Se cumple:  base_value + Σ SHAP = logit(p) = log(p / (1 - p)).
Por tanto, un SHAP de +0.5 NO significa +0.5 puntos porcentuales de probabilidad.

Las variables categóricas se transforman en varias columnas one-hot; como SHAP es
aditivo, sumamos las contribuciones de esas columnas para obtener la contribución
de la variable original (p. ej. todas las columnas `Contract_*` -> `Contract`).
"""

from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.pipeline import Pipeline
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src import config
from src.data_preprocessing import get_feature_mapping, split_features_target
from src.evaluate import load_test_set
from src.plotting import apply_style, save_figure
from src.utils import get_logger, load_json

logger = get_logger(__name__)


@dataclass
class ShapResult:
    """Valores SHAP (log-odds) a nivel transformado y agregados a variables originales."""

    values: np.ndarray  # (n_clientes, n_columnas_transformadas)
    base_value: float  # valor esperado del modelo en log-odds
    data: pd.DataFrame  # datos transformados (columnas = etiquetas legibles)
    mapping: pd.DataFrame  # transformada -> original
    original: pd.DataFrame  # (n_clientes, n_variables_originales)


def compute_shap(pipeline: Pipeline, X: pd.DataFrame, background: pd.DataFrame) -> ShapResult:
    """Calcula valores SHAP en log-odds para el modelo del pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    mapping = get_feature_mapping(preprocessor)
    X_t = preprocessor.transform(X)

    if isinstance(model, XGBClassifier):
        # TreeExplainer exacto; salida "raw" = margen en log-odds.
        explainer = shap.TreeExplainer(model, model_output="raw")
    elif isinstance(model, LogisticRegression):
        # Explicación intervencional respecto a una muestra de fondo de TRAIN.
        explainer = shap.LinearExplainer(model, preprocessor.transform(background))
    else:
        raise TypeError(f"Unsupported model type for SHAP: {type(model).__name__}")

    explanation = explainer(X_t)
    values = np.asarray(explanation.values, dtype=float)
    base_value = float(np.ravel(explanation.base_values)[0])

    # Comprobación de aditividad: base + Σ SHAP debe reproducir el log-odds del modelo.
    proba = np.clip(pipeline.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    max_err = np.abs(base_value + values.sum(axis=1) - logit(proba)).max()
    if max_err > 1e-3:
        raise RuntimeError(f"SHAP additivity check failed (max error {max_err:.2e}).")
    logger.info("SHAP additivity check passed (max |error| = %.1e log-odds)", max_err)

    # Agregación a variables originales mediante una matriz indicadora (transformada x original).
    indicator = pd.get_dummies(mapping["original"]).reindex(columns=config.FEATURES).astype(float)
    original = pd.DataFrame(values @ indicator.to_numpy(), columns=config.FEATURES, index=X.index)
    data = pd.DataFrame(X_t, columns=mapping["label"].tolist(), index=X.index)
    return ShapResult(values, base_value, data, mapping, original)


def _format_value(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:g}" if float(value).is_integer() else f"{value:.2f}"
    return str(value)


def top_drivers(result: ShapResult, X: pd.DataFrame, n: int = config.TOP_N_DRIVERS
                ) -> tuple[list[str], list[str]]:
    """Para cada cliente: n variables que más AUMENTAN y más REDUCEN el riesgo (log-odds)."""
    increasing, decreasing = [], []
    for idx in X.index:
        contrib = result.original.loc[idx].sort_values(ascending=False)
        fmt = lambda items: "; ".join(
            f"{f}={_format_value(X.at[idx, f])} ({v:+.2f})" for f, v in items.items())
        eps = config.MIN_DRIVER_CONTRIBUTION
        increasing.append(fmt(contrib[contrib >= eps].head(n)))
        decreasing.append(fmt(contrib[contrib <= -eps].sort_values().head(n)))
    return increasing, decreasing


def global_importance(result: ShapResult) -> pd.DataFrame:
    """Importancia global = media del |SHAP| por variable original (en log-odds)."""
    imp = pd.DataFrame({
        "feature": config.FEATURES,
        "mean_abs_shap_log_odds": result.original.abs().mean().to_numpy(),
        "mean_shap_log_odds": result.original.mean().to_numpy(),
    })
    imp = imp.sort_values("mean_abs_shap_log_odds", ascending=False).reset_index(drop=True)
    imp["share_of_total"] = imp["mean_abs_shap_log_odds"] / imp["mean_abs_shap_log_odds"].sum()
    return imp


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def _explanation(result: ShapResult) -> shap.Explanation:
    return shap.Explanation(values=result.values, base_values=np.full(len(result.values), result.base_value),
                            data=result.data.to_numpy(), feature_names=list(result.data.columns))


def plot_global(result: ShapResult, model_name: str) -> None:
    fig_dir = config.FIGURES_DIR
    xlabel = "SHAP value (impact on churn log-odds)"

    # 1) Summary plot (vista de distribución, estilo "violin").
    shap.summary_plot(result.values, result.data, max_display=15, plot_type="violin", show=False)
    fig = plt.gcf()
    fig.axes[0].set_xlabel(xlabel)
    fig.axes[0].set_title(f"SHAP summary — {model_name} (test set)", loc="left")
    save_figure(fig, fig_dir / "shap_summary_plot.png")

    # 2) Beeswarm: un punto por cliente; color = valor de la variable (transformada).
    shap.plots.beeswarm(_explanation(result), max_display=15, show=False)
    fig = plt.gcf()
    fig.axes[0].set_xlabel(xlabel)
    fig.axes[0].set_title(f"SHAP beeswarm — {model_name} (test set)", loc="left")
    save_figure(fig, fig_dir / "shap_beeswarm.png")

    # 3) Bar plot a nivel de columna transformada (categoría concreta).
    shap.plots.bar(_explanation(result), max_display=15, show=False)
    fig = plt.gcf()
    fig.axes[0].set_xlabel("mean(|SHAP value|) in log-odds")
    fig.axes[0].set_title("SHAP bar plot — encoded features", loc="left")
    save_figure(fig, fig_dir / "shap_bar.png")

    # 4) Bar plot agregado a variables originales (la vista más útil para negocio).
    imp = global_importance(result).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(imp["feature"], imp["mean_abs_shap_log_odds"], color="#2a78d6", height=0.65)
    for i, v in enumerate(imp["mean_abs_shap_log_odds"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8.5, color="#52514e")
    ax.set_xlabel("mean(|SHAP value|) in log-odds — one-hot columns summed per original feature")
    ax.set_title(f"Global feature importance — {model_name}")
    ax.grid(axis="y", visible=False)
    save_figure(fig, fig_dir / "shap_bar_original_features.png")


def select_example_customers(proba: pd.Series, threshold: float) -> dict[str, object]:
    """Elige clientes representativos de distintos niveles de probabilidad."""
    order = proba.sort_values()
    picks = {
        "highest_risk": order.index[-1],
        "high_risk_p90": (proba - proba.quantile(0.90)).abs().idxmin(),
        "near_threshold": (proba - threshold).abs().idxmin(),
        "lowest_risk": order.index[0],
    }
    return picks


def plot_waterfalls(result: ShapResult, X: pd.DataFrame, ids: pd.Series, proba: pd.Series,
                    picks: dict[str, object]) -> pd.DataFrame:
    """Waterfall por cliente con variables originales; devuelve la tabla de contribuciones."""
    rows = []
    for label, idx in picks.items():
        contrib = result.original.loc[idx]
        names = [f"{f} = {_format_value(X.at[idx, f])}" for f in config.FEATURES]
        exp = shap.Explanation(values=contrib.to_numpy(), base_values=result.base_value,
                               data=None, feature_names=names)
        shap.plots.waterfall(exp, max_display=12, show=False)
        fig = plt.gcf()
        fig.axes[0].set_title(
            f"Customer {ids.loc[idx]} — predicted churn probability {proba.loc[idx]:.1%}\n"
            f"Contributions in log-odds: E[f(x)] = {result.base_value:.2f} "
            f"(≈{1 / (1 + np.exp(-result.base_value)):.1%}), f(x) = logit(p)",
            loc="left", fontsize=10)
        save_figure(fig, config.FIGURES_DIR / f"shap_waterfall_{label}.png")
        for f in config.FEATURES:
            rows.append({"example": label, config.ID_COL: ids.loc[idx],
                         "churn_probability": proba.loc[idx], "feature": f,
                         "value": _format_value(X.at[idx, f]),
                         "shap_log_odds": contrib[f]})
    out = pd.DataFrame(rows)
    out["abs"] = out["shap_log_odds"].abs()
    return out.sort_values(["example", "abs"], ascending=[True, False]).drop(columns="abs")


def run_explainability() -> dict:
    apply_style()
    metadata = load_json(config.METADATA_PATH)
    pipeline = joblib.load(config.MODEL_PATH)
    background = joblib.load(config.MODELS_DIR / "shap_background.joblib")
    test_df = load_test_set(metadata)
    X_test, _ = split_features_target(test_df)

    result = compute_shap(pipeline, X_test, background)
    proba = pd.Series(pipeline.predict_proba(X_test)[:, 1], index=X_test.index)

    plot_global(result, metadata["final_model"])
    imp = global_importance(result)
    imp.to_csv(config.REPORTS_DIR / "shap_feature_importance.csv", index=False)

    picks = select_example_customers(proba, metadata["threshold"])
    examples = plot_waterfalls(result, X_test, test_df[config.ID_COL], proba, picks)
    examples.to_csv(config.REPORTS_DIR / "shap_individual_explanations.csv", index=False)

    logger.info("Top 5 features by mean |SHAP|: %s", ", ".join(imp["feature"].head(5)))
    return {"importance": imp, "examples": examples}


if __name__ == "__main__":
    run_explainability()
