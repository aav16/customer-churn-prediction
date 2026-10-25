"""Evaluación final sobre el conjunto de test, gráficos e informe de resultados.

El test se usa SOLO aquí, con los modelos y umbrales ya fijados durante el
entrenamiento. Todas las cifras de `reports/model_results.md` se generan a partir
de esta ejecución (no se escriben a mano).
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config
from src.data_preprocessing import (
    clean_data,
    load_raw_data,
    split_features_target,
    train_test_split_stratified,
)
from src.plotting import (
    AQUA,
    BLUE,
    GRAY,
    MODEL_COLORS,
    ORANGE,
    SEQUENTIAL_CMAP,
    TEXT_SECONDARY,
    apply_style,
    save_figure,
)
from src.utils import get_logger, load_json, save_json

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Datos y métricas
# ---------------------------------------------------------------------------
def load_test_set(metadata: dict) -> pd.DataFrame:
    """Reconstruye el test de forma determinista y verifica que coincide con el del entrenamiento."""
    _, test_df = train_test_split_stratified(clean_data(load_raw_data()))
    if test_df[config.ID_COL].tolist() != metadata["test_customer_ids"]:
        raise RuntimeError("Test split differs from the one used in training. Retrain the model.")
    return test_df


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Métricas de clasificación (dependen del umbral) y de ranking (no dependen)."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "f2": fbeta_score(y_true, pred, beta=2),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "flagged_share": float(pred.mean()),
    }


def business_impact(
    monthly_charges: np.ndarray, y_true: np.ndarray, targeted: np.ndarray, success_rate: float
) -> dict:
    """Valor neto esperado de una campaña bajo supuestos HIPOTÉTICOS explícitos.

    Valor = Σ(churners reales contactados) × éxito × cuota × margen × meses − coste × contactados.
    """
    a = config.ECONOMIC_ASSUMPTIONS
    value_if_saved = monthly_charges * a["gross_margin"] * a["retained_months"]
    saved_value = float((value_if_saved * success_rate)[(targeted == 1) & (y_true == 1)].sum())
    cost = float(targeted.sum() * a["cost_per_targeted_customer"])
    return {
        "customers_targeted": int(targeted.sum()),
        "churners_reached": int(((targeted == 1) & (y_true == 1)).sum()),
        "expected_customers_retained": float(((targeted == 1) & (y_true == 1)).sum() * success_rate),
        "expected_value_saved": saved_value,
        "campaign_cost": cost,
        "expected_net_value": saved_value - cost,
    }


def build_business_table(test_df: pd.DataFrame, proba: np.ndarray, metadata: dict) -> pd.DataFrame:
    """Compara estrategias de targeting en test con distintos supuestos de éxito."""
    y = test_df[config.TARGET_COL].to_numpy()
    mc = test_df["MonthlyCharges"].to_numpy()
    final = metadata["candidates"][metadata["final_model_key"]]
    alt_key = f"f{config.ALT_THRESHOLD_BETA:g}_optimal_threshold_reference"
    rng = np.random.default_rng(config.RANDOM_STATE)

    n_selected = int((proba >= final["threshold"]).sum())
    random_target = np.zeros(len(y), dtype=int)
    random_target[rng.choice(len(y), size=n_selected, replace=False)] = 1

    strategies = {
        f"Model, selected threshold ({final['threshold']:.2f})": (proba >= final["threshold"]).astype(int),
        f"Model, recall-oriented F2 threshold ({final[alt_key]:.2f})": (proba >= final[alt_key]).astype(int),
        "Model, default threshold (0.50)": (proba >= 0.5).astype(int),
        "Random targeting (same volume as selected)": random_target,
        "Target all customers": np.ones(len(y), dtype=int),
    }
    rows = []
    for rate in config.SENSITIVITY_SUCCESS_RATES:
        for name, targeted in strategies.items():
            rows.append({"strategy": name, "retention_success_rate": rate,
                         **business_impact(mc, y, targeted, rate)})
    return pd.DataFrame(rows)


def build_segment_summary(test_df: pd.DataFrame, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    """Riesgo predicho vs churn observado por segmento (clientes de test)."""
    df = test_df.assign(
        predicted_proba=proba,
        flagged=(proba >= threshold).astype(int),
        tenure_band=pd.cut(test_df["tenure"], [-1, 6, 12, 24, 48, 72],
                           labels=["0-6", "7-12", "13-24", "25-48", "49-72"]).astype(str),
    )
    frames = []
    for col in ["Contract", "tenure_band", "InternetService", "PaymentMethod", "TechSupport",
                "OnlineSecurity", "SeniorCitizen"]:
        g = df.groupby(col, observed=True).agg(
            customers=("predicted_proba", "size"),
            mean_predicted_proba=("predicted_proba", "mean"),
            observed_churn_rate=(config.TARGET_COL, "mean"),
            share_flagged=("flagged", "mean"),
        )
        frames.append(g.reset_index().rename(columns={col: "segment"}).assign(variable=col))
    out = pd.concat(frames, ignore_index=True)
    out["segment"] = out["segment"].astype(str)
    return out[["variable", "segment", "customers", "mean_predicted_proba",
                "observed_churn_rate", "share_flagged"]]


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def plot_confusion_matrices(y: np.ndarray, proba: np.ndarray, threshold: float, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, t in zip(axes, [0.5, threshold]):
        cm = confusion_matrix(y, proba >= t)
        ax.imshow(cm, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=cm.max() * 1.3)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}\n({v / cm.sum():.1%})", ha="center", va="center", fontsize=11,
                    color="white" if v > cm.max() * 0.65 else "#0b0b0b")
        ax.set_xticks([0, 1], ["Predicted: stay", "Predicted: churn"])
        ax.set_yticks([0, 1], ["Actual: stay", "Actual: churn"])
        ax.grid(False)
        label = "default" if t == 0.5 else "selected"
        ax.set_title(f"Threshold {t:.2f} ({label})")
    fig.suptitle(f"{title} — confusion matrix on test set (n={len(y):,})", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    save_figure(fig, config.FIGURES_DIR / "eval_confusion_matrix.png")


def plot_roc_pr(y: np.ndarray, probas: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="Random (AUC 0.500)")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
           title="ROC curve — test set")
    ax.legend(loc="lower right")
    save_figure(fig, config.FIGURES_DIR / "eval_roc_curve.png")

    fig, ax = plt.subplots(figsize=(5.8, 5))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=MODEL_COLORS[name],
                label=f"{name} (AP {average_precision_score(y, p):.3f})")
    ax.axhline(y.mean(), color=GRAY, linestyle="--", linewidth=1,
               label=f"No-skill baseline ({y.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02),
           title="Precision-Recall curve — test set")
    ax.legend(loc="upper right")
    save_figure(fig, config.FIGURES_DIR / "eval_precision_recall_curve.png")


def plot_metric_comparison(metrics: dict[str, dict]) -> None:
    names = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    labels = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, (model, m) in enumerate(metrics.items()):
        values = [m[n] for n in names]
        bars = ax.bar(x + (i - 0.5) * width, values, width - 0.02, color=MODEL_COLORS[model],
                      label=f"{model} (threshold {m['threshold']:.2f})")
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8.5,
                    color=TEXT_SECONDARY)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.set_title("Model comparison on test set (each at its validation-selected threshold)")
    ax.legend(loc="upper left", ncols=2)
    save_figure(fig, config.FIGURES_DIR / "eval_metric_comparison.png")


def plot_threshold_analysis(oof: pd.DataFrame, model_key: str, model_name: str,
                            threshold: float, alt_threshold: float) -> None:
    """Precision/recall/F1 vs umbral sobre predicciones OOF (validación, no test)."""
    y, p = oof["y_true"].to_numpy(), oof[f"proba_{model_key}"].to_numpy()
    ts = np.round(np.arange(0.05, 0.951, 0.01), 2)
    prec = [precision_score(y, p >= t, zero_division=0) for t in ts]
    rec = [recall_score(y, p >= t) for t in ts]
    f1 = [f1_score(y, p >= t) for t in ts]
    flagged = [(p >= t).mean() for t in ts]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(ts, prec, color=BLUE, label="Precision")
    ax.plot(ts, rec, color=ORANGE, label="Recall")
    ax.plot(ts, f1, color=AQUA, label="F1")
    ax.plot(ts, flagged, color=GRAY, linestyle=":", label="Share of customers flagged")
    ax.axvline(threshold, color="#0b0b0b", linewidth=1.2)
    ax.text(threshold + 0.01, 0.03, f"selected (max F1) = {threshold:.2f}", fontsize=9)
    ax.axvline(alt_threshold, color=GRAY, linewidth=1, linestyle="--")
    ax.text(alt_threshold + 0.01, 0.95, f"max F2 = {alt_threshold:.2f}", fontsize=9, color=TEXT_SECONDARY)
    ax.set(xlabel="Classification threshold", ylabel="Score", ylim=(0, 1.02),
           title=f"{model_name} — threshold selection on out-of-fold validation predictions")
    ax.legend(loc="center right")
    save_figure(fig, config.FIGURES_DIR / "eval_threshold_analysis.png")


def plot_imbalance_strategies(cv: pd.DataFrame) -> None:
    panels = [("cv_average_precision_mean", "CV PR-AUC (threshold-free)"),
              ("oof_recall@0.5", "Recall at threshold 0.5"),
              ("oof_precision@0.5", "Precision at threshold 0.5"),
              ("oof_false_positives@0.5", "False positives at 0.5 (OOF, n=5,634)")]
    strategies = ["none", "weighted", "smote"]
    labels = ["No weighting", "Class weighting", "SMOTE"]
    x = np.arange(len(strategies))
    width = 0.38
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (col, title) in zip(axes, panels):
        for i, model in enumerate(["Logistic Regression", "XGBoost"]):
            sub = cv[cv["model"] == model].set_index("strategy").loc[strategies]
            bars = ax.bar(x + (i - 0.5) * width, sub[col], width - 0.02, color=MODEL_COLORS[model],
                          label=model)
            for b, v in zip(bars, sub[col]):
                txt = f"{v:,.0f}" if "false" in col else f"{v:.3f}"
                ax.text(b.get_x() + b.get_width() / 2, v, txt, ha="center", va="bottom", fontsize=8,
                        color=TEXT_SECONDARY)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="x", visible=False)
        if "false" not in col:
            ax.set_ylim(0, 1.0)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper right", ncols=2)
    fig.suptitle("Class-imbalance strategies — 5-fold CV on training data only", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    save_figure(fig, config.FIGURES_DIR / "eval_imbalance_strategies.png")


def plot_calibration(y: np.ndarray, probas: dict[str, np.ndarray]) -> None:
    """Curva de fiabilidad: ¿una probabilidad de 0.7 significa ~70% de abandono real?"""
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="Perfect calibration")
    for name, p in probas.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", markersize=5, color=MODEL_COLORS[name],
                label=f"{name} (Brier {brier_score_loss(y, p):.3f})")
    ax.set(xlabel="Mean predicted probability (decile bins)", ylabel="Observed churn rate",
           title="Calibration curve — test set", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    save_figure(fig, config.FIGURES_DIR / "eval_calibration_curve.png")


def plot_probability_distribution(y: np.ndarray, proba: np.ndarray, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(proba[y == 0], bins=bins, histtype="step", linewidth=1.8, color=BLUE, label="Actual: stayed")
    ax.hist(proba[y == 1], bins=bins, histtype="step", linewidth=1.8, color=ORANGE, label="Actual: churned")
    ax.axvline(threshold, color="#0b0b0b", linewidth=1.2)
    ax.text(threshold + 0.01, ax.get_ylim()[1] * 0.9, f"threshold {threshold:.2f}", fontsize=9)
    ax.set(xlabel="Predicted churn probability", ylabel="Customers",
           title="Predicted probability by actual outcome — test set")
    ax.legend()
    save_figure(fig, config.FIGURES_DIR / "eval_probability_distribution.png")


# ---------------------------------------------------------------------------
# Informe markdown
# ---------------------------------------------------------------------------
def _fmt_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Convierte un DataFrame a tabla markdown sin dependencias extra."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def write_markdown_report(metadata, cv, test_metrics, default_metrics, report_text, business) -> None:
    final_key = metadata["final_model_key"]
    final = metadata["candidates"][final_key]
    alt_key = f"f{config.ALT_THRESHOLD_BETA:g}_optimal_threshold_reference"
    a = config.ECONOMIC_ASSUMPTIONS

    cv_table = cv[["model", "strategy", "cv_average_precision_mean", "cv_average_precision_std",
                   "cv_roc_auc_mean", "oof_precision@0.5", "oof_recall@0.5",
                   "oof_false_positives@0.5", "oof_false_negatives@0.5", "oof_brier"]].copy()
    cv_table.columns = ["Model", "Imbalance strategy", "CV PR-AUC", "± std", "CV ROC-AUC",
                        "Precision@0.5", "Recall@0.5", "FP@0.5", "FN@0.5", "Brier"]

    def test_rows(metrics_by_model):
        rows = []
        for key, m in metrics_by_model.items():
            rows.append({"Model": metadata["candidates"][key]["model"],
                         "Strategy": metadata["candidates"][key]["strategy"],
                         "Threshold": m["threshold"], "Accuracy": m["accuracy"],
                         "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
                         "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"], "Brier": m["brier"],
                         "TP": m["tp"], "FP": m["fp"], "FN": m["fn"], "TN": m["tn"]})
        return pd.DataFrame(rows)

    fm = test_metrics[final_key]
    biz = business[business["retention_success_rate"] == a["retention_success_rate"]].drop(
        columns="retention_success_rate")
    sens = business.pivot_table(index="strategy", columns="retention_success_rate",
                                values="expected_net_value", sort=False).reset_index()
    sens.columns = ["Strategy"] + [f"Net value @ {c:.0%} success" for c in sens.columns[1:]]

    params = ", ".join(f"`{k}={v}`" for k, v in final["best_params"].items())
    md = f"""# Model Results

> Auto-generated by `src/evaluate.py` from the latest run of `python main.py`
> (model trained at {metadata['trained_at_utc']} UTC). Do not edit by hand.

## 1. Setup

- Train / test split: {metadata['n_train']:,} / {metadata['n_test']:,} customers, stratified
  (churn rate {metadata['train_churn_rate']:.1%}), `random_state={metadata['random_state']}`.
- Model and imbalance-strategy selection: {metadata['selection_metric']} on the training set only.
- Threshold selection: {metadata['threshold_criterion']}.
- The test set was used **once**, for the final evaluation reported in section 3.

## 2. Cross-validation: models × class-imbalance strategies (training data)

Precision, recall, false positives (FP) and false negatives (FN) are computed on out-of-fold
predictions at the default 0.5 threshold, to show how each imbalance strategy shifts the
precision/recall trade-off. PR-AUC and ROC-AUC are threshold-free.

{_fmt_table(cv_table)}

**Selected model:** {final['model']} with imbalance strategy `{final['strategy']}` ({params}),
highest mean CV PR-AUC ({final['cv_metrics']['average_precision']:.3f}).

**Selected threshold:** {final['threshold']:.2f} (max F1 on out-of-fold predictions).
Recall-oriented alternative (max F2): {final[alt_key]:.2f}.

![Imbalance strategies](figures/eval_imbalance_strategies.png)
![Threshold analysis](figures/eval_threshold_analysis.png)

## 3. Final evaluation on the held-out test set

### At each model's validation-selected threshold

{_fmt_table(test_rows(test_metrics))}

### At the default threshold (0.50), for reference

{_fmt_table(test_rows(default_metrics))}

### Classification report — {final['model']} at threshold {final['threshold']:.2f}

```text
{report_text}
```

With the selected threshold the final model flags {fm['flagged_share']:.1%} of test customers,
detects {fm['tp']:,} of {fm['tp'] + fm['fn']:,} actual churners (recall {fm['recall']:.3f}) and
{fm['precision']:.1%} of flagged customers actually churned (precision).
A naive "nobody churns" classifier would reach {1 - (fm['tp'] + fm['fn']) / metadata['n_test']:.1%}
accuracy with zero recall, which is why accuracy is reported only for reference.

![Confusion matrix](figures/eval_confusion_matrix.png)
![ROC curve](figures/eval_roc_curve.png)
![PR curve](figures/eval_precision_recall_curve.png)
![Metric comparison](figures/eval_metric_comparison.png)
![Probability distribution](figures/eval_probability_distribution.png)
![Calibration](figures/eval_calibration_curve.png)

## 4. Illustrative business impact (HYPOTHETICAL assumptions)

Assumptions (not company data): cost per targeted customer = {a['cost_per_targeted_customer']:.0f}
(currency units); a targeted true churner is retained with probability
{a['retention_success_rate']:.0%}; a retained customer is worth
{a['retained_months']} months × MonthlyCharges × {a['gross_margin']:.0%} gross margin.
Non-churners who are contacted generate cost but no saved value. Computed on the
{metadata['n_test']:,} test customers.

{_fmt_table(biz, floatfmt="{:,.0f}")}

**Sensitivity of expected net value to the retention success rate:**

{_fmt_table(sens, floatfmt="{:,.0f}")}

These figures are scenario calculations, not measured outcomes: the real effect of any retention
action must be validated with a controlled experiment (A/B test).
"""
    config.MODEL_RESULTS_MD.write_text(md, encoding="utf-8")
    logger.info("Wrote %s", config.MODEL_RESULTS_MD.name)


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def run_evaluation() -> dict:
    apply_style()
    metadata = load_json(config.METADATA_PATH)
    test_df = load_test_set(metadata)
    X_test, y_test = split_features_target(test_df)
    y = y_test.to_numpy()

    probas, test_metrics, default_metrics = {}, {}, {}
    for key, cand in metadata["candidates"].items():
        model = joblib.load(config.MODELS_DIR / cand["model_path"])
        p = model.predict_proba(X_test)[:, 1]
        probas[cand["model"]] = p
        test_metrics[key] = compute_metrics(y, p, cand["threshold"])
        default_metrics[key] = compute_metrics(y, p, 0.5)
        logger.info("%s test: PR-AUC %.4f | ROC-AUC %.4f | recall %.3f | precision %.3f",
                    cand["model"], test_metrics[key]["pr_auc"], test_metrics[key]["roc_auc"],
                    test_metrics[key]["recall"], test_metrics[key]["precision"])

    final_key = metadata["final_model_key"]
    final = metadata["candidates"][final_key]
    final_proba = probas[final["model"]]
    alt_key = f"f{config.ALT_THRESHOLD_BETA:g}_optimal_threshold_reference"

    report_text = classification_report(y, (final_proba >= final["threshold"]).astype(int),
                                        target_names=["No churn", "Churn"], digits=3)
    cv = pd.read_csv(config.CV_RESULTS_PATH)
    oof = pd.read_csv(config.MODELS_DIR / "oof_predictions.csv")

    plot_confusion_matrices(y, final_proba, final["threshold"], final["model"])
    plot_roc_pr(y, probas)
    plot_metric_comparison({metadata["candidates"][k]["model"]: m for k, m in test_metrics.items()})
    plot_threshold_analysis(oof, final_key, final["model"], final["threshold"], final[alt_key])
    plot_imbalance_strategies(cv)
    plot_probability_distribution(y, final_proba, final["threshold"])
    plot_calibration(y, probas)

    business = build_business_table(test_df, final_proba, metadata)
    business.to_csv(config.REPORTS_DIR / "business_impact_scenarios.csv", index=False)
    build_segment_summary(test_df, final_proba, final["threshold"]).to_csv(
        config.REPORTS_DIR / "segment_risk_summary.csv", index=False)

    results = {"final_model": final["model"], "test_metrics": test_metrics,
               "default_threshold_metrics": default_metrics}
    save_json(results, config.TEST_METRICS_PATH)
    write_markdown_report(metadata, cv, test_metrics, default_metrics, report_text, business)
    return results


if __name__ == "__main__":
    run_evaluation()
