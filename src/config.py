"""Configuración central del proyecto: rutas, columnas y parámetros.

Centralizar aquí las constantes evita "números mágicos" repartidos por el código
y garantiza que todos los módulos usan exactamente la misma configuración.
"""

from pathlib import Path

# --- Rutas (relativas a la raíz del proyecto, nunca absolutas) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
CV_RESULTS_PATH = REPORTS_DIR / "cv_results.csv"
TEST_METRICS_PATH = REPORTS_DIR / "test_metrics.json"
PREDICTIONS_PATH = REPORTS_DIR / "customer_churn_predictions.csv"
MODEL_RESULTS_MD = REPORTS_DIR / "model_results.md"

# --- Esquema de datos ---
ID_COL = "customerID"
TARGET_COL = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Reproducibilidad y validación ---
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Métrica principal para seleccionar modelo: PR-AUC (Average Precision).
# Es sensible al rendimiento sobre la clase minoritaria, a diferencia de accuracy.
PRIMARY_METRIC = "average_precision"

# Criterio de umbral: maximizar F1 sobre predicciones out-of-fold (validación).
# Se evaluó F2 (más peso al recall), pero su óptimo es muy plano y marcaba ~50% de la cartera
# como riesgo: poco realista para una campaña. El umbral F2 se guarda como referencia.
THRESHOLD_BETA = 1.0
ALT_THRESHOLD_BETA = 2.0

# --- Niveles de riesgo ---
# Low = por debajo del umbral (no se marca); Medium = marcado con p < HIGH_RISK_PROBA; High = resto.
HIGH_RISK_PROBA = 0.60

# Número de factores SHAP que se muestran por cliente en el CSV de predicciones.
TOP_N_DRIVERS = 3
# Contribuciones menores que esto (en log-odds) se consideran despreciables y no se listan.
MIN_DRIVER_CONTRIBUTION = 0.01

# --- Supuestos económicos HIPOTÉTICOS (solo para ilustrar el impacto; no son datos reales) ---
ECONOMIC_ASSUMPTIONS = {
    "cost_per_targeted_customer": 30.0,  # contacto + incentivo de retención, por cliente contactado
    "retention_success_rate": 0.25,  # % de churners reales contactados que deciden quedarse
    "gross_margin": 0.50,  # margen bruto sobre la cuota mensual
    "retained_months": 12,  # meses de ingresos que se conservan si el cliente se queda
}
SENSITIVITY_SUCCESS_RATES = [0.10, 0.25, 0.40]
