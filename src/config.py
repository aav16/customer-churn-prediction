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

# Criterio de umbral: F-beta con beta=2 (el recall pesa el doble que la precisión),
# porque no detectar a un cliente que se va suele costar más que contactar a uno que se queda.
THRESHOLD_BETA = 2.0

# --- Niveles de riesgo (bandas de probabilidad, límite inferior inclusivo) ---
RISK_BANDS = [(0.0, "Low"), (0.30, "Medium"), (0.60, "High")]

# Número de factores SHAP que se muestran por cliente en el CSV de predicciones.
TOP_N_DRIVERS = 3
