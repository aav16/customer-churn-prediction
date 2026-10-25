"""Punto de entrada: ejecuta el pipeline completo de principio a fin.

    python main.py                  # entrenamiento + evaluación + SHAP + predicciones
    python main.py --skip-training  # reutiliza el modelo guardado en models/
"""

import argparse
import sys
import time

from src.evaluate import run_evaluation
from src.explain import run_explainability
from src.predict import run_test_set_predictions
from src.train import run_training
from src.utils import get_logger

logger = get_logger("main")


def main() -> int:
    parser = argparse.ArgumentParser(description="Customer churn prediction pipeline.")
    parser.add_argument("--skip-training", action="store_true",
                        help="Reuse the model saved in models/ instead of retraining.")
    args = parser.parse_args()

    steps = [("Evaluation", run_evaluation), ("SHAP explainability", run_explainability),
             ("Customer predictions", run_test_set_predictions)]
    if not args.skip_training:
        steps.insert(0, ("Training", run_training))

    start = time.perf_counter()
    for name, step in steps:
        logger.info("===== %s =====", name)
        try:
            step()
        except Exception:
            logger.exception("Step '%s' failed", name)
            return 1
    logger.info("Pipeline finished in %.1f s", time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
