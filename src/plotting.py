"""Estilo gráfico común para todas las figuras del proyecto.

Paleta categórica validada para daltonismo (orden fijo: azul, naranja, aqua),
texto en tinta neutra y rejilla discreta.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: permite generar figuras desde scripts y tests
import matplotlib.pyplot as plt  # noqa: E402

from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#8a8985"
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
GRID = "#e4e3df"

# Colores por entidad (se mantienen en todas las figuras)
CHURN_COLORS = {"No": BLUE, "Yes": ORANGE}
MODEL_COLORS = {"Logistic Regression": BLUE, "XGBoost": ORANGE}
SEQUENTIAL_CMAP = "Blues"
DIVERGING_CMAP = "RdBu_r"


def apply_style() -> None:
    """Aplica un estilo limpio y legible a matplotlib/seaborn."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT_PRIMARY,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "font.size": 10,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Guarda y cierra una figura (evita fugas de memoria al generar muchas)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved figure %s", path.name)
