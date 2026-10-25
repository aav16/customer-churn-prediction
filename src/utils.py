"""Utilidades compartidas: logging y lectura/escritura de JSON."""

import json
import logging
from pathlib import Path
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger con formato homogéneo para todo el proyecto."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def save_json(obj: Any, path: Path) -> None:
    """Guarda un objeto serializable como JSON legible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def load_json(path: Path) -> Any:
    """Carga un JSON; lanza un error claro si el fichero no existe."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}. Run `python main.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))
