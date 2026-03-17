# random_search/plotting.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

import auxiliar
from .metrics import fit_segment_and_rmse


def plot_series_with_piecewise_lines(
    y: np.ndarray,
    cuts: List[int],
    x: Optional[np.ndarray] = None,
    *,
    show_cuts: bool = False,
    title: str = "Serie + ajuste lineal por segmentos",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Dibuja:
      - La serie completa como línea "normal"
      - Las rectas ajustadas por segmento encima

    Cambios pedidos:
      - Sin leyenda
      - Serie sin puntos (sin markers), solo línea
      - Rectas por segmento encima
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    if x is None:
        x = np.arange(n, dtype=float)
    else:
        x = np.asarray(x, dtype=float)
        if len(x) != n:
            raise ValueError("x e y deben tener la misma longitud.")

    segments = auxiliar.cuts_to_segments_shared(cuts, n)

    plt.figure()

    # Serie como línea continua (sin puntos)
    plt.plot(x, y, linewidth=1)

    # Rectas por segmento
    for start, end in segments:
        m, b, _ = fit_segment_and_rmse(y, start, end, x=x)
        xs = x[start:end]
        y_hat = m * xs + b
        plt.plot(xs, y_hat, linewidth=2)

    # Opcional: marcar cortes con líneas verticales
    if show_cuts:
        for c in cuts:
            plt.axvline(x[c], linestyle="--", linewidth=1, alpha=0.4)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)

    # Guardado robusto: crea carpeta si no existe
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()