# metrics.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import auxiliar


def _to_numpy_1d(arr) -> np.ndarray:
    """Convierte a np.array 1D de float."""
    return np.asarray(arr, dtype=float)


def _default_x(n: int) -> np.ndarray:
    """Eje x por defecto: índices 0..n-1."""
    return np.arange(n, dtype=float)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fit_line(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    """
    Ajusta la recta y = m*x + b por mínimos cuadrados.
    Devuelve (m, b).
    """
    A = np.column_stack([xs, np.ones_like(xs)])
    m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(m), float(b)


def es_valido(cortes, n ):
    """
    Verifica que los cortes sean estrictamente crecientes y 
    mantengan el tamaño mínimo de 2 puntos por segmento.
    """
    if any(cortes[i] >= cortes[i+1] for i in range(len(cortes)-1)):
        return False
    # El primer segmento: de 0 a cortes[0] (debe tener al menos 2 puntos: 0 y 1)
    if cortes[0] < 1: return False
    # El último segmento: de cortes[-1] a n-1
    if cortes[-1] > n - 2: return False
    
    # Entre cortes (segmentos compartidos c_i a c_i+1)
    # Para que un segmento tenga 2 puntos, la diferencia debe ser al menos 1 
    # (ej. corte en 5 y corte en 6 incluye puntos 5 y 6)
    for i in range(len(cortes)-1):
        if cortes[i+1] - cortes[i]< 1:
            return False
    return True


def fit_segment_and_rmse(
    y: np.ndarray,
    start: int,
    end: int,
    x: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Ajusta una recta al segmento y[start:end] y devuelve (m, b, rmse_segmento).

    - end es EXCLUIDO (slicing de Python)
    - si el segmento tiene <2 puntos => NaN (no hay ajuste lineal fiable)
    """
    y = _to_numpy_1d(y)
    x = _default_x(len(y)) if x is None else _to_numpy_1d(x)

    if len(x) != len(y):
        raise ValueError("x e y deben tener la misma longitud.")
    if not (0 <= start < end <= len(y)):
        raise ValueError("Segmento inválido.")
    if end - start < 2:
        return np.nan, np.nan, np.nan

    xs = x[start:end]
    ys = y[start:end]

    m, b = fit_line(xs, ys)
    y_hat = m * xs + b
    return m, b, rmse(ys, y_hat)


def global_rmse_for_cuts(y: np.ndarray, cuts: List[int], x: Optional[np.ndarray] = None) -> float:
    """
    RMSE global de la aproximación por tramos lineales.

    OJO: tus cortes se COMPARTEN (corte = último del anterior y primero del siguiente).
    Para no contar ese punto dos veces en el RMSE global:
      - En todos los segmentos excepto el primero, ignoramos el primer punto.
    """
    y = _to_numpy_1d(y)
    n = len(y)
    x = _default_x(n) if x is None else _to_numpy_1d(x)

    segments = auxiliar.cuts_to_segments_shared(cuts, n)

    total_sse = 0.0
    total_n = 0

    for seg_idx, (start, end) in enumerate(segments):
        m, b, _ = fit_segment_and_rmse(y, start, end, x=x)
        if np.isnan(m) or np.isnan(b):
            return float("inf")  # segmento demasiado corto

        xs = x[start:end]
        ys = y[start:end]
        y_hat = m * xs + b

        # Evitar doble conteo del punto compartido
        if seg_idx > 0:
            ys = ys[1:]
            y_hat = y_hat[1:]

        total_sse += float(np.sum((ys - y_hat) ** 2))
        total_n += len(ys)

    return float(np.sqrt(total_sse / total_n)) if total_n > 0 else float("inf")


def segment_report(
    y: np.ndarray,
    cuts: List[int],
    x: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Devuelve (y opcionalmente imprime) el detalle por segmento:
    start, end, m, b, rmse_segmento.
    """
    y = _to_numpy_1d(y)
    x = _default_x(len(y)) if x is None else _to_numpy_1d(x)

    segments = auxiliar.cuts_to_segments_shared(cuts, len(y))
    report: List[Dict] = []

    for start, end in segments:
        m, b, seg_rmse = fit_segment_and_rmse(y, start, end, x=x)
        row = {"start": start, "end": end, "m": m, "b": b, "rmse": seg_rmse}
        report.append(row)

        if verbose:
            print(f"Segmento [{start}:{end}] -> m={m:.4f}, b={b:.4f}, RMSE={seg_rmse:.4f}")

    return report