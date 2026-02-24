# search.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import segmentos
from .metrics import global_rmse_for_cuts


@dataclass(frozen=True)
class SearchConfig:
    """Parámetros de la búsqueda por épocas."""
    num_segmentos: int = 9
    epochs: int = 200
    candidates_per_epoch: int = 50
    p_random: float = 0.25
    seed: int = 123
    step_start: int = 6
    step_end: int = 1


def _epoch_step(epoch: int, epochs: int, step_start: int, step_end: int) -> int:
    """
    Reduce el tamaño de mutación linealmente:
      - al inicio: step_start
      - al final:  step_end
    """
    if epochs <= 1:
        return max(1, step_end)
    t = (epoch - 1) / (epochs - 1)
    step = round(step_start + (step_end - step_start) * t)
    return max(1, int(step))


def mutate_cuts(cuts: List[int], n: int, step: int) -> List[int]:
    """
    Mutación compatible con TU generador actual.

    Mantiene estas reglas (tal y como las impone tu generar_segmentos):
      - rango: cortes en [2, n-3]  (porque tú usas range(2, n-2))
      - orden creciente
      - separación mínima entre cortes: >= 2
    """
    if not cuts:
        return []

    lo, hi = 2, n - 3
    new_cuts = cuts[:]  # copia

    idx = random.randrange(len(new_cuts))
    new_cuts[idx] += random.randint(-step, step)

    # 1) clamp a rango permitido
    new_cuts[idx] = max(lo, min(hi, new_cuts[idx]))

    # 2) ordenar
    new_cuts.sort()

    # 3) reparar separación mínima (>=2)
    for i in range(1, len(new_cuts)):
        if new_cuts[i] - new_cuts[i - 1] < 2:
            new_cuts[i] = new_cuts[i - 1] + 2

    # 4) si nos pasamos por arriba, empujar hacia atrás
    if new_cuts[-1] > hi:
        new_cuts[-1] = hi
        for i in range(len(new_cuts) - 2, -1, -1):
            new_cuts[i] = min(new_cuts[i], new_cuts[i + 1] - 2)

    # 5) si nos pasamos por abajo, empujar hacia delante
    if new_cuts[0] < lo:
        new_cuts[0] = lo
        for i in range(1, len(new_cuts)):
            new_cuts[i] = max(new_cuts[i], new_cuts[i - 1] + 2)

    return new_cuts


def optimize_cuts(
    y: np.ndarray,
    cfg: SearchConfig,
    x: Optional[np.ndarray] = None,
) -> Tuple[List[int], float]:
    """
    Búsqueda por épocas:

    - Con probabilidad p_random: reinicio aleatorio usando TU segmentos.generar_segmentos()
    - Si no: mutación de la mejor solución
    - Objetivo: minimizar el RMSE global
    """
    random.seed(cfg.seed)
    y = np.asarray(y, dtype=float)
    n = len(y)

    best_cuts = segmentos.generar_segmentos(cfg.num_segmentos, n)
    best_score = global_rmse_for_cuts(y, best_cuts, x=x)
    print(f"Inicial | RMSE global={best_score:.6f} | cuts={best_cuts}")

    for epoch in range(1, cfg.epochs + 1):
        step = _epoch_step(epoch, cfg.epochs, cfg.step_start, cfg.step_end)
        improved = False

        for _ in range(cfg.candidates_per_epoch):
            if random.random() < cfg.p_random:
                candidate = segmentos.generar_segmentos(cfg.num_segmentos, n)
            else:
                candidate = mutate_cuts(best_cuts, n, step=step)

            score = global_rmse_for_cuts(y, candidate, x=x)

            if score < best_score:
                best_score = score
                best_cuts = candidate
                improved = True

        print(f"Epoch {epoch:4d} | RMSE global={best_score:.6f} | step={step} | improved={improved}")

    return best_cuts, best_score