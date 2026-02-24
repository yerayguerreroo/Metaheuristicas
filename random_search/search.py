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
    """Parámetros de la búsqueda aleatoria."""
    num_segmentos: int = 9
    epochs: int = 200


def optimize_cuts(
    y: np.ndarray,
    cfg: SearchConfig,
    x: Optional[np.ndarray] = None,
) -> Tuple[List[int], float]:
    """
    Búsqueda puramente aleatoria por épocas:
    - Genera un vector de segmentos inicial y evalúa su RMSE.
    - En cada época, genera otro vector de segmentos nuevo.
    - Evalúa el RMSE de este nuevo candidato y conserva siempre el mejor.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    # 1. Genera el primer segmento y evalúa su RMSE medio
    best_cuts = segmentos.generar_segmentos(cfg.num_segmentos, n)
    best_score = global_rmse_for_cuts(y, best_cuts, x=x)
    print(f"Inicial | RMSE global={best_score:.6f} | cuts={best_cuts}")

    # 2. Repite el proceso el número de épocas indicada
    for epoch in range(1, cfg.epochs + 1):
        # Genera otro vector de segmentos de la misma manera
        candidate_cuts = segmentos.generar_segmentos(cfg.num_segmentos, n)
        
        # Evalúa el RMSE medio del nuevo candidato
        candidate_score = global_rmse_for_cuts(y, candidate_cuts, x=x)

        improved = False
        # Se queda con el mejor
        if candidate_score < best_score:
            best_score = candidate_score
            best_cuts = candidate_cuts
            improved = True

        print(f"Epoch {epoch:4d} | RMSE global={best_score:.6f} | improved={improved}")

    return best_cuts, best_score