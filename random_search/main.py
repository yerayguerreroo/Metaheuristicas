# random_search/main.py
from .metrics import segment_report
from .search import SearchConfig, optimize_cuts
from .plotting import plot_series_with_piecewise_lines
import cronometro


def random_search(serie, num_segmentos) -> None:

    cfg = SearchConfig(
        num_segmentos,
        epochs=50000,
    )

    cronometro.comenzar_cronometro()
    best_cuts, best_rmse = optimize_cuts(serie, cfg)

    print("\n===== MEJOR SOLUCIÓN =====")
    print("Best cuts:", best_cuts)
    print("Best RMSE global:", best_rmse)

    print("\nDetalle por segmento:")
    segment_report(serie, best_cuts, verbose=True)

    cronometro.parar_cronometro()
    # --- DIBUJO ---
    plot_series_with_piecewise_lines(
        serie,
        best_cuts,
        show_cuts=True,
        title="TS1 + rectas por segmentos (mejor solución)",
        save_path="./random_search/resultados/resultado_ts1.png",  # p.ej. "resultado_ts1.png" si quieres guardarlo
    )


if __name__ == "__main__":
    random_search()