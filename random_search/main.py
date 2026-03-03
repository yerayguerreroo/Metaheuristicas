# random_search/main.py
from .metrics import segment_report
from .search import SearchConfig, optimize_cuts
from .plotting import plot_series_with_piecewise_lines
import cronometro
import segmentos
import mrse
import numpy as np

def random_search(datos, archivo_txt, k_segmentos):
    cronometro.comenzar_cronometro()
    
    # Calculamos una solución inicial aleatoria para tener un RMSE base 
    # y así poder calcular el porcentaje de mejora en el reporte
    n = len(datos)
    solucion_inicial = segmentos.generar_segmentos(k_segmentos, n)
    error_inicial = calcular_error_total(datos, solucion_inicial)

    cfg = SearchConfig(
        k_segmentos,
        epochs=2000,
    )

    print(f"--- Iniciando Random Search para {archivo_txt} (k={k_segmentos}) ---")
    best_cuts, best_rmse = optimize_cuts(datos, cfg)

    print("\n===== MEJOR SOLUCIÓN =====")
    print("Best cuts:", best_cuts)
    print("Best RMSE global:", best_rmse)

    print("\nDetalle por segmento:")
    segment_report(datos, best_cuts, verbose=True)

    # --- DIBUJO ---
    # Hacemos el título y la ruta de guardado dinámicos según el archivo
    file_serie = archivo_txt.split('.')[0]
    # plot_series_with_piecewise_lines(
    #     datos,
    #     best_cuts,
    #     show_cuts=True,
    #     title=f"{file_serie} + rectas por segmentos (mejor solución RS)",
    #     save_path=f"./random_search/resultados/resultado_{file_serie}.png", 
    # )
    
    # 3. Devolvemos el diccionario estructurado
    return {
        "solucion": best_cuts,
        "rmse_inicial": error_inicial,
        "rmse_final": best_rmse,
        "tiempo": cronometro.parar_cronometro()
    }

def calcular_error_total(y, cortes):
    """
    Calcula el promedio de los RMSE de todos los segmentos definidos por los cortes.
    Utiliza tu función mrse.piecewise_linear_rmse_from_cuts.
    """
    # Tu función devuelve una lista de diccionarios con el RMSE de cada segmento
    resultados = mrse.piecewise_linear_rmse_from_cuts(y, cortes)
    
    # Extraemos solo los valores de RMSE
    rmses = [res['rmse'] for res in resultados]
    
    # Si hay algún NaN (segmento demasiado corto), devolvemos un error infinito
    if any(np.isnan(r) for r in rmses):
        return float('inf')
        
    return np.mean(rmses)


if __name__ == "__main__":
    random_search()