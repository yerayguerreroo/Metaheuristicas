import statistics as stats
from excel import exportar_reporte_excel
import random

def reporte_SA(simulated_annealing_fn, repeticiones,
                      T0, alpha, L, Tf, file, serie):
    

    def G(rmse_0, rmse_f):
      return (rmse_0 - rmse_f) / rmse_0 * 100

    resultados = []

    for i in range(repeticiones):
        random.seed()  # distinta semilla cada vez

        res = simulated_annealing_fn(T0, alpha, L, Tf, file, serie)
        resultados.append(res)

    rmses = [r["rmse_final"] for r in resultados]
    mejoras = [ G(r["rmse_inicial"], r["rmse_final"]) for r in resultados]
    tiempos = [r["tiempo"] for r in resultados]


    print("\n===== REPORTE ESTADÍSTICO =====")
    print(f"Ejecuciones: {repeticiones}")
    print(f"Media RMSE: {stats.mean(rmses):.6f}")
    print(f"Desviación típica RMSE: {stats.stdev(rmses):.6f}")
    print(f"Mejora Media (G): {stats.mean(mejoras)}%")
    print(f"Mejor RMSE: {min(rmses):.6f}")
    print(f"Peor RMSE: {max(rmses):.6f}")
    print(f"Media tiempo: {stats.mean(tiempos):.3f}s")
    print(f"Desviación tiempo: {stats.stdev(tiempos):.3f}s\n")


    #Debería de añadir los valores base para hacer el reporte mejor
    resumen = {
      "media_rmse": stats.mean(rmses),
      "std_rmse": stats.stdev(rmses) if len(rmses) > 1 else 0,
      "%_rmse": stats.mean(mejoras),
      "min_rmse": min(rmses),
      "max_rmse": max(rmses),
      "media_tiempo": stats.mean(tiempos),
      "std_tiempo": stats.stdev(tiempos) if len(tiempos) > 1 else 0,
      "resultados_individuales": resultados
    }

    path = exportar_reporte_excel(resumen, "./simulated_annealing/excel/")
    print(f"Guardado en {path}")