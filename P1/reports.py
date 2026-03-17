import statistics as stats
from excel import exportar_reporte_excel
import random

def reporte_SA(simulated_annealing_fn, repeticiones,
                      T0, alpha, Tf, file, serie):
    

    def G(rmse_0, rmse_f):
      return (rmse_0 - rmse_f) / rmse_0 * 100

    resultados = []

    for i in range(repeticiones):
        random.seed()  # distinta semilla cada vez

        res = simulated_annealing_fn(T0, alpha, Tf, file, serie)
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
      "file": file['file'],
      "repeticiones": repeticiones,
      "T0": T0,
      "Tf": Tf,
      "alpha": alpha,
      "media_rmse": stats.mean(rmses),
      "std_rmse": stats.stdev(rmses) if len(rmses) > 1 else 0,
      "%_rmse": stats.mean(mejoras),
      "min_rmse": min(rmses),
      "max_rmse": max(rmses),
      "media_tiempo": stats.mean(tiempos),
      "std_tiempo": stats.stdev(tiempos) if len(tiempos) > 1 else 0,
      "resultados_individuales": resultados
    }

    file_serie = file['file'].split('.')[0]
    print(f"./simulated_annealing/excel/{file_serie}/")
    path = exportar_reporte_excel(resumen, f"./simulated_annealing/excel/{file_serie}")
    print(f"Guardado en {path}")


def reporte_RS(random_search_fn, repeticiones, file, serie):
    # Extraemos los valores del diccionario
    nombre_archivo = file['file']
    k_segmentos = file['k']
    
    def G(rmse_0, rmse_f):
        if rmse_0 == 0: return 0 
        return (rmse_0 - rmse_f) / rmse_0 * 100

    resultados = []

    for i in range(repeticiones):
        random.seed()  # distinta semilla cada vez

        # Llamamos a Random Search con la misma estructura
        res = random_search_fn(serie, nombre_archivo, k_segmentos)
        resultados.append(res)

    rmses = [r["rmse_final"] for r in resultados]
    mejoras = [G(r["rmse_inicial"], r["rmse_final"]) for r in resultados]
    tiempos = [r["tiempo"] for r in resultados]

    print("\n===== REPORTE ESTADÍSTICO RANDOM SEARCH =====")
    print(f"Archivo: {nombre_archivo} | Segmentos (k): {k_segmentos}")
    print(f"Ejecuciones: {repeticiones}")
    print(f"Media RMSE: {stats.mean(rmses):.6f}")
    print(f"Desviación típica RMSE: {stats.stdev(rmses) if len(rmses) > 1 else 0:.6f}")
    print(f"Mejora Media (G): {stats.mean(mejoras):.2f}%")
    print(f"Mejor RMSE: {min(rmses):.6f}")
    print(f"Peor RMSE: {max(rmses):.6f}")
    print(f"Media tiempo: {stats.mean(tiempos):.3f}s")
    print(f"Desviación tiempo: {stats.stdev(tiempos) if len(tiempos) > 1 else 0:.3f}s\n")

    resumen = {
      "file": nombre_archivo,
      "k_segmentos": k_segmentos,
      "repeticiones": repeticiones,
      "media_rmse": stats.mean(rmses),
      "std_rmse": stats.stdev(rmses) if len(rmses) > 1 else 0,
      "%_rmse": stats.mean(mejoras),
      "min_rmse": min(rmses),
      "max_rmse": max(rmses),
      "media_tiempo": stats.mean(tiempos),
      "std_tiempo": stats.stdev(tiempos) if len(tiempos) > 1 else 0,
      "resultados_individuales": resultados
    }

    file_serie = nombre_archivo.split('.')[0]
    ruta_guardado = f"./random_search/excel/{file_serie}"
    print(f"{ruta_guardado}/")
    
    path = exportar_reporte_excel(resumen, ruta_guardado)
    print(f"Guardado en {path}")


def reporte_HC_Simple(hill_climbing_fn, repeticiones, file, serie):
    
    def G(rmse_0, rmse_f):
        if rmse_0 == 0: return 0 
        return (rmse_0 - rmse_f) / rmse_0 * 100

    resultados = []

    for i in range(repeticiones):
        random.seed()  # distinta semilla cada vez

        # Llamamos a Hill Climbing con los parámetros originales
        res = hill_climbing_fn(serie, file['file'], file['k'])
        resultados.append(res)

    rmses = [r["rmse_final"] for r in resultados]
    mejoras = [G(r["rmse_inicial"], r["rmse_final"]) for r in resultados]
    tiempos = [r["tiempo"] for r in resultados]

    print("\n===== REPORTE ESTADÍSTICO HILL CLIMBING =====")
    print(f"Archivo: {file['file']}")
    print(f"Ejecuciones: {repeticiones}")
    print(f"Media RMSE: {stats.mean(rmses):.6f}")
    print(f"Desviación típica RMSE: {stats.stdev(rmses) if len(rmses) > 1 else 0:.6f}")
    print(f"Mejora Media (G): {stats.mean(mejoras):.2f}%")
    print(f"Mejor RMSE: {min(rmses):.6f}")
    print(f"Peor RMSE: {max(rmses):.6f}")
    print(f"Media tiempo: {stats.mean(tiempos):.3f}s")
    print(f"Desviación tiempo: {stats.stdev(tiempos) if len(tiempos) > 1 else 0:.3f}s\n")

    resumen = {
      "file": file['file'],
      "repeticiones": repeticiones,
      "media_rmse": stats.mean(rmses),
      "std_rmse": stats.stdev(rmses) if len(rmses) > 1 else 0,
      "%_rmse": stats.mean(mejoras),
      "min_rmse": min(rmses),
      "max_rmse": max(rmses),
      "media_tiempo": stats.mean(tiempos),
      "std_tiempo": stats.stdev(tiempos) if len(tiempos) > 1 else 0,
      "resultados_individuales": resultados
    }

    file_serie = file['file'].split('.')[0]
    # Guardamos en la carpeta correspondiente a hill_climbing
    ruta_guardado = f"./hill_climbing/excelHCS/{file_serie}"
    print(f"{ruta_guardado}/")
    
    path = exportar_reporte_excel(resumen, ruta_guardado)
    print(f"Guardado en {path}")


def reporte_HC_Maxima_Pendiente(hill_climbing_fn, repeticiones, file, serie):
    
    def G(rmse_0, rmse_f):
        if rmse_0 == 0: return 0 
        return (rmse_0 - rmse_f) / rmse_0 * 100

    resultados = []

    for i in range(repeticiones):
        random.seed()  # distinta semilla cada vez

        # Llamamos a Hill Climbing con los parámetros originales
        res = hill_climbing_fn(serie, file['file'], file['k'])
        resultados.append(res)

    rmses = [r["rmse_final"] for r in resultados]
    mejoras = [G(r["rmse_inicial"], r["rmse_final"]) for r in resultados]
    tiempos = [r["tiempo"] for r in resultados]

    print("\n===== REPORTE ESTADÍSTICO HILL CLIMBING =====")
    print(f"Archivo: {file['file']}")
    print(f"Ejecuciones: {repeticiones}")
    print(f"Media RMSE: {stats.mean(rmses):.6f}")
    print(f"Desviación típica RMSE: {stats.stdev(rmses) if len(rmses) > 1 else 0:.6f}")
    print(f"Mejora Media (G): {stats.mean(mejoras):.2f}%")
    print(f"Mejor RMSE: {min(rmses):.6f}")
    print(f"Peor RMSE: {max(rmses):.6f}")
    print(f"Media tiempo: {stats.mean(tiempos):.3f}s")
    print(f"Desviación tiempo: {stats.stdev(tiempos) if len(tiempos) > 1 else 0:.3f}s\n")

    resumen = {
      "file": file['file'],
      "repeticiones": repeticiones,
      "media_rmse": stats.mean(rmses),
      "std_rmse": stats.stdev(rmses) if len(rmses) > 1 else 0,
      "%_rmse": stats.mean(mejoras),
      "min_rmse": min(rmses),
      "max_rmse": max(rmses),
      "media_tiempo": stats.mean(tiempos),
      "std_tiempo": stats.stdev(tiempos) if len(tiempos) > 1 else 0,
      "resultados_individuales": resultados
    }

    file_serie = file['file'].split('.')[0]
    # Guardamos en la carpeta correspondiente a hill_climbing
    ruta_guardado = f"./hill_climbing/excelHCMP/{file_serie}"
    print(f"{ruta_guardado}/")
    
    path = exportar_reporte_excel(resumen, ruta_guardado)
    print(f"Guardado en {path}")
