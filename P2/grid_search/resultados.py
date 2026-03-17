import pandas as pd

def guardar_en_csv(mejor_accuracy, mejores_parametros, tiempo_ejecucion):
    df_resultados = pd.DataFrame([mejores_parametros])

    df_resultados['Mejor_accuracy'] = round(mejor_accuracy,4)
    df_resultados['Tiempo_Segundos'] = round(tiempo_ejecucion,2)
    df_resultados['Algoritmo'] = 'Grid Search'

    df_resultados.to_csv('resultados_grid_search.csv', sep = ';', index = False)
