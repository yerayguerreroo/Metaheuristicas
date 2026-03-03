import numpy as np
import mrse
import segmentos
import auxiliar
import cronometro

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

def es_valido(cortes, n):
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

def hill_climbing(datos, archivo_txt, k_segmentos):
    """
    Implementación de Hill Climbing para optimizar los puntos de corte.
    """
    cronometro.comenzar_cronometro()
    
    y = datos
    n = len(y)
    
    solucion_actual = segmentos.generar_segmentos(k_segmentos, n)
    error_inicial = calcular_error_total(y, solucion_actual)
    error_actual = error_inicial # Guardamos el inicial intacto
    
    print(f"--- Iniciando Hill Climbing para {archivo_txt} (k={k_segmentos}) ---")
    print(f"Error inicial: {error_inicial:.6f}")

    mejorando = True
    paso_iteracion = 0
    
    while mejorando:
        mejorando = False
        for i in range(len(solucion_actual)):
            valor_original = solucion_actual[i]
            
            for movimiento in [-1, 1]:
                nueva_solucion = list(solucion_actual)
                nueva_solucion[i] += movimiento
                
                if es_valido(nueva_solucion, n):
                    nuevo_error = calcular_error_total(y, nueva_solucion)
                    
                    if nuevo_error < error_actual:
                        error_actual = nuevo_error
                        solucion_actual = nueva_solucion
                        mejorando = True
        
        paso_iteracion += 1
        if paso_iteracion % 10 == 0:
            print(f"Iteración {paso_iteracion}, Error actual: {error_actual:.6f}")

    print(f"Máximo local (óptimo de error) alcanzado.")

    # 3. Devolvemos un diccionario para que la función de reporte lo pueda procesar
    return {
        "solucion": solucion_actual,
        "rmse_inicial": error_inicial,
        "rmse_final": error_actual,
        "tiempo": cronometro.parar_cronometro()
    }


def hill_climbing_maxima_pendiente(datos, archivo_txt, k_segmentos):
    """
    Implementación de Hill Climbing de Máxima Pendiente (Steepest Descent).
    Evalúa todos los vecinos y escoge el que produzca la mayor mejora.
    """
    cronometro.comenzar_cronometro()

    y = datos
    n = len(y)
    
    solucion_actual = segmentos.generar_segmentos(k_segmentos, n)
    error_inicial = calcular_error_total(y, solucion_actual)
    error_actual = error_inicial # Guardamos el inicial intacto
    
    print(f"--- Iniciando Hill Climbing de Máxima Pendiente para {archivo_txt} (k={k_segmentos}) ---")
    print(f"Error inicial: {error_inicial:.6f}")

    mejorando = True
    paso_iteracion = 0
    
    while mejorando:
        mejorando = False
        
        # Variables para rastrear el MEJOR vecino de toda la vecindad en esta iteración
        mejor_vecino = None
        mejor_error_vecino = error_actual
        
        # Evaluamos TODA la vecindad
        for i in range(len(solucion_actual)):
            for movimiento in [-1, 1]:
                nueva_solucion = list(solucion_actual)
                nueva_solucion[i] += movimiento
                
                # Solo evaluamos si el movimiento es válido
                if es_valido(nueva_solucion, n):
                    nuevo_error = calcular_error_total(y, nueva_solucion)
                    
                    # Guardamos el vecino solo si es ESTRICTAMENTE MEJOR que 
                    # el mejor que hemos encontrado hasta ahora en esta iteración
                    if nuevo_error < mejor_error_vecino:
                        mejor_error_vecino = nuevo_error
                        mejor_vecino = nueva_solucion
        
        # Una vez evaluados todos los vecinos, decidimos si nos movemos
        if mejor_vecino is not None:
            error_actual = mejor_error_vecino
            solucion_actual = mejor_vecino
            mejorando = True  # Hubo mejora, repetimos el bucle
        
        paso_iteracion += 1
        if paso_iteracion % 10 == 0:
            print(f"Iteración {paso_iteracion}, Error actual: {error_actual:.6f}")

    print(f"Mínimo local alcanzado tras {paso_iteracion} iteraciones.")

    # 3. Devolvemos el diccionario con la misma estructura que el HC simple
    return {
        "solucion": solucion_actual,
        "rmse_inicial": error_inicial,
        "rmse_final": error_actual,
        "tiempo": cronometro.parar_cronometro()
    }