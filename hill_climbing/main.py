import numpy as np
import mrse
import segmentos
import auxiliar

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
    # 1. Cargar datos usando tu función de main.py
    y = datos
    n = len(y)
    
    # 2. Generar solución inicial usando tu función de segmentos.py
    solucion_actual = segmentos.generar_segmentos(k_segmentos, n)
    error_actual = calcular_error_total(y, solucion_actual)
    
    print(f"--- Iniciando Hill Climbing para {archivo_txt} (k={k_segmentos}) ---")
    print(f"Error inicial: {error_actual:.6f}")

    mejorando = True
    paso_iteracion = 0
    
    while mejorando:
        mejorando = False
        # 3. Iteramos por cada punto de corte para intentar moverlo
        for i in range(len(solucion_actual)):
            valor_original = solucion_actual[i]
            
            # Probamos mover el punto de corte a la izquierda (-1) y derecha (+1)
            for movimiento in [-1, 1]:
                nueva_solucion = list(solucion_actual)
                nueva_solucion[i] += movimiento
                
                # Solo evaluamos si el movimiento es válido
                if es_valido(nueva_solucion, n):
                    nuevo_error = calcular_error_total(y, nueva_solucion)
                    
                    # Si el error disminuye, aceptamos el cambio (esto es Hill Climbing)
                    if nuevo_error < error_actual:
                        error_actual = nuevo_error
                        solucion_actual = nueva_solucion
                        mejorando = True
                        # En Hill Climbing simple, una vez que mejoramos un punto, 
                        # podemos seguir con el siguiente o reiniciar
        
        paso_iteracion += 1
        if paso_iteracion % 10 == 0:
            print(f"Iteración {paso_iteracion}, Error actual: {error_actual:.6f}")

    print(f"Máximo local (óptimo de error) alcanzado.")
    return solucion_actual, error_actual

if __name__ == "__main__":
    # Configuración según la Práctica 1 (PDF)
    configuraciones = [
        ('TS1.txt', 9),
        ('TS2.txt', 10),
        # ('TS3.txt', 20),
        # ('TS4.txt', 50),
    ]
    
    for archivo, k in configuraciones:
        mejores_cortes, error_final = hill_climbing(archivo, k)
        print(f"RESULTADO FINAL {archivo}:")
        print(f"Cortes: {mejores_cortes}")
        print(f"RMSE Promedio: {error_final:.6f}\n")




