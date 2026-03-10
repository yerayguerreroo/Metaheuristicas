import segmentos
import simulated_annealing.metrics as metrics
from auxiliar import cargar_datos, cuts_to_segments_shared
import numpy as np
import math
import random
import cronometro

def main(): 
    serie = cargar_datos("./TS1.txt") #ejemplo hardcoded con ts1
    return 0

def getL_mejorado(T0, T, num_puntos, num_segmentos):
    ratio = T / T0
    # Aumentamos la base. Factor depende de los cortes y los puntos
    base_L = (num_puntos / num_segmentos) * (num_segmentos - 1) 
    
    # mp va de 1.0 (al inicio, exploración rápida) a 2.5 (al final, búsqueda local profunda)
    mp = 1.0 + 1.5 * (1 - ratio) 
    
    L = max(20, int(round(base_L * mp)))
    return L


def simulated_annealing(T0, alpha, Tf, file, serie):
    
    T = T0
    n = len(serie)
    cont = 0 #aceptados
    mejores = 0
    tot = 0 #hechos
    k = file['k'];

    cronometro.comenzar_cronometro()
    s = segmentos.generar_segmentos(file['k'], n)
    origen = s

    #Cache para hacer más eficiente esto
    segment_sse_cache = {}

    def rmse_of(cuts):
        # 1. Convertimos los cortes en pares de segmentos (start, end)
        segments = cuts_to_segments_shared(cuts, n)
        
        total_sse = 0.0
        
        for start, end in segments:
            key = (start, end)
            
            # 2. Si este trozo exacto no está en caché, lo calculamos
            if key not in segment_sse_cache:
                
                # Usamos tu función para obtener la pendiente (m) y la intersección (b)
                m, b, _ = metrics.fit_segment_and_rmse(serie, start, end)
                
                if np.isnan(m) or np.isnan(b):
                    return float("inf") # Segmento inválido
                
                # Reconstruimos xs, ys y la predicción (igual que en tu metrics.py)
                xs = np.arange(start, end, dtype=float)
                ys = serie[start:end]
                y_hat = m * xs + b
                
                # 3. Tu lógica clave: Evitar doble conteo
                if start > 0:
                    ys = ys[1:]
                    y_hat = y_hat[1:]
                    
                # 4. Calculamos la Suma de Errores al Cuadrado (SSE) para este segmento
                sse = float(np.sum((ys - y_hat) ** 2))
                
                # Lo guardamos en caché
                segment_sse_cache[key] = sse
                
            # 5. Sumamos el SSE (ya sea recién calculado o sacado gratis de la caché)
            total_sse += segment_sse_cache[key]
            
        # 6. Devolvemos el RMSE global exacto. 
        # Como evitamos el doble conteo, el número de puntos totales evaluados siempre es n.
        return float(np.sqrt(total_sse / n))

    RMSE_s = rmse_of(s)
    og_err = RMSE_s

    best_ans = s.copy()
    best_rmse = RMSE_s

    TAM_VECINDARIO = 2 * (k-1);

    # step_max = int(round(.07 * n)) # step_max = 7% de los puntos

    while T >= Tf:
        # Se podría meter en una función
        
        step = 1

        L = getL_mejorado(T0, T, n, k)

        for _ in range(L):
            s_cand = generarVecino(s, len(serie), step) #Segmento con el nuevo corte en base al vecindario de s, posicion aleatoria
            if s_cand is None : continue
            # Casi imposible pero asi tratamos la excepcion de generarVecino()

            RMSE_cand = rmse_of(s_cand)
            delta = RMSE_cand - RMSE_s

            rn = random.random()
            if delta < 0 or rn < math.exp(-delta / T):
                s = s_cand
                RMSE_s = RMSE_cand
                cont+=1

                if RMSE_s < best_rmse:
                    mejores+=1
                    best_rmse = RMSE_s
                    best_ans = s.copy()

            tot+=1


        T *= alpha


    return {
        "rmse_inicial": og_err,
        "rmse_final": best_rmse,
        "mejor_sol": best_ans,
        "sol_inicial": origen,
        "aceptados": cont,
        "total_movimientos": tot,
        "mejoras": mejores,
        "tiempo": cronometro.parar_cronometro()
    }

def generarVecino(cuts, n, step):
    # Alternativa para no tener que generar un vecindario gigantesco

    attempts = 20

    for _ in range(attempts):
        v = cuts.copy()
        i = random.randrange(len(v))
        d = random.randint(1, step)
        if(random.random() < .5): sign = 1
        else: sign = -1
        v[i] += sign * d

        if metrics.es_valido(v, n):
            return v

    return None

if __name__ == "__main__":
    main()