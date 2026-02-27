import segmentos
import simulated_annealing.metrics as metrics
from auxiliar import cargar_datos
import numpy as np
import copy 
import math
import random
from .plotting import plot_series_with_piecewise_lines

def main(): 
    serie = cargar_datos("./TS1.txt") #ejemplo hardcoded con ts1
    return 0

# Input (T0, 𝛼, L, Tf)
#     T = T0
#     s = sol_actual
#     While T ≥ Tf Do:    
#         For count 1 To L Do:
#             sol_cand = vecino(s)
#             𝛿 = F(scand)-F(s)
#             if ( U(0,1) < e^(-𝛿/T) || 𝛿 < 0)
#                 s = sol_cand
#         T = aT
#     Return s

#calcular temperatura de manera relativa al tamaño de entrada
#Funcion objetivo: reducir el rmse global
#calcular L

def simulated_annealing(T0, alpha, L, Tf, k, serie):
    # alpha = [0,1]
    T = T0
    s = segmentos.generar_segmentos(k, len(serie)) 
    origen = s
    c_step = int(.4 * len(serie))
    vecindario = generarVecindario(s,len(serie),c_step)
    RMSE_s = metrics.global_rmse_for_cuts(serie, origen)
    og_err = RMSE_s

    while (T >= Tf):
        # Se podría meter en una función
        mp = (T-Tf) / (T0-Tf)
        step = max(1, int(round(c_step * mp)))

        for i in range(0, L):
            vecindario = generarVecindario(s, len(serie), step)
            pos = random.randint(0, len(vecindario)-1)
            s_cand = vecindario[pos] #Segmento con el nuevo corte en base al vecindario de s, posicion aleatoria
            delta = metrics.global_rmse_for_cuts(serie, s_cand) - RMSE_s
            # U(0,1) = random.random() 
            # e^(-S/T) para T baja e^(-S/T) aprox 0 => No se cumple la condicion

            if ((random.random() < math.exp(-delta / T)) or (delta < 0)):
                s = s_cand
                RMSE_s = metrics.global_rmse_for_cuts(serie, s)

            print(f"{s_cand} : delta = {delta} : improved => {delta < 0}")

        T *= alpha # alpha < 1 -> T disminuye (cooling)

    return origen, og_err, s, RMSE_s

def generarVecindario(solucion, n, step):
    # Creamos el vecindario entero para una solución s
    # Cada punto de corte se trata con un +-1;
    # return vecindario = [ [s_cand1] , [s_cand2] , ... , [s_candN] ]

    # (0,3) (3,7) (7,10) [3,7]
    # soluciones
    # (0,3) (3,6) (6,10) [3,6]
    # (0,3) (3,8) (8,10) [3,8]
    # (0,2) (2,7) (7,10) 
    # (0,2) (2,6) (6,10)
    # (0,2) (2,8) (8,10)
    # (0,4) (4,7) (7,10)
    # (0,4) (4,6) (6,10)
    # (0,4) (4,8) (8,10)

    vecindario = []
    seen = set() #hashset, para algo tenia que servir leetcode
    for count in range(len(solucion)):
        for d in range (1, step+1):
            v_up = solucion.copy()
            v_down = solucion.copy()

            v_up[count] += d
            t = tuple(v_up)
            if (metrics.es_valido(v_up,n)
                and t not in seen):
                print(f"{v_up}")
                vecindario.append(v_up)

            v_down[count] -= d
            t = tuple(v_down)
            if (metrics.es_valido(v_down,n)
                and t not in seen):
                print(f"{v_down}")
                vecindario.append(v_down)

    return vecindario

if __name__ == "__main__":
    main()