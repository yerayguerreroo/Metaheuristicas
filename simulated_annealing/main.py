import segmentos
import metrics
from auxiliar import cargar_datos
import numpy as np
import math
import random

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

def simulated_annealing(T0, alpha, L, Tf, serie):
    # alpha = [0,1]
    T = T0
    s = segmentos.generar_segmentos() #Funcion de los cortes
    while (T >= Tf):
        vecindario = generarVecindario(s, serie.size())
        L = len(vecindario)-1
        for count in range(0, L):
            s_cand = vecindario[count] #Segmento con el nuevo corte en base al vecindario de s
            delta = metrics.global_rmse_for_cuts(serie, s) - metrics.global_rmse_for_cuts(serie, s_cand)

            # U(0,1) = random.random() 
            # e^(-S/T) para T baja e^(-S/T) aprox 0 => No se cumple la condicion

            if (random.random() < math.exp(-delta / T)) or (delta < 0):
                s = s_cand
        T *= alpha # alpha < 1 -> T disminuye (cooling)
    return s

def generarVecindario(solucion, n):
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
    for count in range(len(solucion)):
        vecino = solucion.copy()
        vecino[count] -= 1 # [2,7] | [3,6]
        if (metrics.es_valido(vecino,n)):
            vecindario.append(vecino)
        vecino[count] += 2 # [4,7] | [3,8]
        if(metrics.es_valido(vecino,n)):
            vecindario.append(vecino)    
    
    return vecindario

if __name__ == "__main__":
    main()