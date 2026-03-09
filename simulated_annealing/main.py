import segmentos
import simulated_annealing.metrics as metrics
from auxiliar import cargar_datos
import numpy as np
import copy 
import math
import random
from .plotting import plot_series_with_piecewise_lines
import cronometro

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

def simulated_annealing(T0, alpha, Tf, file, serie):
    
    T = T0
    n = len(serie)
    cont = 0 #aceptados
    mejores = 0
    tot = 0 #hechos

    cronometro.comenzar_cronometro()
    s = segmentos.generar_segmentos(file['k'], n)
    origen = s

    #Cache para hacer más eficiente esto
    # Simplemente para ahorrar recursos ya esta
    #Nos ahorra muchos usos de RMSE()
    rmse_cache = {}

    def rmse_of(cuts):
        key = tuple(cuts)
        if key not in rmse_cache:
            rmse_cache[key] = metrics.global_rmse_for_cuts(serie, cuts)
        return rmse_cache[key]

    RMSE_s = rmse_of(s)
    og_err = RMSE_s

    best_ans = s.copy()
    best_rmse = RMSE_s

    # step_max = int(round(.07 * n)) # step_max = 7% de los puntos

    while T >= Tf:
        # Se podría meter en una función
        
        # mp = (T-Tf) / (T0-Tf)
        # step = max(1, int(round(step_max * mp)))
        step = 1;
        # max_L = step * 4
        # min_L = int(round(step * 2))
        L = min(k, 20);

        for _ in range(L):
            s_cand = generarVecino(s, len(serie), step) #Segmento con el nuevo corte en base al vecindario de s, posicion aleatoria
            if s_cand is None : continue
            # Casi imposible pero asi tratamos la excepcion de generarVecino()

            RMSE_cand = rmse_of(s_cand)
            delta = RMSE_cand - RMSE_s
            # U(0,1) = random.random() 
            # e^(-S/T) para T baja e^(-S/T) aprox 0 => No se cumple la condicion

            rn = random.random()
            if delta < 0 or rn < math.exp(-delta / T):
                s = s_cand
                RMSE_s = RMSE_cand
                cont+=1

                if RMSE_s < best_rmse:
                    mejores+=1
                    best_rmse = RMSE_s
                    best_ans = s.copy()

                # print(f"{s_cand} : delta = {delta} : improved => {delta < 0} : {rn} < {math.exp(-delta / T)}")
            tot+=1


        T *= alpha # alpha < 1 -> T disminuye (cooling)

    # print(f"step = {step_max}")
    # print(f"points = {len(serie)}")
    # print(f"%step = {step/len(serie)}")

    # print(f"cont = {cont}")
    # print(f"tot = {tot}")
    # print(f"mejores = {mejores}")
    # print(f"%cambios = {(cont/tot)*100:.3f}\n")

    # print(f"RESULTADO INICIAL {file['file']}:")
    # print(f"Cortes: {origen}")
    # print(f"RMSE Promedio: {og_err:.6f}\n")
    # print(f"RESULTADO FINAL {file['file']}:")
    # print(f"Cortes: {best_ans}")
    # print(f"RMSE Promedio: {best_rmse:.6f}\n")

    # plot_series_with_piecewise_lines(
    #     serie,
    #     origen,
    #     show_cuts=True,
    #     title=f"{file['file']} + rectas por segmentos (solución inicial)",
    #     save_path="./simulated_annealing/inicios/resultado_ts1.png",
    # )


    # plot_series_with_piecewise_lines(
    #     serie,
    #     best_ans,
    #     show_cuts=True,
    #     title=f"{file['file']} + rectas por segmentos (mejor solución)",
    #     save_path="./simulated_annealing/resultados/resultado_ts1.png",
    # )

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