import numpy as np
from blackbox import BlackBoxModel
from main import run_genetic_algorithm

bb = BlackBoxModel("blackbox_modelA.pkl")

# Acumulador de puntos medios de pares ya encontrados
found_midpoints = []
#Guarda todos los pares de puntos
all_pairs = []

LAMBDA = 10.0   # penalización por clase igual (puedes usar M grande, ej. 100)
MU     = 0.5    # peso de la dispersión

def evaluate_solution(params):
    p1 = np.array([params[0], params[1]])
    p2 = np.array([params[2], params[3]])

    # 1. Distancia euclidea
    dist = np.linalg.norm(p1 - p2)

    # 2. Penalización de clase
    f1 = bb.predict(p1)
    f2 = bb.predict(p2)
    P_clase = 0.0 if f1 != f2 else LAMBDA

    # 3. Dispersión del punto medio respecto a los ya encontrados
    m = (p1 + p2) / 2.0
    if len(found_midpoints) == 0:
        disp = 0.0  # primer par: sin dispersión todavía
    else:
        disp = min(np.linalg.norm(m - mj) for mj in found_midpoints)

    fitness = dist + P_clase - MU * disp+

    for i in range(20):  # 20 pares de frontera
        print(f"\n=== Buscando par {i+1}/20 ===")
        params, fitness = run_genetic_algorithm(pop_size=30, generations=50)
        p1 = np.array([params[0], params[1]])
        p2 = np.array([params[2], params[3]])
        found_midpoints.append((p1 + p2) / 2.0)
        all_pairs.append((p1, p2))

    # El AG maximiza → devolvemos negativo (queremos minimizar fitness)
    return -fitness