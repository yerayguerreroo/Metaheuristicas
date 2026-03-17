import numpy as np

def cuts_to_segments_shared(cuts, n):
    """
    cuts: lista de índices de corte (k-1), estrictamente crecientes
    n: longitud de la serie (len(y))

    Convención: cada corte c pertenece a ambos segmentos:
    [0..c1], [c1..c2], ..., [c_last..n-1]

    Devuelve: lista de tuplas (start, end) para slicing [start:end)
    """
    cuts = list(map(int, cuts))

    for i in range(len(cuts)-1):
        if(cuts[i] >= cuts[i+1]):
            raise ValueError("cuts debe ser estrictamente creciente.")

    if len(cuts) == 0:
        return [(0, n)]

    if cuts[0] < 0 or cuts[-1] > n - 1:
        raise ValueError("cuts fuera de rango.")

    segments = []
    start = 0
    for c in cuts:
        segments.append((start, c + 1))  # incluye el punto c
        start = c                        # el siguiente segmento empieza en c (lo comparte)
    segments.append((start, n))
    return segments


#Función para cargar datos
def cargar_datos(filename):
    try:
        with open(filename, 'r') as f:
            contenido = f.read().replace('[', '').replace(']', '').split()
            return np.array([float(x) for x in contenido])
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filename}")
        return None


