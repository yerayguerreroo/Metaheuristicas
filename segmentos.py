import random


#Falta añadir la condicion de que los segmentos sean minimo de 2 elementos
def generar_segmentos(num_segmentos, numero_puntos_serie):
    num_cortes = num_segmentos - 1
    if numero_puntos_serie < (num_cortes + 1) * 2:
        raise ValueError(f"La serie es muy corta ({numero_puntos_serie} puntos) para hacer {num_cortes} cortes con segmentos de mínimo 2 puntos.")
    
    cortes = []

    while True:
        # Generamos una muestra aleatoria pura en el rango total
        cortes = random.sample(range(2, numero_puntos_serie - 2), num_cortes)
        cortes.sort()
        
        # Comprobamos si la distancia mínima es respetada
        distancia_valida = True
        for i in range(1, num_cortes):
            if cortes[i] - cortes[i-1] < 2:
                distancia_valida = False
                break
                
        # Si todos los elementos cumplen la distancia >= 2, devolvemos el vector
        if distancia_valida:
            return cortes