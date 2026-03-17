import time

# Variable global para guardar el momento de inicio
inicio_tiempo = None

def comenzar_cronometro():
    global inicio_tiempo
    inicio_tiempo = time.time()
    print("Cronómetro iniciado...")

def parar_cronometro():
    global inicio_tiempo
    if inicio_tiempo is None:
        print("El cronómetro no ha sido iniciado.")
        return
    
    fin_tiempo = time.time()
    tiempo_transcurrido = fin_tiempo - inicio_tiempo
    
    print(f"\nCronómetro detenido.")
    print(f"Tiempo total: {tiempo_transcurrido:.2f} segundos\n")
    
    # Reiniciamos la variable para el siguiente uso
    inicio_tiempo = None

    return tiempo_transcurrido;