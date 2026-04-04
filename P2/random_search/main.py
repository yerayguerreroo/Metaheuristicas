import random
import time
import csv
import os
from random_forest import evaluate_solution

def random_search():
    iteraciones = 25000 # Hacer pruebas con 15000 20000 25000
    mejor_solucion = []
    mejor_score = 0.0
    tiempo_inicio = time.time()
    for i in range(iteraciones):
        solucion = generar_solucion()
        score = evaluate_solution(solucion)
        print(f"Iteración {i+1}/{iteraciones} - Score: {score:.4f}")
        if score > mejor_score:
            mejor_score = score
            mejor_solucion = solucion

    tiempo_fin = time.time()
    tiempo_total = tiempo_fin - tiempo_inicio

    print("-" * 40)
    print("BÚSQUEDA FINALIZADA")
    print(f"Tiempo total de ejecución: {tiempo_total:.2f} segundos")
    print(f"Mejor Score obtenido: {mejor_score:.4f}")
    print("-" * 40)
    
    # 3. Guardar todo en el CSV
    guardar_en_csv(iteraciones,tiempo_total, mejor_score, mejor_solucion)

    return mejor_solucion, mejor_score
        

def generar_solucion():
    # 1. n_estimators: entero 10 – 300
    n_estimators = random.randint(10, 300)
    
    # 2. max_depth: entero 2 – 30
    max_depth = random.randint(2, 30)
    
    # 3. min_samples_split: entero 2 – 20
    min_samples_split = random.randint(2, 20)
    
    # 4. min_samples_leaf: entero 1 – 20
    min_samples_leaf = random.randint(1, 20)
    
    # 5. max_features: real 0.1 – 1.0
    max_features = random.uniform(0.1, 1.0)
    
    # 6. bootstrap: binario 0 / 1
    bootstrap = random.randint(0, 1)
    
    # 7. criterion: categórico 0 = gini, 1 = entropy
    criterion = random.randint(0, 1)
    
    # 8. class_weight: binario 0 = None, 1 = balanced
    class_weight = random.randint(0, 1)
    
    # 9. max_leaf_nodes: entero 10 – 200
    max_leaf_nodes = random.randint(10, 200)
    
    # 10. min_impurity_decrease: real 0 – 0.1
    min_impurity_decrease = random.uniform(0.0, 0.1)
    
    # Agrupamos todos los valores en un vector
    resultados = [
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        bootstrap,
        criterion,
        class_weight,
        max_leaf_nodes,
        min_impurity_decrease
    ]
    
    return resultados

def guardar_en_csv(iteraciones, tiempo_total, mejor_score, mejor_solucion, nombre_archivo="resultados_random_search.csv", carpeta="random_search"):
    """
    Guarda las iteraciones, el tiempo total, el mejor score y la combinación 
    en un archivo CSV dentro de una carpeta específica. Añade los datos si ya existe.
    """
    # 1. Crear la carpeta si no existe
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
        
    ruta_completa = os.path.join(carpeta, nombre_archivo)
    
    # 2. Comprobar si el archivo ya existe (para saber si escribir los encabezados o no)
    archivo_existe = os.path.isfile(ruta_completa)
    
    encabezados = [
        "Iteraciones",
        "Tiempo_Total_Segundos", 
        "Mejor_Score", 
        "n_estimators", 
        "max_depth",
        "min_samples_split", 
        "min_samples_leaf", 
        "max_features",
        "bootstrap", 
        "criterion", 
        "class_weight", 
        "max_leaf_nodes",
        "min_impurity_decrease"
    ]
    
    # 3. Escribir los datos en modo 'a' (append) en lugar de 'w' (write)
    with open(ruta_completa, mode='a', newline='', encoding='utf-8') as archivo_csv:
        writer = csv.writer(archivo_csv)
        
        # Si el archivo NO existía, escribimos los encabezados primero
        if not archivo_existe:
            writer.writerow(encabezados)
        
        # Añadimos 'iteraciones' al principio de la fila y guardamos
        fila = [iteraciones, tiempo_total, mejor_score] + mejor_solucion
        writer.writerow(fila)
        
    print(f"\n[+] Resultados añadidos en '{ruta_completa}'.")