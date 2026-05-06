import numpy as np
import random
import sys
import os
import csv
import time
from blackbox import BlackBoxModel
from graficar import plot_individual_and_boundary

# ==============================================================================
# CONFIGURACIÓN GENERAL Y CONSTANTES
# ==============================================================================

MIN_PARES = 10   # mínimo 20 puntos
MAX_PARES = 50   # máximo 100 puntos

# Para A
X_MIN, X_MAX = -3.5, 2.0
Y_MIN, Y_MAX = -2.0, 3.5
# Para B -1.5 1.5
#X_MIN, X_MAX = -1.0, 1.0
#Y_MIN, Y_MAX = -1.0, 1.0

MAX_DISP_NORM = np.sqrt((X_MAX - X_MIN)**2 + (Y_MAX - Y_MIN)**2)

PATIENCE = 30  # generaciones consecutivas sin mejora para parar

# 1. Diagonal de tu espacio original optimizado [-1.5, 1.5]
# L_original = sqrt(3^2 + 3^2) = 4.2426...
DIAGONAL_ORIGINAL = np.sqrt((1.5 - (-1.5))**2 + (1.5 - (-1.5))**2)

# 2. Factor de proporción entre el espacio nuevo y el optimizado
FACTOR_ESCALA = MAX_DISP_NORM / DIAGONAL_ORIGINAL

# 3. Hiperparámetros base
LAMBDA_BASE = 100.0
DELTA_BASE  = 20.0
MU_BASE     = 12.0

# 4. Hiperparámetros ajustados
LAMBDA = LAMBDA_BASE
DELTA  = DELTA_BASE / FACTOR_ESCALA
MU     = MU_BASE / FACTOR_ESCALA
BONUS_POR_PUNTO = 0.0009  # Bonus leve por usar más puntos

bb = BlackBoxModel("blackbox_modelA.pkl")

def main():
    # Pedir el número de iteraciones por teclado
    try:
        n_reps_str = input("\nIntroduce el número de veces que deseas ejecutar el algoritmo: ")
        n_reps = int(n_reps_str)
        if n_reps <= 0:
            raise ValueError
    except ValueError:
        print("Valor inválido. Se ejecutará 1 vez por defecto.")
        n_reps = 1

    poblacion_dinamica = 200    
    generaciones_dinamicas = 600
    
    # Listas para almacenar los resultados de cada iteración
    fitnesses_obtenidos = []
    tiempos_obtenidos = []

    # Variables para rastrear al mejor absoluto de todas las ejecuciones
    mejor_individuo_global = None
    mejor_fitness_global = -float('inf')

    print(f"\n---> Iniciando {n_reps} ejecuciones del AG puntos) <---")

    for i in range(n_reps):
        print(f"\n{'='*50}")
        print(f"EJECUCIÓN {i+1} DE {n_reps}")
        print(f"{'='*50}")

        start_time = time.time()
        
        mejor_ind, mejor_fit = run_genetic_algorithm(
            pop_size=poblacion_dinamica, 
            generations=generaciones_dinamicas, 
            mutation_method='creep_dynamic',
            crossover_method='uniform', 
            adaptive_pc_pm='improvement'
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Guardar datos de esta iteración
        fitnesses_obtenidos.append(mejor_fit)
        tiempos_obtenidos.append(elapsed_time)

        # Comprobar si es el mejor histórico absoluto
        if mejor_fit > mejor_fitness_global:
            mejor_fitness_global = mejor_fit
            # Es vital usar deep_copy para no perder la referencia si el individuo muta luego
            mejor_individuo_global = deep_copy_individual(mejor_ind) 

    # --- CÁLCULO DE ESTADÍSTICAS ---
    media_fitness = np.mean(fitnesses_obtenidos)
    std_fitness = np.std(fitnesses_obtenidos)
    media_tiempo = np.mean(tiempos_obtenidos)
    std_tiempo = np.std(tiempos_obtenidos)

    print("\n" + "*"*50)
    print("RESUMEN ESTADÍSTICO DE LAS EJECUCIONES")
    print("*"*50)
    print(f"Mejor Fitness Absoluto : {mejor_fitness_global:.4f}")
    print(f"Fitness Medio          : {media_fitness:.4f} ± {std_fitness:.4f}")
    print(f"Tiempo Medio           : {media_tiempo:.4f}s ± {std_tiempo:.4f}s")
    print("*"*50)

    # Guardar en CSV
    guardar_estadisticas_csv(len(mejor_individuo_global), n_reps, media_fitness, std_fitness, media_tiempo, std_tiempo, mejor_fitness_global)

    # Pintar solo la mejor solución global encontrada
    if mejor_individuo_global is not None:
        plot_individual_and_boundary(mejor_individuo_global, bb, X_MIN, X_MAX, Y_MIN, Y_MAX)


# ==============================================================================
# 1. CREACIÓN DE LA POBLACIÓN
# ==============================================================================

def initialize_individual():
    """
    Un individuo tiene un número VARIABLE de pares de puntos.
    Se sortea aleatoriamente entre MIN_PARES y MAX_PARES pares.
    """
    num_pares = random.randint(MIN_PARES, MAX_PARES)
    puntos = []
    for _ in range(num_pares * 2):   # cada par son 2 puntos
        x = random.uniform(X_MIN, X_MAX)
        y = random.uniform(Y_MIN, Y_MAX)
        puntos.append([x, y])
    return puntos

def initialize_population(pop_size):
    """
    Crea la población inicial. Ya no recibe num_points porque
    cada individuo decide su propio tamaño al crearse.
    """
    return [initialize_individual() for _ in range(pop_size)]

# ==============================================================================
# 2. EVALUACIÓN Y FITNESS
# ==============================================================================

def evaluate_solution(params):
    """
    Evalúa un individuo que es un vector 2D de N puntos.
    params: [[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...]
    """
    # Convertimos a un array de NumPy para facilitar las matemáticas
    puntos = np.array(params)
    total_fitness = 0.0
    num_pares = len(puntos) / 2
    
    # 1. Calcular TODOS los midpoints del individuo de una vez
    p1s = puntos[0::2]
    p2s = puntos[1::2]
    midpoints_individuo = (p1s + p2s) / 2.0

    # 2. PREDICCIÓN EN BATCH (¡Súper rápido!)
    all_points = np.vstack([p1s, p2s])
    all_preds  = bb.predict(all_points) 
    f1s = all_preds[:len(p1s)]
    f2s = all_preds[len(p1s):]

    # 1. Distancias euclídeas de todos los pares de golpe
    dists = np.linalg.norm(p1s - p2s, axis=1)

    # 2. Penalizaciones de clase de todos los pares de golpe
    # Si son distintos, penalización 0.0, si son iguales, LAMBDA
    p_clases = np.where(f1s != f2s, 0.0, LAMBDA)

    # 3. Dispersión: Matriz de distancias entre todos los midpoints de golpe (Broadcasting)
    if len(midpoints_individuo) > 1:
        # Restamos cada midpoint con todos los demás simultáneamente
        diffs = midpoints_individuo[:, np.newaxis, :] - midpoints_individuo[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diffs, axis=2)
        
        # Llenamos la diagonal con infinito para que un punto no se encuentre a sí mismo como el más cercano
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Cogemos la distancia mínima para cada punto
        disps = np.min(dist_matrix, axis=1)
    else:
        # Si solo hay un par, la dispersión es la máxima por defecto
        disps = np.array([MAX_DISP_NORM])

    # Normalizamos todas las dispersiones y las limitamos a MAX_DISP_NORM
    disps_normalizadas = np.clip(disps * num_pares, 0.0, MAX_DISP_NORM)

    # 4. Calcular el fitness final sumando los arrays vectorizados
    fitness_por_par = (-DELTA * dists - p_clases + MU * disps_normalizadas) / num_pares
    
    total_fitness = np.sum(fitness_por_par)

    return total_fitness

# ==============================================================================
# 3. OPERADORES DE SELECCIÓN
# ==============================================================================

def tournament_selection(population, fitnesses, k=3, tolerance=0.2):
    """
    Selecciona un padre mediante torneo de tamaño k.
    Si el mejor individuo y otros tienen un fitness muy similar (dentro
    del margen de 'tolerance'), prefiere al que tenga mayor número de puntos.
    """
    selected_indices = random.sample(range(len(population)), k)
    
    # Ordenar los candidatos seleccionados de mejor a peor fitness
    candidates = sorted(selected_indices, key=lambda idx: fitnesses[idx], reverse=True)
    
    best_idx = candidates[0]
    best_fit = fitnesses[best_idx]
    
    # Variables para buscar al "mejor por desempate"
    chosen_idx = best_idx
    max_pares = len(population[best_idx]) // 2
    
    # Calculamos un umbral absoluto de tolerancia (ej: 5% del mejor fitness)
    # Como el fitness puede ser negativo, usamos valor absoluto para el margen
    margen = tolerance * abs(best_fit) if best_fit != 0 else tolerance
    
    # Comparamos al mejor con los finalistas
    for idx in candidates[1:]:
        fit = fitnesses[idx]
        
        # Si el fitness del candidato está dentro del margen de tolerancia del mejor...
        if abs(best_fit - fit) <= margen:
            pares_actual = len(population[idx]) // 2
            # ... y tiene más puntos, le damos preferencia
            if pares_actual > max_pares:
                max_pares = pares_actual
                chosen_idx = idx
        else:
            # Como están ordenados por fitness, si este ya no cumple la tolerancia, 
            # los siguientes tampoco lo harán.
            break
            
    return population[chosen_idx]

# ==============================================================================
# 4. OPERADORES DE CRUCE
# ==============================================================================

def random_size_pool_crossover(parent1, parent2):
    """
    Cruce de tamaño aleatorio acotado por los padres.
    
    1. Calcula el número de pares de cada padre.
    2. Elige un tamaño para el hijo aleatorio entre el min y max de los padres.
    3. Llena el hijo tomando pares aleatorios del conjunto total de ambos padres.
    """
    # 1. Extraer los pares de cada padre (cada par son 2 puntos)
    pares_p1 = [parent1[i:i+2] for i in range(0, len(parent1), 2)]
    pares_p2 = [parent2[i:i+2] for i in range(0, len(parent2), 2)]
    
    # 2. Determinar el tamaño del hijo
    min_pares = min(len(pares_p1), len(pares_p2))
    max_pares = max(len(pares_p1), len(pares_p2))
    
    # Si ambos padres tienen el mismo tamaño, randint devuelve ese mismo valor
    num_pares_hijo = random.randint(min_pares, max_pares)
    
    # 3. Crear el pool combinando todos los pares de ambos padres
    pool = pares_p1 + pares_p2
    
    if len(pool) == 0:
        return initialize_individual() # Salvaguarda de seguridad
        
    # 4. Seleccionar aleatoriamente los pares para el hijo (sin reemplazo)
    # Usamos random.sample para asegurarnos de tomar pares diferentes del pool
    pares_hijo = random.sample(pool, num_pares_hijo)
    
    # 5. Aplanar la lista de pares -> lista secuencial de puntos para el hijo
    child = [list(punto) for par in pares_hijo for punto in par]
    
    return child

def uniform_crossover(parent1, parent2):
    """
    Cruce uniforme adaptado para individuos de longitud variable.
    Cruza los pares en las posiciones coincidentes al 50%.
    Los pares sobrantes del padre más largo se heredan con un 50% de probabilidad.
    """
    # 1. Agrupar los puntos secuenciales en sus respectivos pares lógicos
    pares_p1 = [parent1[i:i+2] for i in range(0, len(parent1), 2)]
    pares_p2 = [parent2[i:i+2] for i in range(0, len(parent2), 2)]
    
    child_pares = []
    
    min_len = min(len(pares_p1), len(pares_p2))
    max_len = max(len(pares_p1), len(pares_p2))
    
    # 2. Cruce uniforme clásico para la sección donde ambos padres tienen pares
    for i in range(min_len):
        if random.random() < 0.5:
            # Aseguramos copia profunda usando list()
            child_pares.append([list(pares_p1[i][0]), list(pares_p1[i][1])])
        else:
            child_pares.append([list(pares_p2[i][0]), list(pares_p2[i][1])])
            
    # 3. Tratamiento de los genes excedentes del padre más largo
    padre_largo = pares_p1 if len(pares_p1) > len(pares_p2) else pares_p2
    for i in range(min_len, max_len):
        # 50% de probabilidad de heredar cada par extra (regula el tamaño final)
        if random.random() < 0.5:
            child_pares.append([list(padre_largo[i][0]), list(padre_largo[i][1])])

    # 4. Aplanar la lista de pares para devolver el formato original de lista de puntos
    child = [punto for par in child_pares for punto in par]
    
    return child

# ==============================================================================
# 4. OPERADORES DE MUTACIÓN
# ==============================================================================

def mutate_random(individual, mutation_rate):
    """Mutación aleatoria: resetea el gen a un valor completamente nuevo dentro de sus límites."""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i][0] = random.uniform(X_MIN, X_MAX)
            individual[i][1] = random.uniform(Y_MIN, Y_MAX)
    return individual

def mutate_creep(individual, mutation_rate, creep_scale=0.025):
    """
    Mutación por perturbación (creep): suma un pequeño delta al punto existente.
    Hace ajustes finos cerca de buenas soluciones en lugar de destruirlas con un
    salto aleatorio grande.
    """
    rango_x = X_MAX - X_MIN
    rango_y = Y_MAX - Y_MIN
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            dx = random.uniform(-creep_scale * rango_x, creep_scale * rango_x)
            dy = random.uniform(-creep_scale * rango_y, creep_scale * rango_y)
            individual[i][0] = max(X_MIN, min(X_MAX, individual[i][0] + dx))
            individual[i][1] = max(Y_MIN, min(Y_MAX, individual[i][1] + dy))
    return individual

def mutate_creep_dynamic(individual, mutation_rate, no_improve, patience, creep_inicial=0.05, creep_final=0.001):
    """
    Mutación por perturbación que reduce su radio de acción a medida que
    el algoritmo se estanca, permitiendo un ajuste micrométrico antes de rendirse.
    """
    # El progreso va de 0.0 (recién mejorado) a 1.0 (a punto de rendirse)
    progreso_estancamiento = no_improve / patience
    
    # Interpolación lineal del tamaño del salto
    creep_scale = creep_inicial - progreso_estancamiento * (creep_inicial - creep_final)
    
    rango_x = X_MAX - X_MIN
    rango_y = Y_MAX - Y_MIN
    
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            dx = random.uniform(-creep_scale * rango_x, creep_scale * rango_x)
            dy = random.uniform(-creep_scale * rango_y, creep_scale * rango_y)
            individual[i][0] = max(X_MIN, min(X_MAX, individual[i][0] + dx))
            individual[i][1] = max(Y_MIN, min(Y_MAX, individual[i][1] + dy))
            
    return individual

# ==============================================================================
# 6. COPIA PROFUNDA DE UN INDIVIDUO
# ==============================================================================
 
def deep_copy_individual(individual):
    """Devuelve una copia completamente independiente del individuo."""
    return [list(point) for point in individual]

def memetic_refinement(individual, bb_model, steps=5):
    """
    Aplica la búsqueda local binaria a TODOS los pares del individuo de golpe,
    usando operaciones vectorizadas de NumPy para no perder rendimiento.
    """
    puntos = np.array(individual)
    
    # 1. Separar en p1s y p2s
    p1s = puntos[0::2]
    p2s = puntos[1::2]
    
    # 2. Predicción inicial en BATCH (igual que en tu fitness)
    all_points = np.vstack([p1s, p2s])
    all_preds  = bb_model.predict(all_points)
    c1s = all_preds[:len(p1s)]
    c2s = all_preds[len(p1s):]
    
    # 3. Filtrar: Solo nos interesan los pares que cruzan la frontera (clases distintas)
    mask = (c1s != c2s)
    
    # Si ningún par cruza la frontera, devolvemos el individuo tal cual
    if not np.any(mask):
        return individual
        
    # Extraemos solo los puntos y clases que vamos a refinar
    v_p1s = p1s[mask]
    v_p2s = p2s[mask]
    v_c1s = c1s[mask]
    
    # 4. Bucle de búsqueda binaria vectorizado
    for _ in range(steps):
        # Calcular todos los midpoints de golpe
        midpoints = (v_p1s + v_p2s) / 2.0
        
        # Predecir todos los midpoints en una sola llamada a la BlackBox
        mid_preds = bb_model.predict(midpoints)
        
        # Máscara booleana: ¿El midpoint tiene la misma clase que p1?
        replace_p1_mask = (mid_preds == v_c1s)
        
        # Actualizamos p1 o p2 simultáneamente usando máscaras
        v_p1s[replace_p1_mask] = midpoints[replace_p1_mask]
        v_p2s[~replace_p1_mask] = midpoints[~replace_p1_mask]
        
    # 5. Volcar los puntos refinados de vuelta a sus arrays originales
    p1s[mask] = v_p1s
    p2s[mask] = v_p2s
    
    # 6. Reconstruir el individuo aplanado en formato lista de listas
    refined_puntos = np.empty((len(puntos), 2))
    refined_puntos[0::2] = p1s
    refined_puntos[1::2] = p2s
    
    return refined_puntos.tolist()


# ==============================================================================
# 5. BUCLE PRINCIPAL DEL ALGORITMO GENÉTICO
# ==============================================================================

def run_genetic_algorithm(pop_size=20, generations=50, mutation_rate=0.1,
                          mutation_method='random', crossover_method='random_size',
                          model='generational', adaptive_pc_pm='improvement',
                          elite_pct=0.1):
    print(f"Iniciando Algoritmo Genético... [mutación: {mutation_method} | cruce: {crossover_method} | modelo: {model} | adaptive: {adaptive_pc_pm}]")

    mutate    = mutate_random if mutation_method == 'random' else mutate_creep if mutation_method == 'creep' else mutate_creep_dynamic
    crossover = (uniform_crossover if crossover_method == 'uniform'
                else random_size_pool_crossover)

    PC_MAX, PC_MIN = 0.8, 0.2
    DELTA = 0.05  # paso de ajuste por generación en modo 'improvement'

    # Ambos modos parten de 0.5/0.5
    Pc = 0.5
    Pm = 0.5

    population = initialize_population(pop_size)
    fitnesses  = [evaluate_solution(ind) for ind in population]
    best_overall_individual = None
    best_overall_fitness = -float('inf')
    no_improve = 0

    for t in range(generations):
        print(f"\n--- Generación {t+1}/{generations} ---")

        # Calcular Pc y Pm según el modo elegido
        if adaptive_pc_pm == 'linear' and generations > 1:
            # Lineal: Pc 0.9→0.1, Pm 0.1→0.9 a lo largo de todas las generaciones
            Pc = PC_MAX - (PC_MAX - PC_MIN) * t / (generations - 1)
            Pm = 1 - Pc
        elif adaptive_pc_pm == 'improvement':
            # Basado en mejora: si hubo mejora → subir Pc (explotar), si no → subir Pm (explorar)
            # Se actualiza al final de cada generación según no_improve
            pass  # Pc y Pm se actualizan al final del bucle
        else:
            Pc = 1.0
            Pm = mutation_rate

        if model == 'generational':
            best_gen_idx      = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            best_gen_fitness  = fitnesses[best_gen_idx]
            best_gen_individual = population[best_gen_idx]

            print(f"Mejor Fitness de la generación: {best_gen_fitness:.4f}" +
                  (f"  [Pc={Pc:.2f} Pm={Pm:.2f}]" if adaptive_pc_pm else ""))

            if (best_gen_fitness - best_overall_fitness) > 0.0001:
                best_overall_fitness    = best_gen_fitness
                best_overall_individual = deep_copy_individual(best_gen_individual)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= PATIENCE:
                print(f"  Parada anticipada (sin mejora en {PATIENCE} generaciones consecutivas)")
                break

            n_elite = max(1, int(pop_size * elite_pct))
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:n_elite]
            new_population = [deep_copy_individual(population[i]) for i in elite_indices]
            while len(new_population) < pop_size:
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)

                # Pc: probabilidad de cruzar; si no, el hijo es copia del parent1
                if random.random() < Pc:
                    child = crossover(parent1, parent2)
                else:
                    child = list(parent1)

                # Pm: probabilidad de mutar el individuo (muta un gen aleatorio)
                if random.random() < Pm:
                    if mutation_method == 'creep_dynamic':
                        child = mutate(child, mutation_rate, no_improve, PATIENCE)
                    else:
                        child = mutate(child, mutation_rate)

                if random.random() < 0.1:                    # El hijo refina sus pares acercándolos a la frontera matemáticamente
                    child = memetic_refinement(child, bb, steps=4)

                new_population.append(child)

            population = new_population
            fitnesses  = [evaluate_solution(ind) for ind in population]

            # Actualizar Pc/Pm basado en si hubo mejora esta generación
            if adaptive_pc_pm == 'improvement':
                if no_improve == 0:
                    # Hubo mejora → explotar más: subir Pc, bajar Pm
                    Pc = min(PC_MAX, Pc + DELTA)
                else:
                    # Sin mejora → explorar más: bajar Pc, subir Pm
                    Pc = max(PC_MIN, Pc - DELTA)
                Pm = 1 - Pc

        else:  # steady-state
            # Cada "generación" = pop_size reemplazos → mismo nº de evaluaciones que generacional
            for _ in range(pop_size):
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)
                child   = crossover(parent1, parent2)
                child   = mutate(child, mutation_rate)
                child_fitness = evaluate_solution(child)

                # Reemplazar al peor (nunca al mejor — elitismo implícito)
                worst_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
                best_idx  = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
                if worst_idx != best_idx:
                    population[worst_idx] = child
                    fitnesses[worst_idx]  = child_fitness

            best_gen_idx     = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            best_gen_fitness = fitnesses[best_gen_idx]
            print(f"Mejor Fitness de la generación: {best_gen_fitness:.4f}")

            if best_gen_fitness > best_overall_fitness:
                best_overall_fitness    = best_gen_fitness
                best_overall_individual = list(population[best_gen_idx])

    print("\n=== FIN DE LA EJECUCIÓN ===")
    print(f"Mejor Fitness Global: {best_overall_fitness:.4f}")
    print(best_overall_individual)

    return best_overall_individual, best_overall_fitness


def guardar_estadisticas_csv(num_puntos, num_ejecuciones, media_fit, std_fit, media_time, std_time, mejor_fit):
    """Guarda las estadísticas completas de las ejecuciones en un archivo CSV."""
    nombre_archivo = "estadisticas_experimentos.csv"
    file_exists = os.path.isfile(nombre_archivo)
    
    with open(nombre_archivo, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Si el archivo es nuevo, escribimos la cabecera
        if not file_exists:
            writer.writerow([
                "Num_Puntos", 
                "Ejecuciones", 
                "Media_Fitness", 
                "Std_Fitness", 
                "Media_Tiempo(s)", 
                "Std_Tiempo(s)", 
                "Mejor_Fitness_Global"
            ])
        
        # Redondeamos los valores flotantes para que el CSV quede limpio
        writer.writerow([
            num_puntos, 
            num_ejecuciones, 
            round(media_fit, 4), 
            round(std_fit, 4), 
            round(media_time, 4), 
            round(std_time, 4), 
            round(mejor_fit, 4)
        ])
    print(f"\n[INFO] Estadísticas guardadas correctamente en '{nombre_archivo}'")


if __name__ == "__main__":
    main()