import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from blackbox import BlackBoxModel
from graficar import plot_individual_and_boundary

# ==============================================================================
# CONFIGURACIÓN GENERAL Y CONSTANTES
# ==============================================================================

# Para A -3.5 3.5
# Para B -1.5 1.5
X_MIN, X_MAX = -1.5, 1.5
Y_MIN, Y_MAX = -1.5, 1.5

PUNTOS = 50
PATIENCE = 30  # generaciones consecutivas sin mejora para parar

# Importancia: Bien clasificados > Distancia > Dispersión

LAMBDA = 100.0   # Penalización por clase igual
DELTA  = 10.0    # Peso de la distancia entre puntos
MU     = 5.0    # Peso de la dispersión

bb = BlackBoxModel("blackbox_modelB.pkl")

# Acumulador de puntos medios de pares ya encontrados
found_midpoints = []
#Guarda todos los pares de puntos
all_pairs = []

def main():
    
    #print(evaluate_solution(initialize_individual(8)))
    mejor_individuo, _ = run_genetic_algorithm(pop_size=100, generations=300, mutation_method='creep', adaptive_pc_pm='improvement')

    print(f"\nMejor individuo: {mejor_individuo}")

    plot_individual_and_boundary(mejor_individuo, bb, X_MIN, X_MAX, Y_MIN, Y_MAX)


# ==============================================================================
# 1. CREACIÓN DE LA POBLACIÓN
# ==============================================================================

def generate_random_points(num_points):
    """Genera una lista con 'num_points' puntos aleatorios: [[x1, y1], [x2, y2], ...]"""
    puntos = []
    for _ in range(num_points):
        x = random.uniform(X_MIN, X_MAX)
        y = random.uniform(Y_MIN, Y_MAX)
        puntos.append([x, y])
    return puntos

def initialize_individual(num_points):
    """Un individuo es simplemente un conjunto de X puntos"""
    return generate_random_points(num_points)

def initialize_population(pop_size, num_points):
    """Crea la población inicial pasándole la cantidad de puntos por individuo"""
    return [initialize_individual(num_points) for _ in range(pop_size)]

# ==============================================================================
# 2. EVALUACIÓN Y FITNESS
# ==============================================================================

def evaluate_solution(params):
    """
    Evalúa un individuo que es un vector 2D de N puntos.
    params: [[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...]
    """
    # Convertimos a un array de NumPy (si no lo es ya) para facilitar las matemáticas
    puntos = np.array(params)
    total_fitness = 0.0
    
    # Copiamos los puntos medios globales para la repulsión local
    local_midpoints = list(found_midpoints) 

    # Iteramos la matriz saltando de 2 en 2 para agarrar pares (p1 y p2)
    for i in range(0, len(puntos), 2):
        p1 = puntos[i]
        p2 = puntos[i+1]

        # 1. Distancia euclídea
        dist = np.linalg.norm(p1 - p2)

        # 2. Penalización de clase
        f1 = bb.predict(p1)
        f2 = bb.predict(p2)
        P_clase = 0.0 if f1 != f2 else LAMBDA

        # 3. Dispersión del punto medio respecto a los ya encontrados
        m = (p1 + p2) / 2.0
        if len(local_midpoints) == 0:
            disp = 0.0  
        else:
            disp = min(np.linalg.norm(m - mj) for mj in local_midpoints)

        # Sumamos el fitness de este par al total del individuo
        total_fitness += (-DELTA * dist - P_clase + MU * disp)
        
        # Añadimos el punto medio al registro local
        local_midpoints.append(m)

    # El AG maximiza → devolvemos negativo
    return total_fitness

# ==============================================================================
# 3. OPERADORES DE SELECCIÓN
# ==============================================================================

def tournament_selection(population, fitnesses, k=3):
    """Selecciona un padre mediante torneo de tamaño k"""
    selected_indices = random.sample(range(len(population)), k)
    best_index = max(selected_indices, key=lambda idx: fitnesses[idx])
    return population[best_index]

# ==============================================================================
# 4. OPERADORES DE CRUCE
# ==============================================================================

def uniform_crossover(parent1, parent2):
    """Cruza dos padres par a par (de 2 en 2 genes) con 50% de probabilidad"""
    child = []
    for i in range(0, len(parent1), 2):
        if random.random() < 0.5:
            # Hereda el par completo del padre 1
            child.extend([list(parent1[i]), list(parent1[i+1])])
        else:
            # Hereda el par completo del padre 2
            child.extend([list(parent2[i]), list(parent2[i+1])])
    return child

def two_point_crossover(parent1, parent2):
    """
    Cruce en dos puntos: se eligen dos cortes aleatorios c1 y c2.
    El hijo toma [0..c1] de parent1, [c1+1..c2] de parent2, [c2+1..fin] de parent1.
    """
    n = len(parent1)
    c1, c2 = sorted(random.sample(range(n), 2))
    # FIX: copia profunda de cada punto para no compartir referencias
    child = (
        [list(p) for p in parent1[:c1 + 1]] +
        [list(p) for p in parent2[c1 + 1:c2 + 1]] +
        [list(p) for p in parent1[c2 + 1:]]
    )
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

def mutate_creep(individual, mutation_rate, creep_scale=0.1):
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

# --- 5. BUCLE PRINCIPAL DEL ALGORITMO GENÉTICO ---

#Inicio
#	t=0;
#	Inicializar P(t);
#	Evaluar P(t);
#	Mientras NO SE CUMPLA CRITERIO DE PARADA hacer
#		Seleccionar P(t+1) desde P(t)
#		Cruzar P(t+1)
#		Mutar P(t+1)
#		Evaluar P(t+1)
#
#		Reemplazar P(t) por P(t+1)
#
#		t=t+1
#	fin mientras
#fin

# ==============================================================================
# 6. COPIA PROFUNDA DE UN INDIVIDUO
# ==============================================================================
 
def deep_copy_individual(individual):
    """Devuelve una copia completamente independiente del individuo."""
    return [list(point) for point in individual]


# ==============================================================================
# 5. BUCLE PRINCIPAL DEL ALGORITMO GENÉTICO
# ==============================================================================

def run_genetic_algorithm(pop_size=20, generations=50, mutation_rate=0.1,
                          mutation_method='random', crossover_method='uniform',
                          model='generational', adaptive_pc_pm='improvement',
                          elite_pct=0.1):
    print(f"Iniciando Algoritmo Genético... [mutación: {mutation_method} | cruce: {crossover_method} | modelo: {model} | adaptive: {adaptive_pc_pm}]")

    mutate    = mutate_random if mutation_method == 'random' else mutate_creep
    crossover = uniform_crossover if crossover_method == 'uniform' else two_point_crossover

    PC_MAX, PC_MIN = 0.9, 0.1
    DELTA = 0.05  # paso de ajuste por generación en modo 'improvement'

    # Ambos modos parten de 0.5/0.5
    Pc = 0.5
    Pm = 0.5

    population = initialize_population(pop_size, PUNTOS)
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

            if best_gen_fitness > best_overall_fitness:
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
                    child = mutate(child, mutation_rate)

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


# --- COMPARACIÓN: ELITISMO 1 INDIVIDUO vs 10% ---

if __name__ == "__main__":
    main()

    # POP_SIZE      = 50
    # GENERATIONS   = 50
    # MUTATION_RATE = 0.1
    # N_REPS        = 3

    # configs      = {'elite_1': 1/50, 'elite_10pct': 0.1}
    # results      = {k: [] for k in configs}
    # best_params  = {k: None for k in configs}
    # best_fitness = {k: -1 for k in configs}

    # for label, pct in configs.items():
    #     for rep in range(N_REPS):
    #         print(f"\n[{label}] Rep {rep+1}/{N_REPS}")
    #         params, fitness = run_genetic_algorithm(
    #             pop_size=POP_SIZE, generations=GENERATIONS,
    #             mutation_rate=MUTATION_RATE, elite_pct=pct
    #         )
    #         results[label].append(fitness)
    #         if fitness > best_fitness[label]:
    #             best_fitness[label] = fitness
    #             best_params[label]  = params

    # print("\n" + "="*50)
    # print("COMPARACIÓN ELITISMO 1 INDIVIDUO vs 10%")
    # print("="*50)
    # for label, fitnesses in results.items():
    #     print(f"  {label:12s} → media: {np.mean(fitnesses):.4f}  "
    #           f"std: {np.std(fitnesses):.4f}  "
    #           f"mejor: {max(fitnesses):.4f}")
    # print("="*50)

    # for label in configs:
    #     prefijo = f"ag_elitismo-{label}_pop{POP_SIZE}_gen{GENERATIONS}_reps{N_REPS}"
    #     guardar_resultados_excel(best_params[label], best_fitness[label], PARAM_NAMES, prefijo)
