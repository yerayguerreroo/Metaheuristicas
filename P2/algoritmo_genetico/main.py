import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from guardardatosexcel import guardar_resultados_excel

# --- 1. PREPARACIÓN DE DATOS (Basado en el enunciado) ---
try:
    data = pd.read_csv("winequality-red.csv", sep=";")
    # sep=";" porque el CSV usa punto y coma como separador (formato europeo)
    data["quality"] = (data["quality"] >= 6).astype(int)
    X = data.drop("quality", axis=1)
    y = data["quality"]
except FileNotFoundError:
    print("Por favor, sube el archivo 'winequality-red.csv' para poder ejecutar el modelo.")
    X, y = None, None

# --- 2. DEFINICIÓN DE RANGOS DE GENES ---
PARAM_NAMES = [
    "n_estimators", "max_depth", "min_samples_split",
    "min_samples_leaf", "max_features", "bootstrap",
    "criterion", "class_weight", "max_leaf_nodes",
    "min_impurity_decrease"
]
PARAM_BOUNDS = [
    (10, 300, int),       # 0: n_estimators
    (2, 30, int),         # 1: max_depth
    (2, 20, int),         # 2: min_samples_split
    (1, 20, int),         # 3: min_samples_leaf
    (0.1, 1.0, float),    # 4: max_features
    (0, 1, int),          # 5: bootstrap (0/1)
    (0, 1, int),          # 6: criterion (0=gini, 1=entropy)
    (0, 1, int),          # 7: class_weight (0=None, 1=balanced)
    (10, 200, int),       # 8: max_leaf_nodes
    (0.0, 0.1, float)     # 9: min_impurity_decrease
]

# --- 3. EVALUACIÓN (Función del enunciado) ---
def evaluate_solution(params):
    if X is None: return 0.0

    model = RandomForestClassifier(
        n_estimators=int(params[0]),
        max_depth=int(params[1]),
        min_samples_split=int(params[2]),
        min_samples_leaf=int(params[3]),
        max_features=float(params[4]),
        bootstrap=bool(params[5]),
        criterion="gini" if params[6] == 0 else "entropy",
        class_weight=None if params[7] == 0 else "balanced",
        max_leaf_nodes=int(params[8]),
        min_impurity_decrease=float(params[9]),
        random_state=42
    )
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    return scores.mean()

# --- 4. OPERADORES DEL ALGORITMO GENÉTICO ---

def generate_random_gene(index):
    """Genera un valor aleatorio válido para un gen específico según sus límites."""
    min_val, max_val, gene_type = PARAM_BOUNDS[index]
    if gene_type == int:
        return random.randint(min_val, max_val)
    else:
        return random.uniform(min_val, max_val)

def initialize_population(pop_size):
    """Crea la población inicial P(0)"""
    population = []
    for _ in range(pop_size):
        individual = [generate_random_gene(i) for i in range(len(PARAM_BOUNDS))]
        population.append(individual)
    return population

def tournament_selection(population, fitnesses, k=3):
    """Selecciona un padre mediante torneo de tamaño k"""
    selected_indices = random.sample(range(len(population)), k)
    best_index = max(selected_indices, key=lambda idx: fitnesses[idx])
    return population[best_index]

def uniform_crossover(parent1, parent2):
    """Cruza dos padres gen a gen con 50% de probabilidad"""
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

def two_point_crossover(parent1, parent2):
    """
    Cruce en dos puntos: se eligen dos cortes aleatorios c1 y c2.
    El hijo toma [0..c1] de parent1, [c1+1..c2] de parent2, [c2+1..fin] de parent1.
    Preserva bloques contiguos de genes de cada padre. Menos mezcla que el uniforme
    pero más que el cruce en un punto.
    """
    n = len(parent1)
    c1, c2 = sorted(random.sample(range(n), 2))
    return parent1[:c1+1] + parent2[c1+1:c2+1] + parent1[c2+1:]

def mutate_random(individual, mutation_rate):
    """Mutación aleatoria: resetea el gen a un valor completamente nuevo dentro de sus límites."""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = generate_random_gene(i)
    return individual

def mutate_creep(individual, mutation_rate):
    """
    Mutación por perturbación (creep): en lugar de resetear el gen completamente,
    le suma un pequeño delta proporcional al rango del parámetro (±10%).
    Útil en generaciones avanzadas: hace ajustes finos cerca de buenas soluciones
    en lugar de destruirlas con un salto aleatorio grande.
    Los genes binarios (bootstrap, criterion, class_weight) se tratan igual que
    en la mutación aleatoria — no tiene sentido perturbar un valor 0/1.
    """
    BINARY_GENES = {5, 6, 7}  # índices de genes binarios

    for i in range(len(individual)):
        if random.random() < mutation_rate:
            min_val, max_val, gene_type = PARAM_BOUNDS[i]

            if i in BINARY_GENES:
                # Para binarios: flip igual que antes
                individual[i] = 1 - individual[i]
            else:
                delta = random.uniform(-0.1 * (max_val - min_val),
                                        0.1 * (max_val - min_val))
                new_val = individual[i] + delta
                # Clamp: mantener el valor dentro de los límites del parámetro
                new_val = max(min_val, min(max_val, new_val))
                individual[i] = int(round(new_val)) if gene_type == int else new_val

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

    population = initialize_population(pop_size)
    fitnesses  = [evaluate_solution(ind) for ind in population]
    best_overall_individual = None
    best_overall_fitness = -1
    no_improve = 0
    PATIENCE   = 10  # generaciones consecutivas sin mejora para parar

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
                best_overall_individual = list(best_gen_individual)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= PATIENCE:
                print(f"  Parada anticipada (sin mejora en {PATIENCE} generaciones consecutivas)")
                break

            n_elite = max(1, int(pop_size * elite_pct))
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:n_elite]
            new_population = [list(population[i]) for i in elite_indices]
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
                    gene_idx = random.randint(0, len(child) - 1)
                    child[gene_idx] = generate_random_gene(gene_idx)

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

    POP_SIZE      = 50
    GENERATIONS   = 50
    MUTATION_RATE = 0.1
    N_REPS        = 3

    configs      = {'elite_1': 1/50, 'elite_10pct': 0.1}
    results      = {k: [] for k in configs}
    best_params  = {k: None for k in configs}
    best_fitness = {k: -1 for k in configs}

    for label, pct in configs.items():
        for rep in range(N_REPS):
            print(f"\n[{label}] Rep {rep+1}/{N_REPS}")
            params, fitness = run_genetic_algorithm(
                pop_size=POP_SIZE, generations=GENERATIONS,
                mutation_rate=MUTATION_RATE, elite_pct=pct
            )
            results[label].append(fitness)
            if fitness > best_fitness[label]:
                best_fitness[label] = fitness
                best_params[label]  = params

    print("\n" + "="*50)
    print("COMPARACIÓN ELITISMO 1 INDIVIDUO vs 10%")
    print("="*50)
    for label, fitnesses in results.items():
        print(f"  {label:12s} → media: {np.mean(fitnesses):.4f}  "
              f"std: {np.std(fitnesses):.4f}  "
              f"mejor: {max(fitnesses):.4f}")
    print("="*50)

    for label in configs:
        prefijo = f"ag_elitismo-{label}_pop{POP_SIZE}_gen{GENERATIONS}_reps{N_REPS}"
        guardar_resultados_excel(best_params[label], best_fitness[label], PARAM_NAMES, prefijo)
