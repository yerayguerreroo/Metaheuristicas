import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# --- 1. PREPARACIÓN DE DATOS (Basado en el enunciado) ---
# Cargar dataset (Asegúrate de tener el CSV en la misma carpeta o en tu Colab)
try:
    data = pd.read_csv("winequality-red.csv")
    # Convertir problema a clasificación binaria (>=6 es 1, <6 es 0)
    data["quality"] = (data["quality"] >= 6).astype(int)
    X = data.drop("quality", axis=1)
    y = data["quality"]
except FileNotFoundError:
    print("Por favor, sube el archivo 'winequality-red.csv' para poder ejecutar el modelo.")
    X, y = None, None

# --- 2. DEFINICIÓN DE RANGOS DE GENES ---
# Tupla: (min, max, tipo). Esto facilita la inicialización y la mutación.
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
    if X is None: return 0.0 # Control de seguridad
    
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
        random_state=42 # Para reproducibilidad
    )
    # 5 folds, scoring=accuracy
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

def mutate(individual, mutation_rate):
    """Muta los genes de un individuo con una probabilidad dada"""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = generate_random_gene(i) # Genera un nuevo valor válido
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

def run_genetic_algorithm(pop_size=20, generations=10, mutation_rate=0.1):
    print("Iniciando Algoritmo Genético...")
    
    # t = 0; Inicializar P(t)
    population = initialize_population(pop_size)
    best_overall_individual = None
    best_overall_fitness = -1

    # Bucle de generaciones
    for t in range(generations):
        print(f"\n--- Generación {t+1}/{generations} ---")
        
        # Evaluar P(t)
        fitnesses = [evaluate_solution(ind) for ind in population]
        
        # Guardar el mejor de la generación (Elitismo)
        best_gen_fitness = max(fitnesses)
        best_gen_idx = fitnesses.index(best_gen_fitness)
        best_gen_individual = population[best_gen_idx]
        
        print(f"Mejor Fitness de la generación: {best_gen_fitness:.4f}")
        
        if best_gen_fitness > best_overall_fitness:
            best_overall_fitness = best_gen_fitness
            best_overall_individual = list(best_gen_individual)

        # Nueva población P(t+1)
        new_population = []
        
        # Elitismo: pasamos el mejor individuo directamente
        new_population.append(list(best_gen_individual))
        
        # Generar el resto de la nueva población
        while len(new_population) < pop_size:
            # Seleccionar
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # Cruzar
            child = uniform_crossover(parent1, parent2)
            
            # Mutar
            child = mutate(child, mutation_rate)
            
            new_population.append(child)
            
        # Reemplazar P(t) por P(t+1)
        population = new_population
        
    print("\n=== FIN DE LA EJECUCIÓN ===")
    print(f"Mejor Fitness Global: {best_overall_fitness:.4f}")
    print("Mejores Hiperparámetros encontrados:")
    print(best_overall_individual)
    
    return best_overall_individual, best_overall_fitness

