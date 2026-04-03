"""
Prueba definitiva: mutación aleatoria vs mutación creep
Ejecutar cuando se quiera comparar ambos operadores con configuración final.
"""
from main import run_genetic_algorithm, guardar_resultados_excel, PARAM_NAMES
import numpy as np

POP_SIZE      = 20
GENERATIONS   = 20
MUTATION_RATE = 0.1
N_REPS        = 5

results      = {'random': [], 'creep': []}
best_params  = {'random': None, 'creep': None}
best_fitness = {'random': -1,   'creep': -1}

for method in ['random', 'creep']:
    for rep in range(N_REPS):
        print(f"\n[{method}] Rep {rep+1}/{N_REPS}")
        params, fitness = run_genetic_algorithm(
            pop_size=POP_SIZE, generations=GENERATIONS,
            mutation_rate=MUTATION_RATE, mutation_method=method
        )
        results[method].append(fitness)
        if fitness > best_fitness[method]:
            best_fitness[method] = fitness
            best_params[method]  = params

print("\n" + "="*50)
print("COMPARACIÓN DE OPERADORES DE MUTACIÓN")
print("="*50)
for method, fitnesses in results.items():
    print(f"  {method:8s} → media: {np.mean(fitnesses):.4f}  "
          f"std: {np.std(fitnesses):.4f}  "
          f"mejor: {max(fitnesses):.4f}")
print("="*50)

for method in ['random', 'creep']:
    prefijo = f"ag_mutacion-{method}_pop{POP_SIZE}_gen{GENERATIONS}_reps{N_REPS}"
    guardar_resultados_excel(best_params[method], best_fitness[method], PARAM_NAMES, prefijo)
