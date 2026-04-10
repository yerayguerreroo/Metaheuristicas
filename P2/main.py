import sys
import os
import time

# Añadir subcarpetas al path para poder importar desde ellas
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'random_search'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'grid_search'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'algoritmo_genetico'))

from random_search.main import random_search
from grid_search.main import grid_search
from algoritmo_genetico.main import run_genetic_algorithm

def menu():
    while True:
        print("\n" + "="*50)
        print("PRÁCTICA 2 - OPTIMIZACIÓN DE HIPERPARÁMETROS")
        print("="*50)
        print("1. Random Search")
        print("2. Grid Search")
        print("3. Algoritmo Genético")
        print("4. Salir")

        opcion = input("\nOpción: ").strip()

        if opcion == '4':
            break

        elif opcion == '1':
            print("\n[Random Search]")
            inicio = time.time()
            mejor_solucion, mejor_score = random_search()
            tiempo = time.time() - inicio
            print("-"*50)
            print(f"Mejor Score:    {mejor_score:.4f}")
            print(f"Mejor Solución: {mejor_solucion}")
            print(f"Tiempo:         {tiempo:.2f}s")
            print("-"*50)

        elif opcion == '2':
            print("\n[Grid Search]")
            os.chdir(os.path.join(os.path.dirname(__file__), 'grid_search'))
            grid_search()
            os.chdir(os.path.dirname(__file__))

        elif opcion == '3':
            print("\n[Algoritmo Genético]")
            os.chdir(os.path.join(os.path.dirname(__file__), 'algoritmo_genetico'))
            mejor_individuo, mejor_fitness = run_genetic_algorithm(
                pop_size=50, generations=50, mutation_rate=0.1, adaptive_pc_pm=True
            )
            os.chdir(os.path.dirname(__file__))
            print("-"*50)
            print(f"Mejor Fitness:  {mejor_fitness:.4f}")
            print(f"Mejor Solución: {mejor_individuo}")
            print("-"*50)

        else:
            print("Opción no válida.")

if __name__ == "__main__":
    menu()
