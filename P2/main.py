from random_search.main import random_search
from random_forest import evaluate_solution
from algoritmo_genetico.main import run_genetic_algorithm
from grid_search.main import grid_search
import numpy as np

def menu():

    while True:
        print("Seleccione Algoritmo:")
        print("1. Random Search")
        print("2. Grid Search")
        print("3. Algoritmo Genético")
        print("4. Salir")

        opcion = input("Opción: ")

        if opcion == '4':
            break

        elif opcion == '1':
            resultados_rs, score = random_search()
            print("-"*50)
            print(f"Mejor resultados: {resultados_rs}, Score: {score}")
            print("-"*50)

        elif opcion == '2':
            grid_search()

        elif opcion == '3':

            POP_SIZE      = 50
            GENERATIONS   = 50
            MUTATION_RATE = 0.1

            resultados_ga = run_genetic_algorithm(pop_size=POP_SIZE, generations=GENERATIONS, mutation_rate=MUTATION_RATE, adaptive_pc_pm=True)
            print(f"Resultados: {resultados_ga}")
            resultados = evaluate_solution(resultados_ga)
            print(f"Resultados: {resultados}")
        
        else:
            print("Opción No Válida")

if __name__ == "__main__":
    menu()