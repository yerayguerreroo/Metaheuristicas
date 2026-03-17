from random_search.main import random_search
from random_forest import evaluate_solution
from algoritmo_genetico.main import run_genetic_algorithm
# from hill_climbing.main import hill_climbing, hill_climbing_maxima_pendiente
# from simulated_annealing.main import simulated_annealing
# from reports import  reporte_SA, reporte_HC_Simple, reporte_HC_Maxima_Pendiente, reporte_RS
# from random_search.plotting import plot_series_with_piecewise_lines
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
            resultados_rs = random_search()
            print(f"Resultados: {resultados_rs}")
            resultados = evaluate_solution(resultados_rs)
            print(f"Resultados: {resultados}")

        elif opcion == '2':
            print("GS")

        elif opcion == '3':
            resultados_ga = run_genetic_algorithm()
            print(f"Resultados: {resultados_ga}")
            resultados = evaluate_solution(resultados_ga)
            print(f"Resultados: {resultados}")
        
        else:
            print("Opción No Válida")

if __name__ == "__main__":
    menu()