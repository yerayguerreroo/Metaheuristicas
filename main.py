from random_search.main import random_search
from hill_climbing.main import hill_climbing
from simulated_annealing.main import simulated_annealing
import numpy as np

#Función para cargar datos
def cargar_datos(filename):
    try:
        with open(filename, 'r') as f:
            contenido = f.read().replace('[', '').replace(']', '').split()
            return np.array([float(x) for x in contenido])
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filename}")
        return None

def menu():
    configuracion = {
        '1' :{'file' : 'TS1.txt', 'k' : 9},
        '2': {'file': 'TS2.txt', 'k': 10},
        '3': {'file': 'TS3.txt', 'k': 20},
        '4': {'file': 'TS4.txt', 'k': 50}
    }

    while True:
        print("1. Ejecutar TS1 (k=9)")
        print("2. Ejecutar TS2 (k=10)")
        print("3. Ejecutar TS3 (k=20)")
        print("4. Ejecutar TS4 (k=50)")
        print("5. Salir")

        opcion = input("Seleccione una serie temporal: ")

        if opcion == '5':
            break

        if opcion in configuracion:
            serie = configuracion[opcion]
            datos = cargar_datos(serie['file'])

            if datos is not None:
                print(f"\nSerie cargada: {serie['file']} con {len(datos)} puntos.")
                print("Seleccione Algoritmo:")
                print("a) Búsqueda Aleatoria")
                print("b) Hill Climbing")
                print("c) Simulated Annealing")
                print("d) Ejecutar todos (Comparativa)")

                eleccion = input("Opción: ").lower()
                
                if eleccion == 'a':
                    random_search(datos, serie['k'])
                elif eleccion == 'b':
                    mejores_cortes, error_final = hill_climbing(datos ,serie['file'], serie['k'])
                    print(f"RESULTADO FINAL {serie['file']}:")
                    print(f"Cortes: {mejores_cortes}")
                    print(f"RMSE Promedio: {error_final:.6f}\n")
                elif eleccion == 'c':
                    simulated_annealing(T0, alpha, L, Tf, datos)
                elif eleccion == 'd':
                    for nombre in ["Búsqueda Aleatoria", "Hill Climbing", "Simulated Annealing"]:
                        ejecutar_experimento(None, datos, serie['k'], nombre)
        
        else:
            print("Opción No Válida")

if __name__ == "__main__":
    menu()