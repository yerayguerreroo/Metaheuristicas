from random_search.main import random_search
from hill_climbing.main import hill_climbing, hill_climbing_maxima_pendiente
from simulated_annealing.main import simulated_annealing
from reports import  reporte_SA, reporte_HC_Simple, reporte_HC_Maxima_Pendiente, reporte_RS
from random_search.plotting import plot_series_with_piecewise_lines
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
                print("b) Hill Climbing Simple")
                print("c) Hill Climbing con Máxima Pendiente")
                print("d) Simulated Annealing")

                eleccion = input("Opción: ").lower()
                
                if eleccion == 'a':
                    reporte_RS(
                        random_search,
                        repeticiones=50,
                        file=serie,
                        serie=datos
                    )

                elif eleccion == 'b':
                    reporte_HC_Simple(
                        hill_climbing,
                        repeticiones=50,
                        file=serie,
                        serie=datos
                    )
                
                elif eleccion == 'c':
                    reporte_HC_Maxima_Pendiente(
                        hill_climbing_maxima_pendiente,
                        repeticiones=50,
                        file=serie,
                        serie=datos
                    )

                elif eleccion == 'd':
                    T0 = 0.2
                    alpha = 0.975
                    Tf = 0.001

                    reporte_SA(
                        simulated_annealing,
                        repeticiones=50,
                        T0=T0,
                        alpha=alpha,
                        Tf=Tf,
                        file=serie,
                        serie=datos
                    )
        
        else:
            print("Opción No Válida")

if __name__ == "__main__":
    menu()