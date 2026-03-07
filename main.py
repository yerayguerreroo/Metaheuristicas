from random_search.main import random_search
from hill_climbing.main import hill_climbing, hill_climbing_maxima_pendiente
from simulated_annealing.main import simulated_annealing
from simulated_annealing.plotting import plot_series_with_piecewise_lines
from reports import  reporte_SA, reporte_HC_Simple, reporte_HC_Maxima_Pendiente, reporte_RS
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
                print("e) Ejecutar todos (Comparativa)")

                eleccion = input("Opción: ").lower()
                
                if eleccion == 'a':
                    reporte_RS(
                        random_search,
                        repeticiones=50,
                        file=serie,
                        serie=datos
                    )
                    #random_search(datos, serie['k'])

                elif eleccion == 'b':
                    reporte_HC_Simple(
                        hill_climbing,
                        repeticiones=50,
                        file=serie,
                        serie=datos
                    )

                    #mejores_cortes, error_final = hill_climbing(datos ,serie['file'], serie['k'])
                    #print(f"RESULTADO FINAL {serie['file']}:")
                    #print(f"Cortes: {mejores_cortes}")
                    #print(f"RMSE Promedio: {error_final:.6f}\n")
                
                elif eleccion == 'c':
                    reporte_HC_Maxima_Pendiente(
                        hill_climbing_maxima_pendiente,
                        repeticiones=50,
                        file=serie,
                        serie=datos
                    )
                    
                    #mejores_cortes, error_final = hill_climbing_maxima_pendiente(datos ,serie['file'], serie['k'])
                    #print(f"RESULTADO FINAL {serie['file']}:")
                    #print(f"Cortes: {mejores_cortes}")
                    #print(f"RMSE Promedio: {error_final:.6f}\n")

                elif eleccion == 'd':

                    T0 = 0.2
                    alpha = 0.975
                    Tf = 0.001

                    # simulated_annealing(T0, alpha, L, Tf, serie,datos)
                    reporte_SA(
                        simulated_annealing,
                        repeticiones=50,
                        T0=T0,
                        alpha=alpha,
                        Tf=Tf,
                        file=serie,
                        serie=datos
                    )
                    # gráfica debugging

                elif eleccion == 'e':
                    for nombre in ["Búsqueda Aleatoria", "Hill Climbing", "Simulated Annealing"]:
                        ejecutar_experimento(None, datos, serie['k'], nombre)
        
        else:
            print("Opción No Válida")

if __name__ == "__main__":
    menu()

# TS1.txt (k = 9)

    # T0 = 0.1
    # alpha = .95
    # L = serie['k'] * 4
    # Tf = 0.001
    # Ejecuciones: 1000
    # Media RMSE: 0.829951
    # Desviación típica RMSE: 0.344968
    # Mejora Media (G): 60.41243191379893%
    # Mejor RMSE: 0.418282
    # Peor RMSE: 2.270234
    # Media tiempo: 1.677s
    # Desviación tiempo: 0.373s

# Muy bueno: ≤ 0.45 (si estás por aquí, vas fino)
# Bueno: 0.45 – 0.65
# Aceptable: 0.65 – 1.00
# Malo: > 1.00

# TS2.txt (k = 10)

    # T0 = 0.1
    # alpha = .95
    # L = serie['k'] * 4
    # Tf = 0.001
    # Ejecuciones: 500
    # Media RMSE: 1.298046
    # Desviación típica RMSE: 0.161268
    # Mejora Media (G): 28.512178317534563%
    # Mejor RMSE: 1.066377
    # Peor RMSE: 1.804996
    # Media tiempo: 1.880s
    # Desviación tiempo: 0.073s

# Muy bueno: ≤ 0.90
# Bueno: 0.90 – 1.20
# Aceptable: 1.20 – 1.60
# Malo: > 1.60

# TS3.txt (k = 20)

# Muy bueno: ≤ 0.85
# Bueno: 0.85 – 0.95
# Aceptable: 0.95 – 1.05
# Malo: > 1.05

# TS4.txt (k = 50)

# Muy bueno: ≤ 1.30
# Bueno: 1.30 – 1.55
# Aceptable: 1.55 – 1.80
# Malo: > 1.80