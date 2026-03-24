import pandas as pd
import os

import pandas as pd
import os
from datetime import datetime

def guardar_resultados_excel(best_params, best_fitness, param_names, prefijo_archivo="resultados_ag"):
    """
    Guarda los mejores parámetros en la carpeta 'excel' añadiendo la fecha y hora al nombre para no sobreescribir.
    """
    carpeta = "excel"
    os.makedirs(carpeta, exist_ok=True)
    
    # 1. Generamos un texto con la fecha y hora actual (Ej: 20260324_125721)
    ahora = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 2. Creamos el nombre del archivo uniendo el prefijo, la hora y la extensión
    nombre_archivo = f"{prefijo_archivo}_{ahora}.xlsx"
    ruta_completa = os.path.join(carpeta, nombre_archivo)
    
    # 3. Emparejamos datos y añadimos el fitness
    datos = {nombre: [valor] for nombre, valor in zip(param_names, best_params)}
    datos["Mejor_Fitness_Accuracy"] = [best_fitness]
    
    # 4. Convertimos a DataFrame y guardamos
    df = pd.DataFrame(datos)
    
    try:
        df.to_excel(ruta_completa, index=False)
        print(f"\n¡Éxito! Resultados guardados sin sobreescribir en '{ruta_completa}'")
    except Exception as e:
        print(f"\nHubo un error al guardar el Excel: {e}")