import pandas as pd 
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from resultados import guardar_en_csv

def main():
    print("Cargando el dataset")

    #Aqui cargamos el dataset (winequality-red.csv)
    try: 
        data = pd.read_csv("../winequality-red.csv", sep = ';')
    except FileNotFoundError:
        print("Error al cargar el dataset")
        return 

    #Convertimos nuestro problema a una clasificación binaria como se indica en el enunciado (>= 6 => bueno (1) < 6 => malo (0))
    data["quality"] = (data["quality"] >= 6).astype(int)

    #Separar características y variable objetivo
    #En X guardamos el dataset eliminando la característica de quality, y en y la variable objetivo  quality
    X = data.drop("quality",axis = 1)
    y = data["quality"]

    modelo = RandomForestClassifier(random_state = 42)

    #Creamos la tabla para los parámetros
    parameter_grid = {
        'n_estimators': [10, 150, 300],
        'max_depth': [2, 15, 30],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 10, 20],
        'max_features': [0.1, 0.5, 1.0],
        'bootstrap':[True, False],
        'criterion':['gini', 'entropy'],
        'class_weight': [None, 'balanced'],
        'max_leaf_nodes': [10, 150,200],
        'min_impurity_decrease': [0, 0.05,0.1]
    }

    #Configuramos el modelo de grid search con la validación cruzada de 5 folds

    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid = parameter_grid,
        cv = 5,                         #5 folds
        scoring='accuracy',             #Usamos el accuracy como puntuación
        n_jobs = -1,
        verbose = 1
    )

    tiempo_inicio = time.time()

    #Entrenamos el modelo
    grid_search.fit(X,y);

    tiempo_final = time.time();

    tiempo_ejecucion = tiempo_final - tiempo_inicio

    #Ahora mostramos los resultados finales

    print("\n" + "="*50)
    print("RESULTADOS FINALES DE GRID SEARCH")
    print("="*50)
    print(f"Mejor Accuracy obtenida: {grid_search.best_score_:.4f}")
    print("\nMejores hiperparámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f" - {param}: {value}")
    print(f"\nTiempo de ejecución total: {tiempo_ejecucion:.2f} segundos")
    print("="*50)

    guardar_en_csv(
        mejor_accuracy=grid_search.best_score_,
        mejores_parametros=grid_search.best_params_,
        tiempo_ejecucion=tiempo_ejecucion
    )

if __name__ == "__main__":
    main()