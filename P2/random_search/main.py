import random

def random_search():
    # 1. n_estimators: entero 10 – 300
    n_estimators = random.randint(10, 300)
    
    # 2. max_depth: entero 2 – 30
    max_depth = random.randint(2, 30)
    
    # 3. min_samples_split: entero 2 – 20
    min_samples_split = random.randint(2, 20)
    
    # 4. min_samples_leaf: entero 1 – 20
    min_samples_leaf = random.randint(1, 20)
    
    # 5. max_features: real 0.1 – 1.0
    max_features = random.uniform(0.1, 1.0)
    
    # 6. bootstrap: binario 0 / 1
    bootstrap = random.randint(0, 1)
    
    # 7. criterion: categórico 0 = gini, 1 = entropy
    criterion = random.randint(0, 1)
    
    # 8. class_weight: binario 0 = None, 1 = balanced
    class_weight = random.randint(0, 1)
    
    # 9. max_leaf_nodes: entero 10 – 200
    max_leaf_nodes = random.randint(10, 200)
    
    # 10. min_impurity_decrease: real 0 – 0.1
    min_impurity_decrease = random.uniform(0.0, 0.1)
    
    # Agrupamos todos los valores en un vector
    resultados = [
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        bootstrap,
        criterion,
        class_weight,
        max_leaf_nodes,
        min_impurity_decrease
    ]
    
    return resultados