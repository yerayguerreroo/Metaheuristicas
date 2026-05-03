#!/bin/bash

echo "Iniciando bateria de pruebas..."

# Lista de puntos a evaluar
PUNTOS_LIST=(10 20 30 40 50)

for p in "${PUNTOS_LIST[@]}"
do
    echo ""
    echo "=================================================="
    echo "Ejecutando el algoritmo con $p puntos..."
    echo "=================================================="
    python3 main.py $p
done

echo ""
echo "=================================================="
echo "¡Todas las ejecuciones han terminado!"