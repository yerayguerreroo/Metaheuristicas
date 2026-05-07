#!/bin/bash

echo "Iniciando bateria de pruebas..."

# Lista de puntos a evaluar
PUNTOS_LIST=(A A A A A B B B B B)

for p in "${PUNTOS_LIST[@]}"
do
    echo ""
    echo "=================================================="
    echo "Ejecutando el algoritmo con $p puntos..."
    echo "=================================================="
    echo "10" | python3 main_sin_numero_puntos.py $p
done

echo ""
echo "=================================================="
echo "¡Todas las ejecuciones han terminado!"