#!/bin/bash

echo "Iniciando batería de pruebas exhaustiva..."

# Definición de las variables para las pruebas
MODELOS=("A" "B")
GENERACIONES=(100 150)
POBLACIONES=(150 200 250 300)

# Contador para seguimiento
TOTAL_TESTS=$(( ${#MODELOS[@]} * ${#GENERACIONES[@]} * ${#POBLACIONES[@]} ))
CURRENT_TEST=1

for m in "${MODELOS[@]}"
do
    for g in "${GENERACIONES[@]}"
    do
        for p in "${POBLACIONES[@]}"
        do
            echo ""
            echo "--------------------------------------------------"
            echo "Prueba $CURRENT_TEST de $TOTAL_TESTS"
            echo "Configuración: Modelo $m | Gen: $g | Pob: $p"
            echo "--------------------------------------------------"
            
            # Ejecutamos el script pasando los tres argumentos
            echo 30 | python3 main_sin_numero_puntos.py "$m" "$g" "$p"
            
            CURRENT_TEST=$((CURRENT_TEST + 1))
        done
    done
done

echo ""
echo "=================================================="
echo "¡Batería de pruebas completada con éxito!"
echo "Total de combinaciones ejecutadas: $TOTAL_TESTS"
echo "=================================================="