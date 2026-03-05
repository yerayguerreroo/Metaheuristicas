#!/bin/bash

# Bucle externo: va del 1 al 4
for numero in {1..4}; do

  # Bucle interno: va de la 'a' a la 'd'
  for letra in {a..d}; do

    echo "-----------------------------------"
    echo "Ejecutando con: Número $numero y Letra $letra"

    # Aquí es donde le inyectas los datos a tu programa Python
    # El \n simula que pulsas Enter entre el número y la letra
    echo -e "$numero\n$letra" | python3 main.py

  done

done
