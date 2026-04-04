# Memoria de Fine-Tuning del Algoritmo Genético

## Configuración base del AG

| Parámetro | Valor pruebas | Valor final |
|---|---|---|
| pop_size | 10-20 | 50 |
| generations | 10-20 | 50 |
| mutation_rate | 0.1 | 0.1 |
| selección | torneo k=3 | torneo k=3 |
| cruce | uniforme | uniforme |
| elitismo | sí | sí |

Cada evaluación = 5-fold CV sobre RandomForest ≈ 0.23s por individuo.

---

## 1. Inicialización de población

### Planteamiento
La población inicial puede generarse de forma aleatoria o forzando diversidad mínima entre individuos.

### Métodos comparados

**Aleatoria:** cada individuo se genera completamente al azar dentro de los rangos de cada gen.

**Secuencial (distance-based):** el primer individuo es aleatorio. Cada nuevo candidato solo se acepta si su distancia euclídea normalizada a todos los individuos ya aceptados es >= D. Si tras `max_attempts` intentos no se encuentra candidato válido, D se reduce un 10%.

- Distancia calculada en espacio normalizado [0,1]^10 para evitar que genes con rangos grandes dominen sobre genes con rangos pequeños.
- D inicial = 0.5 (distancia máxima posible: sqrt(10) ≈ 3.16).

### Conclusión
**Se mantiene la inicialización aleatoria.** La secuencial añade coste computacional sin beneficio observable. Con solo 10 genes el espacio es manejable con aleatoriedad simple.

---

## 2. Operador de mutación

### Planteamiento
La mutación actual resetea el gen completamente. Se planteó una mutación por perturbación (creep) que hace ajustes pequeños en lugar de saltos.

### Métodos comparados

**Aleatoria (random):** resetea el gen a un valor aleatorio dentro de sus límites.

**Creep:** suma un delta de ±10% del rango del gen. Los genes binarios (bootstrap, criterion, class_weight — índices 5, 6, 7) hacen flip en lugar de perturbación.

### Resultados

Experimento: `pop_size=10, generations=10, N_REPS=5`

| Mutación | Media | Std | Mejor |
|---|---|---|---|
| random | **0.7442** | **0.0006** | **0.7448** |
| creep | 0.7398 | 0.0019 | 0.7417 |

### Conclusión
**Se mantiene la mutación aleatoria.** Random es mejor en media y 3x más estable. Con 10 genes y rangos amplios, los saltos grandes cubren mejor el espacio. Creep sería útil en problemas con muchos más genes donde un salto grande destruiría soluciones parcialmente buenas.

---

## 3. Selección de padres

### Planteamiento
Torneo k=3 mantenido. Ranking se valoró como alternativa.

### Conclusión
**Torneo k=3 mantenido.** Ranking descartado por bajo impacto esperado en este problema.

---

## 4. Operadores de cruce

### Planteamiento
El cruce uniforme toma cada gen del padre1 o padre2 con probabilidad 50%. Alternativas probadas:

- **0X1 (un punto):** corte aleatorio, hijo toma [0..corte] del padre1 y el resto del padre2.
- **0X2 (dos puntos):** dos cortes, alterna bloques entre ambos padres.

0X1 y 0X2 preservan bloques contiguos de genes. Solo tiene sentido cuando genes cercanos están relacionados. Aquí el orden del cromosoma es arbitrario — no hay relación de vecindad entre hiperparámetros.

### Resultados

Experimento: `pop_size=10, generations=10, N_REPS=5`

| Cruce | Media | Std | Mejor |
|---|---|---|---|
| uniform | **0.7411** | 0.0026 | **0.7455** |
| two_point | 0.7402 | 0.0027 | 0.7436 |

### Conclusión
**Se mantiene el cruce uniforme.** Con genes independientes el uniforme es teóricamente correcto y los resultados lo confirman.

---

## 5. Modelo generacional vs estacionario

### Planteamiento
- **Generacional (actual):** se reemplaza toda la población en cada generación.
- **Estacionario:** cada iteración produce 1 hijo que reemplaza al peor individuo. 1 "generación estacionaria" = pop_size reemplazos (mismo nº evaluaciones que generacional).

### Resultados

Experimento: `pop_size=10, generations=10, N_REPS=2`

| Modelo | Media | Std | Mejor |
|---|---|---|---|
| generational | **0.7417** | **0.0006** | **0.7423** |
| steady_state | 0.7405 | 0.0006 | 0.7411 |

Ambos encontraron configuraciones similares (bootstrap=1, criterion=entropy, class_weight=balanced), confirmando la meseta del problema.

### Conclusión
**Se mantiene el modelo generacional.** Gana en media y mejor valor con igual estabilidad.

---

## 6. Pc/Pm adaptativos

### Planteamiento
Pc (prob. de cruce) y Pm (prob. de mutación) parten de valores extremos opuestos y divergen linealmente con Pc + Pm = 1, con límites [0.1, 0.9] para que ninguno llegue a 0 ni a 1 (si Pm → 1.0 equivaldría a Random Search puro).

```
Prob
0.9 |  ╲── Pc
0.5 |───╳───
0.1 |  ╱── Pm
    └──────────► generaciones
```

- Pc(t) = 0.9 → 0.1: mucho cruce al inicio cuando hay diversidad, menos al final
- Pm(t) = 0.1 → 0.9: poca mutación al inicio, más al final para escapar de convergencia
- Pm es **por individuo** (decide si el individuo muta un gen aleatorio)
- Pc es **por individuo** (decide si se cruza o se copia parent1 directamente)

### Resultados

Experimento: `pop_size=20, generations=20, N_REPS=5`

| Método | Media | Std | Mejor |
|---|---|---|---|
| fijo | 0.7375 | 0.0080 | 0.7461 |
| adaptativo | **0.7427** | **0.0020** | 0.7442 |

El fijo se atascó en óptimos locales en varias reps (ej. Rep 4: 0.7223 durante 12 generaciones). El adaptativo nunca se atascó tanto — cuando Pm sube en generaciones tardías fuerza exploración y escapa del estancamiento.

### Conclusión
**Se adoptan Pc/Pm adaptativos.** Es el único cambio que mejora claramente tanto la media (+0.005) como la estabilidad (std 4x menor). Es también el cambio con mayor justificación teórica: adapta el comportamiento del AG a la fase de la búsqueda.

---

## Observación clave sobre el problema

El espacio de hiperparámetros del RF tiene una **meseta amplia** — muchas combinaciones dan ~0.72-0.745. Ningún cambio de operador ha marcado diferencia dramática por esto.

**Argumento central para el informe:** el GA no gana en accuracy absoluto sino en eficiencia — llega a resultados comparables a Grid Search (~0.741) con ~10x menos evaluaciones (2,500 vs 26,244) y mucho menos tiempo (~10 min vs ~1.7h).

---

## Configuración final decidida

| Componente | Decisión |
|---|---|
| Inicialización | Aleatoria |
| Mutación | Aleatoria por gen (rate=0.1) |
| Selección | Torneo k=3 |
| Cruce | Uniforme |
| Modelo | Generacional |
| Pc/Pm | Adaptativos (Pc: 0.9→0.1 / Pm: 0.1→0.9) |
| Elitismo | Sí |

## Ejecución final

Configuración: `pop=50, gen=50, N_REPS=5, adaptive_pc_pm=True`

| Métrica | Valor |
|---|---|
| Media | 0.7499 |
| Std | 0.0019 |
| Mejor | 0.7523 |

### Comparativa final

| Método | Evaluaciones | Mejor accuracy |
|---|---|---|
| Grid Search | 26,244 | 0.7417 |
| Random Search | 15,000 | — |
| **GA (final)** | **2,500** | **0.7523** |

**El GA supera al Grid Search en accuracy con 10x menos evaluaciones.** Las trazas muestran mejora continua hasta las últimas generaciones gracias a Pm alto — el mecanismo adaptativo funcionando como se esperaba.
