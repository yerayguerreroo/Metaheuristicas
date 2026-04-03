# Memoria de Fine-Tuning del Algoritmo Genético

## Configuración base del AG

| Parámetro | Valor |
|---|---|
| pop_size | 10 (pruebas) / 50 (final) |
| generations | 10 (pruebas) / 50 (final) |
| mutation_rate | 0.1 |
| selección | torneo k=3 |
| cruce | uniforme (50% por gen) |
| elitismo | sí (el mejor pasa directo) |

Cada evaluación = 5-fold CV sobre RandomForest ≈ 0.23s por individuo.

---

## 1. Inicialización de población

### Planteamiento
La población inicial puede generarse de forma aleatoria o forzando diversidad mínima entre individuos.

### Métodos comparados

**Aleatoria:** cada individuo se genera completamente al azar dentro de los rangos de cada gen.

**Secuencial (distance-based):** el primer individuo es aleatorio. Cada nuevo candidato solo se acepta si su distancia euclídea normalizada a todos los individuos ya aceptados es >= D. Si tras `max_attempts` intentos no se encuentra candidato válido, D se reduce un 10% para evitar bucles infinitos.

- Distancia calculada en espacio normalizado [0,1]^10 para evitar que genes con rangos grandes (n_estimators: 10-300) dominen sobre genes con rangos pequeños (max_features: 0.1-1.0).
- D inicial = 0.5 (distancia máxima posible: sqrt(10) ≈ 3.16).

### Resultados
Ambos métodos dieron resultados estadísticamente similares en media y std.

### Conclusión
**Se mantiene la inicialización aleatoria.** La secuencial añade coste computacional sin beneficio observable. Con solo 10 genes el espacio es lo suficientemente manejable como para que la aleatoriedad cubra bien la diversidad inicial.

---

## 2. Operador de mutación

### Planteamiento
La mutación actual resetea el gen a un valor completamente nuevo. Se planteó una mutación por perturbación (creep) que en lugar de saltar hace ajustes pequeños.

### Métodos comparados

**Aleatoria (random):** resetea el gen a un valor aleatorio dentro de sus límites. Para todos los tipos de gen.

**Creep:** suma un delta de ±10% del rango del gen al valor actual, manteniendo el resultado dentro de los límites (clamp). Los genes binarios (bootstrap, criterion, class_weight — índices 5, 6, 7) hacen flip en lugar de perturbación, ya que no tiene sentido perturbar un valor 0/1.

### Resultados

Experimento: `pop_size=10, generations=10, N_REPS=5`

| Mutación | Media | Std | Mejor |
|---|---|---|---|
| random | **0.7442** | **0.0006** | 0.7448 |
| creep | 0.7398 | 0.0019 | **0.7417** |

Experimento previo con `generations=3` mostraba creep más volátil (std=0.0141), lo que sugería que necesitaba más generaciones para explorar. Con generations=10 la diferencia persiste.

### Conclusión
**Se mantiene la mutación aleatoria.** Random es mejor en media y considerablemente más estable (std 4x menor). El argumento es que con 10 genes y rangos amplios, los saltos grandes de random cubren mejor el espacio de búsqueda. Creep sería más adecuado en problemas con muchos más genes donde un salto aleatorio grande destruiría soluciones parcialmente buenas.

---

## 3. Selección de padres

### Planteamiento
Se mantiene torneo de tamaño k=3. Se planteó ranking como alternativa pero se consideró de impacto bajo para este problema.

### Estado
Pendiente — pospuesto.

---

## 4. Operadores de cruce

### Planteamiento
El cruce uniforme actual toma cada gen del padre1 o padre2 con probabilidad 50%. Se plantearon como alternativas:

- **0X1 (un punto):** se elige un punto de corte aleatorio. El hijo toma genes [0..corte] del padre1 y [corte+1..fin] del padre2.
- **0X2 (dos puntos):** se eligen dos puntos de corte. El hijo toma [0..c1] del padre1, [c1+1..c2] del padre2, [c2+1..fin] del padre1.

0X1 y 0X2 preservan bloques contiguos de genes. Tiene sentido cuando genes cercanos están relacionados entre sí. Aquí el orden de los genes es arbitrario (n_estimators en pos. 0, bootstrap en pos. 5 sin relación de vecindad), por lo que el cruce uniforme es teóricamente superior. Aun así vale la pena comprobarlo empíricamente.

### Estado
Pendiente — por probar.

---

## 5. Pc/Pm adaptativos

### Planteamiento
El profesor mostró una gráfica donde la probabilidad de cruce (Pc) y de mutación (Pm) parten ambas de 0.5 y divergen inversamente a lo largo de las generaciones. Se detienen antes de los extremes porque:
- Si Pm → 1.0: cada gen se regenera completamente cada generación → equivale a Random Search.
- Si Pc → 0: no hay recombinación, solo mutación.

Relación propuesta: `Pc + Pm = 1`, con Pc creciendo de 0.5 a 0.9 y Pm bajando de 0.5 a 0.1.

```
Prob
0.9 |              ╱── Pc
0.5 |─────────────╳
0.1 |              ╲── Pm
    └──────────────────► generaciones
```

Implementación lineal por generación:
```
Pc(t) = 0.5 + 0.4 * t / (T - 1)   →  0.5 a 0.9
Pm(t) = 1 - Pc(t)                  →  0.5 a 0.1
```

Nota: actualmente el código no tiene Pc explícito — el cruce siempre se aplica. Para implementar esto habría que añadir Pc como probabilidad de aplicar el cruce a cada pareja; si no se cruzan, el hijo es copia de uno de los padres.

### Estado
Pendiente — por implementar y probar.

---

## 6. Modelo generacional vs estacionario

### Planteamiento
- **Generacional (actual):** se reemplaza toda la población en cada generación.
- **Estacionario:** se reemplaza solo 1-2 individuos por iteración, conservando el resto.

Con evaluaciones caras (cada individuo ~0.23s), el modelo estacionario puede ser ventajoso porque no descarta buenas soluciones tan fácilmente.

### Estado
Pendiente — por valorar.
