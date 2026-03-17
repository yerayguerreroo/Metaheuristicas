import numpy as np
import random
import auxiliar
import segmentos


# ------------------- Ajuste lineal + RMSE por segmento -------------------
def linear_fit_and_rmse_segment(y, start, end, x=None):
    y = np.asarray(y, dtype=float)

    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.asarray(x, dtype=float)
        if len(x) != len(y):
            raise ValueError("x e y deben tener la misma longitud.")

    if not (0 <= start < end <= len(y)):
        raise ValueError("Segmento inválido.")

    xs = x[start:end]
    ys = y[start:end]

    if len(ys) < 2:
        return np.nan, np.nan, np.nan

    A = np.column_stack([xs, np.ones_like(xs)])
    m, b = np.linalg.lstsq(A, ys, rcond=None)[0]

    y_hat = m * xs + b
    rmse = np.sqrt(np.mean((ys - y_hat) ** 2))
    return m, b, rmse


def piecewise_linear_rmse_from_cuts(y, cuts, x=None, verbose=True):
    y = np.asarray(y, dtype=float)
    segments = auxiliar.cuts_to_segments_shared(cuts, len(y))

    out = []
    for (start, end) in segments:
        m, b, rmse = linear_fit_and_rmse_segment(y, start, end, x=x)
        out.append({"start": start, "end": end, "m": m, "b": b, "rmse": rmse})
        if verbose:
            print(f"Segmento [{start}:{end}] -> m: {m:.4f}, b: {b:.4f}, RMSE: {rmse:.4f}")
    return out


# ------------------- Score global (sin doble conteo en cortes) -------------------
def evaluate_cuts_global_rmse(y, cuts, x=None):
    """
    Calcula RMSE global de la aproximación por tramos lineales.
    Importante: como el punto de corte se comparte entre segmentos,
    evitamos contarlo dos veces:
    - en todos los segmentos excepto el primero, ignoramos el primer punto.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    if x is None:
        x = np.arange(n, dtype=float)
    else:
        x = np.asarray(x, dtype=float)

    segments = auxiliar.cuts_to_segments_shared(cuts, n)

    total_sse = 0.0
    total_count = 0

    for i, (start, end) in enumerate(segments):
        m, b, rmse = linear_fit_and_rmse_segment(y, start, end, x=x)
        if np.isnan(m) or np.isnan(b):
            return np.inf  # segmento inválido (muy corto)

        xs = x[start:end]
        ys = y[start:end]
        y_hat = m * xs + b

        # evitar doble conteo del punto compartido
        if i > 0:
            xs = xs[1:]
            ys = ys[1:]
            y_hat = y_hat[1:]

        total_sse += float(np.sum((ys - y_hat) ** 2))
        total_count += len(ys)

    if total_count == 0:
        return np.inf

    return float(np.sqrt(total_sse / total_count))


# ------------------- Mutación local de cortes (respetando tus reglas) -------------------
def mutate_cuts_like_your_generator(cuts, n, step=3):
    """
    Mueve ligeramente algunos cortes y luego repara para mantener:
    - rango: cortes en [2, n-3] (por tu range(2, n-2))
    - orden creciente
    - distancia mínima entre cortes: >= 2 (por tu check)
    """
    if len(cuts) == 0:
        return cuts[:]

    new_cuts = cuts[:]
    idx = random.randrange(len(new_cuts))
    delta = random.randint(-step, step)
    new_cuts[idx] += delta

    # clamp a tu rango
    lo = 2
    hi = n - 3
    new_cuts[idx] = max(lo, min(hi, new_cuts[idx]))

    new_cuts.sort()

    # reparar gaps >= 2
    for i in range(1, len(new_cuts)):
        if new_cuts[i] - new_cuts[i - 1] < 2:
            new_cuts[i] = new_cuts[i - 1] + 2

    # si nos pasamos por arriba, reajustar hacia atrás
    if new_cuts[-1] > hi:
        new_cuts[-1] = hi
        for i in range(len(new_cuts) - 2, -1, -1):
            new_cuts[i] = min(new_cuts[i], new_cuts[i + 1] - 2)

    # si nos pasamos por abajo, reajustar hacia delante
    if new_cuts[0] < lo:
        new_cuts[0] = lo
        for i in range(1, len(new_cuts)):
            new_cuts[i] = max(new_cuts[i], new_cuts[i - 1] + 2)

    return new_cuts


# ------------------- Optimización por épocas -------------------
def optimize_by_epochs(y, num_segmentos, epochs=200, candidates_per_epoch=40, p_random=0.25, seed=123, x=None):
    random.seed(seed)
    y = np.asarray(y, dtype=float)
    n = len(y)

    # inicial: cortes aleatorios usando TU función
    best_cuts = segmentos.generar_segmentos(num_segmentos, n)
    best_score = evaluate_cuts_global_rmse(y, best_cuts, x=x)

    print(f"Inicial | best RMSE global = {best_score:.6f} | cuts = {best_cuts}")

    for ep in range(1, epochs + 1):
        improved = False

        # opcional: ir reduciendo el step para refinar al final
        step = max(1, int(round(6 - 5 * (ep - 1) / max(1, epochs - 1))))  # de ~6 a 1

        for _ in range(candidates_per_epoch):
            if random.random() < p_random:
                # exploración: reinicio aleatorio equiprobable (tu función)
                cuts = segmentos.generar_segmentos(num_segmentos, n)
            else:
                # explotación: mutación de la mejor
                cuts = mutate_cuts_like_your_generator(best_cuts, n, step=step)

            score = evaluate_cuts_global_rmse(y, cuts, x=x)

            if score < best_score:
                best_score = score
                best_cuts = cuts
                improved = True

        print(f"Epoch {ep:4d} | best RMSE global = {best_score:.6f} | step={step} | improved={improved}")

    return best_cuts, best_score


# ------------------- main -------------------
def main():
    serie1 = auxiliar.cargar_datos("TS1.txt")
    if serie1 is None:
        return

    num_segmentos = 9

    best_cuts, best_rmse = optimize_by_epochs(
        serie1,
        num_segmentos=num_segmentos,
        epochs=200,
        candidates_per_epoch=50,
        p_random=0.25,
        seed=123
    )

    print("\n===== MEJOR SOLUCIÓN =====")
    print("Best cuts:", best_cuts)
    print("Best RMSE global:", best_rmse)

    print("\nDetalle por segmento:")
    piecewise_linear_rmse_from_cuts(serie1, best_cuts, verbose=True)


if __name__ == "__main__":
    main()