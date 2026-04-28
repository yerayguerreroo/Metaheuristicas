# def plot_individual_and_boundary(individual, model, x_min, x_max, y_min, y_max, resolution=300,
#                                  title="Frontera de decisión y pares de puntos"):
#     """
#     Dibuja la frontera de decisión del modelo y los pares de puntos de un individuo.

#     Parámetros:
#     - individual: lista de puntos [[x1,y1], [x2,y2], ...]
#                   Se asume que cada par está formado por puntos consecutivos.
#     - model: objeto con método predict(x)
#     - x_min, x_max, y_min, y_max: límites del plano
#     - resolution: densidad de la rejilla para pintar la frontera
#     - title: título de la figura
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     points = np.array(individual)

#     xx, yy = np.meshgrid(
#         np.linspace(x_min, x_max, resolution),
#         np.linspace(y_min, y_max, resolution)
#     )

#     grid_points = np.c_[xx.ravel(), yy.ravel()]
#     zz = np.array([model.predict(p) for p in grid_points]).reshape(xx.shape)

#     plt.figure(figsize=(10, 8))

#     # Regiones de decisión
#     plt.contourf(
#         xx, yy, zz,
#         alpha=0.25,
#         levels=np.arange(zz.max() + 2) - 0.5,
#         cmap="coolwarm"
#     )

#     # Frontera
#     plt.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2)

#     # Pares de puntos
#     for i in range(0, len(points) - 1, 2):
#         p1 = points[i]
#         p2 = points[i + 1]

#         c1 = model.predict(p1)
#         c2 = model.predict(p2)

#         color1 = "blue" if c1 == 0 else "red"
#         color2 = "blue" if c2 == 0 else "red"

#         plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", linewidth=1.5, alpha=0.8)
#         plt.scatter(p1[0], p1[1], color=color1, s=80, edgecolor="black", zorder=3)
#         plt.scatter(p2[0], p2[1], color=color2, s=80, edgecolor="black", zorder=3)

#     # Si sobra un punto sin pareja, también se pinta
#     if len(points) % 2 == 1:
#         p = points[-1]
#         c = model.predict(p)
#         color = "blue" if c == 0 else "red"
#         plt.scatter(p[0], p[1], color=color, s=80, edgecolor="black", zorder=3)

#     plt.xlim(x_min, x_max)
#     plt.ylim(y_min, y_max)
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.title(title)
#     plt.grid(alpha=0.3)
#     plt.show()


def plot_individual_and_boundary(individual, model, x_min, x_max, y_min, y_max, resolution=300,
                                 title="Frontera de decisión y pares de puntos"):
    """
    Dibuja la frontera de decisión del modelo, los pares de puntos de un individuo
    y la frontera deducida al unir los puntos medios de dichos pares.

    Parámetros:
    - individual: lista de puntos [[x1,y1], [x2,y2], ...]
                  Se asume que cada par está formado por puntos consecutivos.
    - model: objeto con método predict(x)
    - x_min, x_max, y_min, y_max: límites del plano
    - resolution: densidad de la rejilla para pintar la frontera
    - title: título de la figura
    """
    import numpy as np
    import matplotlib.pyplot as plt

    points = np.array(individual)

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    zz = np.array([model.predict(p) for p in grid_points]).reshape(xx.shape)

    plt.figure(figsize=(10, 8))

    # Regiones de decisión
    plt.contourf(
        xx, yy, zz,
        alpha=0.25,
        levels=np.arange(zz.max() + 2) - 0.5,
        cmap="coolwarm"
    )

    # Frontera original del modelo
    plt.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2)

    midpoints = [] # Lista para guardar los puntos medios

    # Pares de puntos
    for i in range(0, len(points) - 1, 2):
        p1 = points[i]
        p2 = points[i + 1]

        c1 = model.predict(p1)
        c2 = model.predict(p2)

        color1 = "blue" if c1 == 0 else "red"
        color2 = "blue" if c2 == 0 else "red"

        # Línea discontinua entre el par de puntos
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", linewidth=1.5, alpha=0.8)
        
        # Puntos del par
        plt.scatter(p1[0], p1[1], color=color1, s=80, edgecolor="black", zorder=3)
        plt.scatter(p2[0], p2[1], color=color2, s=80, edgecolor="black", zorder=3)

        # Calcular punto medio y guardarlo
        midpoint = (p1 + p2) / 2.0
        midpoints.append(midpoint)
        
        # Pintar el punto medio (opcional, en amarillo con forma de X para distinguirlo)
        plt.scatter(midpoint[0], midpoint[1], color="yellow", s=60, marker="X", edgecolor="black", zorder=4)

    # Si sobra un punto sin pareja, también se pinta
    if len(points) % 2 == 1:
        p = points[-1]
        c = model.predict(p)
        color = "blue" if c == 0 else "red"
        plt.scatter(p[0], p[1], color=color, s=80, edgecolor="black", zorder=3)

    # --- NUEVO: Dibujar la frontera deducida ---
    # --- NUEVO: Dibujar la frontera deducida (Para formas cerradas) ---
    if len(midpoints) > 1:
        midpoints = np.array(midpoints)
        
        # 1. Calcular el centroide (punto central) de todos los puntos medios
        centroid = np.mean(midpoints, axis=0)
        
        # 2. Calcular el ángulo de cada punto respecto al centroide usando arctan2
        # arctan2 devuelve el ángulo en radianes entre -pi y pi
        angles = np.arctan2(midpoints[:, 1] - centroid[1], midpoints[:, 0] - centroid[0])
        
        # 3. Ordenar los índices basándonos en ese ángulo
        sorted_indices = np.argsort(angles)
        midpoints_sorted = midpoints[sorted_indices]
        
        # 4. Cerrar el polígono uniendo el último punto con el primero
        # Añadimos el primer punto al final del array para que el círculo se cierre
        midpoints_sorted = np.vstack((midpoints_sorted, midpoints_sorted[0]))
        
        # Unir los puntos medios con una línea verde gruesa
        plt.plot(midpoints_sorted[:, 0], midpoints_sorted[:, 1], color="lime", linewidth=3, zorder=5, label="Frontera Deducida")
        
        # Añadir leyenda para la nueva línea
        plt.legend(loc="upper right")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()