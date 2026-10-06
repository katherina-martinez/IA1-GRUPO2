"""
K-Means (20 puntos, 2 grupos) + KNN (3 puntos restantes, K en [1, 3, 5, 6, 9]).

Los 3 puntos que se clasifican con KNN pueden elegirse de dos maneras
(pregunta S/n, por defecto Sí):
  - Sí: los 3 puntos más cercanos a ser equidistantes de ambos centroides
        (los más "ambiguos"), para que el efecto de K se note más.
  - No: simplemente los 3 últimos puntos generados.

K-Means y KNN están implementados desde cero con NumPy (sin sklearn).
Requisitos: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Generación de datos
# --------------------------------------------------------------------------- #
class PointGenerator:
    """Genera puntos 2D aleatorios en el intervalo [low, high]."""

    def __init__(self, n_points=23, low=0.0, high=5.0, seed=None):
        self.n_points = n_points
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(seed)

    def generate(self):
        return self.rng.uniform(self.low, self.high, size=(self.n_points, 2))


# --------------------------------------------------------------------------- #
# K-Means
# --------------------------------------------------------------------------- #
class KMeans:
    """Implementación de K-Means (algoritmo de Lloyd)."""

    def __init__(self, n_clusters=2, max_iter=300, tol=1e-9):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.n_iter_ = 0

    @staticmethod
    def distances(X, centroids):
        """Matriz (n_puntos x n_centroides) de distancias euclídeas."""
        return np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)

    def fit(self, X, initial_centroids):
        X = np.asarray(X, dtype=float)
        centroids = np.array(initial_centroids, dtype=float)

        for i in range(1, self.max_iter + 1):
            # 1) Asignación: cada punto al centroide más cercano
            labels = np.argmin(self.distances(X, centroids), axis=1)

            # 2) Actualización: nuevo centroide = media del grupo
            new_centroids = centroids.copy()
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) > 0:          # si un grupo queda vacío, se conserva
                    new_centroids[k] = members.mean(axis=0)

            # 3) Convergencia
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            self.n_iter_ = i
            if shift <= self.tol:
                break

        self.centroids = centroids
        self.labels_ = np.argmin(self.distances(X, centroids), axis=1)
        return self


# --------------------------------------------------------------------------- #
# KNN
# --------------------------------------------------------------------------- #
class KNNClassifier:
    """Clasificador K-Nearest Neighbors (distancia euclídea, voto por mayoría)."""

    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=int)
        return self

    def predict_one(self, point, k):
        """
        Devuelve (etiqueta, índices_vecinos, distancias_vecinos).
        Desempate (posible con K par, ej. K=6 con 3 vs 3):
        gana el grupo cuya suma de distancias a sus vecinos sea menor.
        """
        dists = np.linalg.norm(self.X - point, axis=1)
        idx = np.argsort(dists)[:k]
        neighbor_labels = self.y[idx]

        classes, counts = np.unique(neighbor_labels, return_counts=True)
        winners = classes[counts == counts.max()]

        if len(winners) == 1:
            label = winners[0]
        else:
            sums = {c: dists[idx][neighbor_labels == c].sum() for c in winners}
            label = min(sums, key=sums.get)

        return int(label), idx, dists[idx]


# --------------------------------------------------------------------------- #
# Visualización
# --------------------------------------------------------------------------- #
class Plotter:
    """Se encarga de todos los gráficos."""

    COLORS = {0: "red", 1: "blue"}
    NAMES = {0: "Grupo 0 (Rojo)", 1: "Grupo 1 (Azul)"}

    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))

    def _setup(self, title):
        self.ax.clear()
        self.ax.set_xlim(-0.2, 5.2)
        self.ax.set_ylim(-0.2, 5.2)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)

    def _refresh(self):
        self.fig.canvas.draw_idle()
        plt.pause(0.1)

    def _draw_groups(self, X, labels):
        for k, color in self.COLORS.items():
            pts = X[labels == k]
            self.ax.scatter(pts[:, 0], pts[:, 1], c=color, s=60,
                            label=self.NAMES[k], alpha=0.8)

    def _draw_centroids(self, centroids):
        for k, c in enumerate(centroids):
            self.ax.scatter(c[0], c[1], c=self.COLORS[k], marker="X", s=250,
                            edgecolors="black", linewidths=1.5,
                            label=f"Centroide {k}")

    def _draw_boundary(self, centroids):
        """Mediatriz entre los dos centroides: puntos equidistantes."""
        c0, c1 = centroids
        v = c1 - c0
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return
        mid = (c0 + c1) / 2
        direction = np.array([-v[1], v[0]]) / norm
        t = np.array([-10.0, 10.0])
        line = mid[None, :] + t[:, None] * direction[None, :]
        self.ax.plot(line[:, 0], line[:, 1], "k--", linewidth=1, alpha=0.6,
                     label="Equidistancia entre centroides")

    def _draw_pending(self, pending):
        self.ax.scatter(pending[:, 0], pending[:, 1], c="black", marker="*",
                        s=250, label="Puntos sin clasificar")
        for i, p in enumerate(pending):
            self.ax.annotate(f"P{i + 1}", p, textcoords="offset points",
                             xytext=(8, 8), fontweight="bold")

    def show_raw(self, X):
        self._setup("Paso 1: 23 puntos aleatorios")
        self.ax.scatter(X[:, 0], X[:, 1], c="gray", s=60)
        self._refresh()

    def show_kmeans(self, X_train, labels, centroids, pending, n_iter):
        self._setup(f"Paso 2: K-Means (2 grupos, {n_iter} iteraciones)")
        self._draw_groups(X_train, labels)
        self._draw_centroids(centroids)
        self._draw_boundary(centroids)
        self._draw_pending(pending)
        self.ax.legend(loc="upper right", fontsize=8)
        self._refresh()

    def show_knn(self, X_train, y_train, centroids, pending, pred_labels,
                 neighbors, k):
        self._setup(f"Paso 3: KNN con K = {k}")
        self._draw_groups(X_train, y_train)
        self._draw_boundary(centroids)
        for i, p in enumerate(pending):
            color = self.COLORS[pred_labels[i]]
            # líneas hacia los K vecinos usados
            for j in neighbors[i]:
                self.ax.plot([p[0], X_train[j, 0]], [p[1], X_train[j, 1]],
                             color=color, alpha=0.35, linewidth=1)
            self.ax.scatter(p[0], p[1], c=color, marker="*", s=350,
                            edgecolors="black", linewidths=1.5)
            self.ax.annotate(f"P{i + 1}", p, textcoords="offset points",
                             xytext=(10, 10), fontweight="bold")
        self.ax.legend(loc="upper right", fontsize=8)
        self._refresh()


# --------------------------------------------------------------------------- #
# Aplicación principal
# --------------------------------------------------------------------------- #
class Application:
    N_POINTS = 23
    N_TRAIN = 20
    N_CLUSTERS = 2
    K_VALUES = [1, 3, 5, 6, 9]

    def __init__(self, seed=None):
        self.points = PointGenerator(self.N_POINTS, 0, 5, seed).generate()
        self.X_train = None      # 20 puntos -> K-Means
        self.X_pending = None    # 3 puntos  -> KNN
        self.plotter = Plotter()
        self.rng = np.random.default_rng(seed)

    # ---- entradas por consola -------------------------------------------- #
    def _ask_centroids(self):
        print("\n=== Centroides iniciales ===")
        print("Ingresá cada centroide como 'x y' (valores entre 0 y 5).")
        print("Presioná Enter sin escribir nada para elegir uno al azar.\n")

        centroids = []
        for k in range(self.N_CLUSTERS):
            while True:
                raw = input(f"Centroide {k} (x y): ").strip().replace(",", " ")
                if raw == "":
                    c = self.points[self.rng.integers(len(self.points))]
                    print(f"  -> aleatorio: ({c[0]:.3f}, {c[1]:.3f})")
                    centroids.append(c)
                    break
                try:
                    x, y = map(float, raw.split())
                    centroids.append(np.array([x, y]))
                    break
                except ValueError:
                    print("  Entrada inválida. Ejemplo: 1.5 3.2")
        return np.array(centroids)

    @staticmethod
    def _ask_yes_no(prompt, default=True):
        hint = "[S/n]" if default else "[s/N]"
        while True:
            raw = input(f"{prompt} {hint}: ").strip().lower()
            if raw == "":
                return default
            if raw in ("s", "si", "sí", "y", "yes"):
                return True
            if raw in ("n", "no"):
                return False
            print("  Respondé con S o N (Enter = opción predefinida).")

    # ---- división entrenamiento / pendientes ------------------------------ #
    @staticmethod
    def _boundary_distance(points, centroids):
        """Distancia de cada punto a la mediatriz entre los dos centroides
        (0 = perfectamente equidistante)."""
        c0, c1 = centroids
        d0 = np.linalg.norm(points - c0, axis=1)
        d1 = np.linalg.norm(points - c1, axis=1)
        sep = max(np.linalg.norm(c1 - c0), 1e-12)
        return np.abs(d0 ** 2 - d1 ** 2) / (2 * sep)

    def _split_default(self, init):
        """20 primeros puntos -> K-Means, 3 últimos -> KNN."""
        train = self.points[:self.N_TRAIN]
        pending = self.points[self.N_TRAIN:]
        km = KMeans(self.N_CLUSTERS).fit(train, init)
        return train, pending, km, True

    def _split_by_ambiguity(self, init, max_rounds=20):
        """
        Elige como pendientes los puntos más cercanos a la equidistancia.
        Como los centroides dependen de los puntos de entrenamiento, se itera
        hasta que la selección sea estable:
          1) K-Means con los 23 puntos -> centroides provisorios
          2) se separan los 3 puntos más cercanos a la frontera
          3) K-Means con los 20 restantes -> centroides nuevos
          4) se repite desde 2) hasta que los 3 puntos no cambien
        Devuelve (X_train, X_pending, kmeans_final, estable).
        """
        n_pending = self.N_POINTS - self.N_TRAIN
        km = KMeans(self.N_CLUSTERS).fit(self.points, init)
        pending_idx, stable = None, False

        for _ in range(max_rounds):
            gap = self._boundary_distance(self.points, km.centroids)
            new_idx = np.argsort(gap)[:n_pending]      # el más ambiguo primero
            if pending_idx is not None and set(new_idx) == set(pending_idx):
                stable = True
                break
            pending_idx = new_idx
            mask = np.ones(self.N_POINTS, dtype=bool)
            mask[pending_idx] = False
            km = KMeans(self.N_CLUSTERS).fit(self.points[mask], init)

        mask = np.ones(self.N_POINTS, dtype=bool)
        mask[pending_idx] = False
        return self.points[mask], self.points[pending_idx], km, stable

    # ---- flujo del programa ---------------------------------------------- #
    def run(self):
        # 1) Puntos
        print(f"Se generaron {self.N_POINTS} puntos aleatorios en [0, 5].")
        self.plotter.show_raw(self.points)
        input("\n[Enter] para ejecutar K-Means...")

        # 2) Configuración + K-Means sobre 20 puntos
        init = self._ask_centroids()
        print()
        ambiguous = self._ask_yes_no(
            "¿Clasificar con KNN los puntos cercanos a la equidistancia "
            "entre centroides?", default=True)

        if ambiguous:
            self.X_train, self.X_pending, kmeans, stable = \
                self._split_by_ambiguity(init)
            if not stable:
                print("  Aviso: la selección no se estabilizó del todo; "
                      "se usa la última.")
        else:
            self.X_train, self.X_pending, kmeans, _ = self._split_default(init)

        print(f"\nK-Means convergió en {kmeans.n_iter_} iteraciones "
              f"(sobre {len(self.X_train)} puntos).")
        for k, c in enumerate(kmeans.centroids):
            n = int(np.sum(kmeans.labels_ == k))
            print(f"  Centroide final {k}: ({c[0]:.3f}, {c[1]:.3f})  -> {n} puntos")

        d = KMeans.distances(self.X_pending, kmeans.centroids)
        gap = self._boundary_distance(self.X_pending, kmeans.centroids)
        titulo = ("más cercanos a la equidistancia" if ambiguous
                  else "últimos 3 generados")
        print(f"\nPuntos a clasificar con KNN ({titulo}):")
        for i, p in enumerate(self.X_pending):
            print(f"  P{i + 1} ({p[0]:.2f}, {p[1]:.2f})  d0={d[i, 0]:.3f}  "
                  f"d1={d[i, 1]:.3f}  distancia a la frontera={gap[i]:.3f}")

        self.plotter.show_kmeans(self.X_train, kmeans.labels_, kmeans.centroids,
                                 self.X_pending, kmeans.n_iter_)

        # ------------------------- BREAKPOINT ------------------------------ #
        input("\n[BREAKPOINT] Enter para clasificar los 3 puntos restantes con KNN...")

        # 3) KNN para cada K
        knn = KNNClassifier().fit(self.X_train, kmeans.labels_)
        print(f"\nValores de K a evaluar: {self.K_VALUES}")

        for n, k in enumerate(self.K_VALUES):
            print(f"\n--- K = {k} ---")
            preds, neighbors = [], []
            for i, p in enumerate(self.X_pending):
                label, idx, _ = knn.predict_one(p, k)
                votes = np.bincount(kmeans.labels_[idx], minlength=self.N_CLUSTERS)
                preds.append(label)
                neighbors.append(idx)
                print(f"  P{i + 1} ({p[0]:.2f}, {p[1]:.2f}) -> "
                      f"{Plotter.NAMES[label]}   "
                      f"[votos: Rojo={votes[0]}, Azul={votes[1]}]")
            self.plotter.show_knn(self.X_train, kmeans.labels_, kmeans.centroids,
                                  self.X_pending, preds, neighbors, k)

            if n < len(self.K_VALUES) - 1:
                input(f"\n[BREAK] Enter para pasar a K = {self.K_VALUES[n + 1]}...")

        print("\nFin. Cerrá la ventana del gráfico para terminar.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    # seed=None -> puntos distintos en cada ejecución. Poné un entero para reproducir.
    Application(seed=None).run()
