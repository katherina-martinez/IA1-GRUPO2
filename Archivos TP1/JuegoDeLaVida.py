"""
================================================================================
JUEGO DE LA VIDA DE CONWAY - Modelado como Sistema de Agentes Inteligentes
================================================================================

Enfoque conceptual (PEAS):
    Cada célula de la grilla se modela como un AGENTE REACTIVO SIMPLE
    (simple reflex agent) que:
        - PERCIBE   -> el estado de sus 8 vecinas (vecindad de Moore) en t
        - DECIDE    -> aplica una regla condición-acción (las reglas de Conway)
        - ACTÚA     -> fija su propio estado en t+1 (viva / muerta)

    El "entorno" completo es un AUTÓMATA CELULAR: múltiples agentes idénticos
    que perciben y actúan de forma SIMULTÁNEA sobre una grilla compartida.

================================================================================
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap


# ============================================================================
# 1. EL ENTORNO (la grilla donde "viven" los agentes-célula)
# ============================================================================

class EntornoJuegoDeLaVida:
    """
    Representa el ENTORNO del autómata celular: una grilla bidimensional
    de células, cada una de las cuales es un agente reactivo simple.

    """

    def __init__(self, filas: int, columnas: int, wrap: bool = True):
        self.filas = filas
        self.columnas = columnas
        self.wrap = wrap
        self.grid = np.zeros((filas, columnas), dtype=np.uint8)

    # -------------------- Inicialización --------------------

    def sembrar_patron(self, patron: np.ndarray, origen: tuple[int, int]):
        """Coloca un patrón (submatriz de 0/1) en la grilla en la posición 'origen'."""
        f0, c0 = origen
        pf, pc = patron.shape
        self.grid[f0:f0 + pf, c0:c0 + pc] = patron

    def sembrar_aleatorio(self, probabilidad_viva: float = 0.2, semilla: int | None = None):
        """Inicializa la grilla de forma aleatoria (útil para pruebas de estabilidad)."""
        rng = np.random.default_rng(semilla)
        self.grid = (rng.random((self.filas, self.columnas)) < probabilidad_viva).astype(np.uint8)

    # -------------------- SENSORES --------------------

    def contar_vecinos_vivos(self) -> np.ndarray:
        """
        SENSORES de cada agente-célula: para cada casilla, cuenta cuántas de
        sus 8 vecinas (vecindad de Moore) están vivas EN LA ITERACIÓN ACTUAL t.

        Se calcula de forma vectorizada: se suman las 8 versiones desplazadas
        de la grilla (np.roll en las 8 direcciones), lo cual es equivalente
        a convolucionar la grilla con un kernel 3x3 de unos con el centro en 0.
        Esto evita recorrer la grilla con bucles anidados en Python puro,
        lo que sería mucho más lento para grillas grandes.
        """
        vecinos = np.zeros_like(self.grid, dtype=np.uint8)

        for df in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if df == 0 and dc == 0:
                    continue  # se excluye la propia célula (centro del kernel)

                if self.wrap:
                    # Tablero toroidal: los bordes se conectan (sin frontera artificial)
                    desplazado = np.roll(np.roll(self.grid, df, axis=0), dc, axis=1)
                else:
                    # Tablero con bordes fijos: fuera de la grilla se considera "muerto"
                    desplazado = np.zeros_like(self.grid)
                    fs_o, fe_o = max(0, -df), self.filas - max(0, df)
                    cs_o, ce_o = max(0, -dc), self.columnas - max(0, dc)
                    fs_d, fe_d = max(0, df), self.filas - max(0, -df)
                    cs_d, ce_d = max(0, dc), self.columnas - max(0, -dc)
                    desplazado[fs_d:fe_d, cs_d:ce_d] = self.grid[fs_o:fe_o, cs_o:ce_o]

                vecinos += desplazado

        return vecinos

    # -------------------- FUNCIÓN DE AGENTE (reglas + actuadores) --------------------

    def actualizar(self):
        """
        Aplica la FUNCIÓN DE AGENTE de cada célula de forma simultánea
        (ACTUADORES): a partir del conteo de vecinos vivos en t, cada
        agente decide su nuevo estado en t+1 según las reglas de Conway.

        Reglas (todas evaluadas sobre el estado en t, nunca sobre el que
        se va calculando, para respetar la actualización simultánea):
            1. Nacer : célula muerta con exactamente 3 vecinas vivas -> viva
            2. Vivir : célula viva con 2 o 3 vecinas vivas -> sigue viva
            3. Morir : en cualquier otro caso -> muerta
               (aislamiento: <2 vecinas ; sobrepoblación: >3 vecinas)
        """
        vecinos = self.contar_vecinos_vivos()
        vivas_actual = self.grid == 1

        sobrevive = vivas_actual & ((vecinos == 2) | (vecinos == 3))   # Regla "Vivir"
        nace = (~vivas_actual) & (vecinos == 3)                        # Regla "Nacer"
        # Todo lo que no cumple sobrevive ni nace queda en 0 -> Regla "Morir"

        nuevo_grid = np.zeros_like(self.grid)
        nuevo_grid[sobrevive | nace] = 1
        self.grid = nuevo_grid

    # -------------------- Métricas de rendimiento --------------------

    def poblacion(self) -> int:
        """Cantidad de células vivas (métrica simple de rendimiento)."""
        return int(self.grid.sum())


# ============================================================================
# 2. VISUALIZACIÓN / SIMULACIÓN ANIMADA
# ============================================================================

class SimuladorVisual:
    """Envuelve un EntornoJuegoDeLaVida y lo anima con matplotlib."""

    def __init__(self, entorno: EntornoJuegoDeLaVida, intervalo_ms: int = 150):
        self.entorno = entorno
        self.intervalo_ms = intervalo_ms
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.cmap = ListedColormap(["#0d1117", "#39d353"])  # muerta / viva
        self.im = self.ax.imshow(self.entorno.grid, cmap=self.cmap, vmin=0, vmax=1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.titulo = self.ax.set_title(self._titulo_texto(0))

    def _titulo_texto(self, t: int) -> str:
        return f"Juego de la Vida de Conway | t={t} | población={self.entorno.poblacion()}"

    def _paso(self, frame: int):
        if frame > 0:
            self.entorno.actualizar()
        self.im.set_data(self.entorno.grid)
        self.titulo.set_text(self._titulo_texto(frame))
        return [self.im, self.titulo]

    def guardar_gif(self, ruta: str, n_frames: int = 60, fps: int = 8):
        """Genera y guarda la animación directamente como un archivo GIF."""
        anim = animation.FuncAnimation(
            self.fig, self._paso, frames=n_frames,
            interval=self.intervalo_ms, blit=False, repeat=False
        )
        # Guarda el GIF usando Pillow como motor de renderizado
        anim.save(ruta, writer=animation.PillowWriter(fps=fps))
        plt.close(self.fig)


# ============================================================================
# 3. DEMO PRINCIPAL
# ============================================================================

def main():
    filas, columnas = 30, 30
    entorno = EntornoJuegoDeLaVida(filas, columnas, wrap=True)

    # Inicialización aleatoria de la grilla
    entorno.sembrar_aleatorio(probabilidad_viva=0.2, semilla=42)

    sim = SimuladorVisual(entorno, intervalo_ms=150)

    # Genera y guarda el archivo GIF directamente al ejecutar en VS Code
    nombre_archivo = "conway_demo.gif"
    print("Generando animación GIF, por favor espera un momento...")
    sim.guardar_gif(nombre_archivo, n_frames=80, fps=8)
    print(f"¡Éxito! La simulación se ha guardado correctamente como '{nombre_archivo}'.")


if __name__ == "__main__":
    main()