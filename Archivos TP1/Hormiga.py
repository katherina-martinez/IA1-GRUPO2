from enum import Enum
from dataclasses import dataclass
import random
import time
import os
from typing import Dict


class Color(Enum):
    #Representa el estado/color de una casilla en la grilla, con un numero 1 o 0.
    WHITE = 0
    BLACK = 1

    def toggle(self) -> 'Color':
        #Alterna el color entre Blanco y Negro.
        return Color.BLACK if self == Color.WHITE else Color.WHITE


class Direction(Enum):
    #Representa la orientación de la hormiga y sus desplazamientos (dx, dy).
    NORTH = (0, -1)
    EAST = (1, 0)
    SOUTH = (0, 1)
    WEST = (-1, 0)

    def turn_right(self) -> 'Direction':
        #Gira 90 grados a la derecha (sentido horario).
        directions = list(Direction)
        idx = (directions.index(self) + 1) % len(directions)
        return directions[idx]

    def turn_left(self) -> 'Direction':
        #Gira 90 grados a la izquierda (sentido antihorario)."""
        directions = list(Direction)
        idx = (directions.index(self) - 1) % len(directions)
        return directions[idx]


@dataclass(frozen=True)
class Position:
    """Objeto de valor inmutable para coordenadas (x, y)."""
    x: int
    y: int

    def move(self, direction: Direction, max_width: int, max_height: int) -> 'Position':
        """
        Calcula la nueva posición según la dirección.
        Aplica un comportamiento esférico/toroidal (wrap-around) al tocar los bordes.
        """
        dx, dy = direction.value
        new_x = (self.x + dx) % max_width
        new_y = (self.y + dy) % max_height
        return Position(new_x, new_y)


class Ant:
    """Representa el agente (Hormiga) con su posición y orientación."""

    def __init__(self, position: Position, direction: Direction):
        self.position = position
        self.direction = direction

    def rotate(self, current_color: Color) -> None:
        """Cambia la orientación según la regla del color actual."""
        if current_color == Color.WHITE:
            self.direction = self.direction.turn_right()
        else:
            self.direction = self.direction.turn_left()

    def advance(self, max_width: int, max_height: int) -> None:
        """Avanza una casilla en la dirección actual."""
        self.position = self.position.move(self.direction, max_width, max_height)


class Grid:
    """Gestiona el tablero y el estado de sus casillas."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Estructura dispersa para optimizar memoria: por defecto toda casilla no mapeada es BLANCA
        self._cells: Dict[Position, Color] = {}

    def get_color(self, position: Position) -> Color:
        """Obtiene el color actual de la casilla."""
        return self._cells.get(position, Color.WHITE)

    def toggle_color(self, position: Position) -> Color:
        """Invierte el color de la casilla dada y devuelve el nuevo color."""
        new_color = self.get_color(position).toggle()
        self._cells[position] = new_color
        return new_color


class LangtonSimulation:
    """Orquestador principal de la simulación de la Hormiga de Langton."""

    def __init__(self, width: int = 30, height: int = 20):
        self.grid = Grid(width, height)

        # Ubicación inicial aleatoria y orientación inicial aleatoria
        initial_position = Position(
            x=random.randint(0, width - 1),
            y=random.randint(0, height - 1)
        )
        initial_direction = random.choice(list(Direction))

        self.ant = Ant(initial_position, initial_direction)
        self.steps = 0

    def step(self) -> None:
        """Ejecuta un ciclo completo según las reglas de Langton."""
        current_pos = self.ant.position
        current_color = self.grid.get_color(current_pos)

        # 1. Cambia el color del cuadrado
        self.grid.toggle_color(current_pos)

        # 2. La hormiga gira 90° a la derecha (blanco) o izquierda (negro)
        self.ant.rotate(current_color)

        # 3. Avanza un cuadrado
        self.ant.advance(self.grid.width, self.grid.height)

        self.steps += 1

    def render(self) -> None:
        """Dibuja el tablero en la consola limpia usando caracteres Unicode."""
        os.system('cls' if os.name == 'nt' else 'clear')

        color_symbols = {
            Color.WHITE: '░░',
            Color.BLACK: '██'
        }
        ant_symbols = {
            Direction.NORTH: '▲▲',
            Direction.EAST: '►►',
            Direction.SOUTH: '▼▼',
            Direction.WEST: '◄◄'
        }

        print(f"Paso: {self.steps} | Posición: ({self.ant.position.x}, {self.ant.position.y}) | Orientación: {self.ant.direction.name}")
        print("┌" + "──" * self.grid.width + "┐")

        for y in range(self.grid.height):
            line = "│"
            for x in range(self.grid.width):
                pos = Position(x, y)
                if pos == self.ant.position:
                    line += ant_symbols[self.ant.direction]
                else:
                    line += color_symbols[self.grid.get_color(pos)]
            line += "│"
            print(line)

        print("└" + "──" * self.grid.width + "┘")

    def run(self, max_steps: int = 300, delay: float = 0.05) -> None:
        """Ejecuta el bucle principal de visualización."""
        try:
            for _ in range(max_steps):
                self.render()
                self.step()
                time.sleep(delay)
            self.render()
            print("\n[+] Simulación completada.")
        except KeyboardInterrupt:
            print("\n[-] Simulación detenida por el usuario.")


# --- Punto de Entrada Normal: Simula TODOS los pasos ---
#if __name__ == "__main__":
#    # Configuración de grilla
#    sim = LangtonSimulation(width=60, height=40)
#    sim.run(max_steps=99999, delay=0)

# --- Bloque que permite la visualizacion de la Avenida ---
if __name__ == "__main__":
    #  Agrandamos el tablero
    sim = LangtonSimulation(width=120, height=60)

    # 2. Forzamos a la hormiga a iniciar en el centro exacto para darle espacio
    center_pos = Position(x=60, y=30)
    sim.ant = Ant(center_pos, Direction.NORTH)

    # 3. Avance rápido: Simulamos 10,000 pasos sin renderizar ni pausas
    for _ in range(10000):
        sim.step()

    # 4. Ahora sí, comenzamos a dibujar.
    # Vas a ver a la hormiga construyendo la avenida en diagonal.
    sim.run(max_steps=7000, delay=0.03)