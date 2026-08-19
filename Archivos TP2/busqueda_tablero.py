"""
Espacio de estados: tablero en forma de cruz (A..W excepto letras usadas)
Nodo inicial: I   |   Nodo objetivo: F
Costo de entrar a una casilla estándar = 1
Costo de entrar a la casilla W = 30
Heurística: Distancia de Manhattan hacia F
Desempate: orden alfabético
"""

import heapq

# ---------------------------------------------------------------
# 1. MODELO DEL ESPACIO DE ESTADOS
# ---------------------------------------------------------------

# (fila, columna) de cada casilla, extraídas de la imagen
COORDS = {
    'A': (1, 4), 'B': (1, 5),
    'C': (2, 4), 'D': (2, 5), 'E': (2, 6),
    'G': (3, 1), 'I': (3, 2), 'W': (3, 3), 'K': (3, 4), 'M': (3, 5), 'N': (3, 6),
    'P': (4, 1), 'Q': (4, 2), 'R': (4, 3), 'T': (4, 4), 'F': (4, 5),
}

# Grafo de adyacencias YA filtrado por los muros rojos:
#   C-D bloqueado, D-E bloqueado, W-R bloqueado, T-F bloqueado
GRAPH = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'K'],
    'D': ['B', 'M'],
    'E': ['N'],
    'G': ['I', 'P'],
    'I': ['G', 'Q', 'W'],
    'K': ['C', 'M', 'T', 'W'],
    'M': ['D', 'F', 'K', 'N'],
    'N': ['E', 'M'],
    'P': ['G', 'Q'],
    'Q': ['I', 'P', 'R'],
    'R': ['Q', 'T'],
    'T': ['K', 'R'],
    'W': ['I', 'K'],
    'F': ['M'],
}

START = 'I'
GOAL = 'F'
WALL_NODE = 'W'
WALL_COST = 30
DEFAULT_COST = 1


def step_cost(node):
    """Costo de ENTRAR a 'node' (independiente del nodo de origen)."""
    return WALL_COST if node == WALL_NODE else DEFAULT_COST


def heuristic(node, goal=GOAL):
    """Distancia de Manhattan entre 'node' y 'goal'."""
    r1, c1 = COORDS[node]
    r2, c2 = COORDS[goal]
    return abs(r1 - r2) + abs(c1 - c2)


def neighbors(node):
    """Vecinos válidos, ordenados alfabéticamente para el desempate."""
    return sorted(GRAPH[node])


def reconstruct(parent, goal):
    """Reconstruye el camino start->goal y su costo acumulado."""
    if goal not in parent:
        return None, None
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    cost = sum(step_cost(n) for n in path[1:])  # el nodo inicial no "cuesta" entrar
    return path, cost


# ---------------------------------------------------------------
# 2. DFS (Depth-First Search) - búsqueda en árbol/grafo recursiva
# ---------------------------------------------------------------

def dfs(start=START, goal=GOAL, verbose=True):
    if verbose:
        print("\n================  DFS (Profundidad Primero)  ================")
    visited = set()
    parent = {start: None}
    order = []
    goal_found = {'flag': False}

    def visit(node):
        visited.add(node)
        order.append(node)
        if verbose:
            print(f"Visitando: {node:2s} | Visitados: {sorted(visited)}")
        if node == goal:
            goal_found['flag'] = True
            return
        for n in neighbors(node):          # orden alfabético = desempate
            if goal_found['flag']:
                return
            if n not in visited:
                parent[n] = node
                visit(n)

    visit(start)
    path, cost = reconstruct(parent, goal)
    if verbose:
        print(f"\nOrden de expansión : {order}")
        print(f"Camino final        : {' -> '.join(path)}")
        print(f"Costo total         : {cost}")
    return order, path, cost


# ---------------------------------------------------------------
# 3. GREEDY BEST-FIRST SEARCH  (f(n) = h(n))
# ---------------------------------------------------------------

def greedy_search(start=START, goal=GOAL, verbose=True):
    if verbose:
        print("\n============  GREEDY BEST-FIRST SEARCH  (f = h)  =============")
    frontier = []
    heapq.heappush(frontier, (heuristic(start), start))   # (h, nombre) -> desempate alfabético automático
    in_frontier = {start}
    closed = set()
    parent = {start: None}
    order = []

    while frontier:
        h, node = heapq.heappop(frontier)
        in_frontier.discard(node)
        if node in closed:
            continue
        closed.add(node)
        order.append(node)
        if verbose:
            print(f"Expandiendo: {node:2s} (h={h:2d}) | Frontera: {sorted(in_frontier)} | Cerrados: {sorted(closed)}")

        if node == goal:
            break

        for n in neighbors(node):
            if n not in closed and n not in in_frontier:
                parent[n] = node
                heapq.heappush(frontier, (heuristic(n), n))
                in_frontier.add(n)

    path, cost = reconstruct(parent, goal)
    if verbose:
        print(f"\nOrden de expansión : {order}")
        print(f"Camino final        : {' -> '.join(path)}")
        print(f"Costo total         : {cost}")
    return order, path, cost


# ---------------------------------------------------------------
# 4. A* SEARCH  (f(n) = g(n) + h(n))
# ---------------------------------------------------------------

def a_star_search(start=START, goal=GOAL, verbose=True):
    if verbose:
        print("\n=====================  A*  (f = g + h)  =======================")
        print(f"{'Nodo':4} {'g(n)':>5} {'h(n)':>5} {'f(n)':>5}")

    frontier = []
    g_score = {start: 0}
    heapq.heappush(frontier, (g_score[start] + heuristic(start), start))
    parent = {start: None}
    closed = set()
    order = []

    while frontier:
        f, node = heapq.heappop(frontier)
        if node in closed:
            continue
        closed.add(node)
        order.append(node)
        if verbose:
            print(f"{node:4} {g_score[node]:5d} {heuristic(node):5d} {f:5d}   <- expandido | Cerrados: {sorted(closed)}")

        if node == goal:
            break

        for n in neighbors(node):
            tentative_g = g_score[node] + step_cost(n)
            if tentative_g < g_score.get(n, float('inf')):
                g_score[n] = tentative_g
                parent[n] = node
                heapq.heappush(frontier, (tentative_g + heuristic(n), n))
                closed.discard(n)   # reabrir si se halló un camino mejor

    path, cost = reconstruct(parent, goal)
    if verbose:
        print(f"\nOrden de expansión : {order}")
        print(f"Camino final        : {' -> '.join(path)}")
        print(f"Costo total         : {cost}")
    return order, path, cost


if __name__ == "__main__":
    dfs()
    greedy_search()
    a_star_search()
