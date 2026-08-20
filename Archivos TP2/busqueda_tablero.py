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
#constantes globales
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
    """Ejecuta busqueda DFS desde START hasta GOAL"""
    if verbose:
        print("\n================  DFS (Profundidad Primero)  ================")
    visited = set()  #para no repetir nodos ya visitados
    parent = {start: None}  #se guardan los padres de cada nodo en un diccionario
                            #el nodo inicial no tiene padre (:c)
    order = []
    goal_found = {'flag': False}

    def visit(node):
            """visita un nodo en DFS y luego lo agrega al conjunto de visitados"""
        visited.add(node)
        order.append(node)
        if verbose:
            print(f"Visitando: {node:2s} | Visitados: {sorted(visited)}")
        if node == goal:
            goal_found['flag'] = True
            return
        for n in neighbors(node):   # orden alfabético = desempate
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
    """Ejecuta búsqueda Greedy, siempre expande el nodo con menor heurística"""
    if verbose:
        print("\n============  GREEDY BEST-FIRST SEARCH  (f = h)  =============")
    frontier = [] #nodo no expandido
    heapq.heappush(frontier, (heuristic(start), start))   # (h, nombre) -> desempate alfabético automático
    in_frontier = {start}  #para identificar nodos ya expandido y evitar duplicados
    closed = set()  #nodo ya procesado
    parent = {start: None}
    order = []

    while frontier: #mientras haya nodos en la frontera se sigue buscando
        h, node = heapq.heappop(frontier) #se busca el nodo con menor h(n)
        in_frontier.discard(node) #cuando el nodo se expande ya no está en la frontera
        if node in closed:
            continue
        closed.add(node)
        order.append(node)
        if verbose:
            print(f"Expandiendo: {node:2s} (h={h:2d}) | Frontera: {sorted(in_frontier)} | Cerrados: {sorted(closed)}")

        if node == goal: #si llegamos al nodo objetivo
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
    """Ejecuta busqueda A*, sumando el costo de g(n) con h(n) para elegir el próximo nodo a expandir"""
    if verbose:
        print("\n=====================  A*  (f = g + h)  =======================")
        print(f"{'Nodo':4} {'g(n)':>5} {'h(n)':>5} {'f(n)':>5}")

    frontier = []
    g_score = {start: 0} #diccionario con el costo real desde el inicio hasta cada nodo, start es cero porque ya estoy ahí
    heapq.heappush(frontier, (g_score[start] + heuristic(start), start)) #
    parent = {start: None} #diccionario de padres
    closed = set() #nodos ya expandidos
    order = [] #orden de expansión de los nodos

    while frontier: #mientras haya nodos candidatos
        f, node = heapq.heappop(frontier) #extrae nodo con menor f(n)
        if node in closed: #verificación para evitar procesamiento doble
            continue
        closed.add(node)
        order.append(node)
        if verbose:
            print(f"{node:4} {g_score[node]:5d} {heuristic(node):5d} {f:5d}   <- expandido | Cerrados: {sorted(closed)}")

        if node == goal: #si llegamos al nodo objetivo terminamos
            break

        for n in neighbors(node): #recorremos los nodos vecinos del actual en orden alfabético
            tentative_g = g_score[node] + step_cost(n) #costo de ir a un nodo vecino=costo para llegar al nodo actual + costo de entrar al nodo vecino
            if tentative_g < g_score.get(n, float('inf')): #comparo el nuevo camino hallado con el conocido anteriormente
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
