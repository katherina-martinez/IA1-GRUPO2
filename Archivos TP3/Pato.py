class Node:
    """Representa una pieza (nodo) del rompecabezas con su dominio y vecinos."""
    def __init__(self, node_id: int, neighbors: list[int], num_colors: int = 7):
        self.id = node_id
        self.neighbors = neighbors
        self.domain = list(range(1, num_colors + 1))
        self.value = None

    def unassigned_neighbors_count(self, nodes_dict: dict) -> int:
        """Devuelve la cantidad de vecinos que aún no tienen un color asignado."""
        return sum(1 for neighbor_id in self.neighbors if nodes_dict[neighbor_id].value is None)


class MapColoringCSP:
    """Resuelve el problema de coloreado aplicando MRV, Grado y Forward Checking."""
    def __init__(self, graph_adjacency: dict[int, list[int]], num_colors: int = 7):
        self.nodes = {
            node_id: Node(node_id, neighbors, num_colors)
            for node_id, neighbors in graph_adjacency.items()
        }
        self.history = []

    def select_variable_mrv(self) -> Node:
        """Selecciona el siguiente nodo usando MRV y Grado como desempate."""
        unassigned = [node for node in self.nodes.values() if node.value is None]
        if not unassigned:
            return None

        # 1. Criterio MRV: Menor tamaño de dominio restante
        min_domain_len = min(len(node.domain) for node in unassigned)
        mrv_candidates = [node for node in unassigned if len(node.domain) == min_domain_len]

        if len(mrv_candidates) == 1:
            return mrv_candidates[0]

        # 2. Desempate por Heurística del Grado: Mayor cantidad de vecinos no asignados
        max_degree = max(node.unassigned_neighbors_count(self.nodes) for node in mrv_candidates)
        degree_candidates = [
            node for node in mrv_candidates 
            if node.unassigned_neighbors_count(self.nodes) == max_degree
        ]

        # 3. Desempate por número de ID
        degree_candidates.sort(key=lambda n: n.id)
        return degree_candidates[0]

    def forward_check(self, current_node: Node, assigned_color: int) -> list[int]:
        """Elimina el color asignado de los dominios de los vecinos no asignados."""
        affected_neighbors = []
        for neighbor_id in current_node.neighbors:
            neighbor = self.nodes[neighbor_id]
            if neighbor.value is None and assigned_color in neighbor.domain:
                neighbor.domain.remove(assigned_color)
                affected_neighbors.append(neighbor_id)
        return affected_neighbors

    def solve(self):
        """Ejecuta el proceso paso a paso registrando la traza de ejecución."""
        # Registrar estado inicial (Paso 0)
        self.history.append({
            "step": 0,
            "node_id": 0,
            "color": None,
            "affected_neighbors": [],
            "domains_snapshot": {
                n_id: list(n.domain) for n_id, n in self.nodes.items()
            }
        })

        step = 1
        while True:
            node = self.select_variable_mrv()
            if node is None:
                break

            # Toma el primer valor disponible en el dominio restante
            assigned_color = node.domain[0]
            node.value = assigned_color

            # Propagación Forward Checking
            affected = self.forward_check(node, assigned_color)

            # Registro de auditoría
            self.history.append({
                "step": step,
                "node_id": node.id,
                "color": assigned_color,
                "affected_neighbors": affected,
                "domains_snapshot": {
                    n_id: list(n.domain) if n.value is None else n.value 
                    for n_id, n in self.nodes.items()
                }
            })
            step += 1

    def print_execution_trace(self):
        """Muestra la justificación del orden de selección y efectos paso a paso."""
        print("### Traza de Ejecución\n")
        print(f"{'Paso':<5} | {'Nodo':<5} | {'Color':<6} | {'Efecto Forward Checking (Vecinos Podados)'}")
        print("-" * 65)
        for log in self.history:
            if log['step'] == 0:
                continue
            affected_str = ", ".join(map(str, log['affected_neighbors'])) or "Ninguno"
            print(f"{log['step']:<5} | {log['node_id']:<5} | {log['color']:<6} | Eliminado color {log['color']} de: [{affected_str}]")
        print("\n")

    def print_markdown_domain_table(self):
        """Imprime el historial de dominios formateado como una tabla Markdown."""
        print("### Tabla de Dominios (Formato Markdown)\n")
        
        node_ids = sorted(self.nodes.keys())
        
        # Construcción de encabezados
        headers = ["Nodo seleccionado"] + [str(nid) for nid in node_ids]
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join([":---:"] * len(headers)) + "|")
        
        # Construcción de filas
        for log in self.history:
            selected = str(log["node_id"])
            row = [selected]
            
            for nid in node_ids:
                estado = log["domains_snapshot"][nid]
                if isinstance(estado, list):
                    # Es un dominio sin asignar, unimos con guiones
                    row.append("-".join(map(str, estado)))
                else:
                    # Es un valor ya asignado, lo resaltamos en negrita
                    row.append(f"**{estado}**")
            
            print("| " + " | ".join(row) + " |")


# --- EJECUCIÓN DEL MODELO ---
if __name__ == "__main__":
    # Grafo de adjacencia
    puzzle_graph = {
        1: [2],
        2: [1, 3, 4],
        3: [2, 4, 9],
        4: [2, 3, 5, 7, 9],
        5: [4, 6, 7],
        6: [5, 7, 10, 11],
        7: [4, 5, 6, 10],
        8: [9, 13],
        9: [3, 4, 8, 10, 13, 14, 15],
        10: [6, 7, 9, 11, 15, 16, 17],
        11: [6, 10, 12, 17],
        12: [11, 17],
        13: [8, 9, 14],
        14: [9, 13, 15],
        15: [9, 10, 14, 16],
        16: [10, 15, 17],
        17: [10, 11, 12, 16]
    }

    solver = MapColoringCSP(puzzle_graph, num_colors=7)
    solver.solve()
    
    # Imprimir los resultados
    solver.print_execution_trace()
    solver.print_markdown_domain_table()