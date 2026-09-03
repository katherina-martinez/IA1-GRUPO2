class MotorResolucion:
    def __init__(self):
        # El conjunto de cláusulas iniciales
        self.clausulas = set()
    
    def agregar_hecho(self, hecho):
        # Un hecho es una cláusula de un solo elemento (ej. {'d'})
        self.clausulas.add(frozenset([hecho]))
        
    def agregar_regla(self, antecedentes, consecuente):
        """
        Convierte una regla en formato A ^ B -> C a la cláusula ~A v ~B v C
        """
        clausula = set()
        # Negar todos los antecedentes
        for ant in antecedentes:
            if ant.startswith('~'):
                clausula.add(ant[1:]) # Si ya es negativo, se vuelve positivo
            else:
                clausula.add(f"~{ant}")
                
        # Agregar el consecuente
        clausula.add(consecuente)
        self.clausulas.add(frozenset(clausula))

    def _es_complementario(self, lit1, lit2):
        # Retorna True si uno es la negación exacta del otro
        return lit1 == f"~{lit2}" or lit2 == f"~{lit1}"

    def _es_tautologia(self, clausula):
        # Una cláusula es tautología si contiene a la vez 'x' y '~x'
        for lit in clausula:
            if lit.startswith('~') and lit[1:] in clausula:
                return True
            if not lit.startswith('~') and f"~{lit}" in clausula:
                return True
        return False

    def formato(self, clausula):
        if len(clausula) == 0:
            return "[] (Cláusula vacía / Absurdo)"
        return f"{{{', '.join(sorted(list(clausula)))}}}"

    def resolver_par(self, c1, c2):
        resolventes = []
        for l1 in c1:
            for l2 in c2:
                if self._es_complementario(l1, l2):
                    # Unimos c1 y c2, quitando los literales cancelados (l1 y l2)
                    nueva = set(c1) | set(c2)
                    nueva.remove(l1)
                    nueva.remove(l2)
                    
                    if not self._es_tautologia(nueva):
                        resolventes.append(frozenset(nueva))
        return resolventes

    def ejecutar(self):
        print("--- INICIANDO MOTOR DE INFERENCIA POR CONTRADICCIÓN (RESOLUCIÓN) ---")
        print("\nCláusulas Iniciales (Hechos y Reglas):")
        for c in self.clausulas:
            print(self.formato(c))
            
        print("\n--- INICIANDO PROCESO DE RAZONAMIENTO ---")
        clausulas_procesadas = set(self.clausulas)
        pares_revisados = set()
        
        hubo_cambios = True
        paso = 1
        
        while hubo_cambios:
            hubo_cambios = False
            lista_clausulas = list(clausulas_procesadas)
            nuevas_esta_ronda = set()
            
            for i in range(len(lista_clausulas)):
                for j in range(i + 1, len(lista_clausulas)):
                    c1 = lista_clausulas[i]
                    c2 = lista_clausulas[j]
                    par_id = frozenset([c1, c2])
                    
                    if par_id not in pares_revisados:
                        pares_revisados.add(par_id)
                        nuevas = self.resolver_par(c1, c2)
                        
                        for n in nuevas:
                            if n not in clausulas_procesadas and n not in nuevas_esta_ronda:
                                nuevas_esta_ronda.add(n)
                                print(f"Paso {paso}: Al resolver {self.formato(c1)} y {self.formato(c2)} se infiere -> {self.formato(n)}")
                                paso += 1
                                
                                # DETECCIÓN DE CONTRADICCIÓN
                                if len(n) == 0:
                                    print("\n¡CONTRADICCIÓN ENCONTRADA! Se ha derivado la cláusula vacía.")
                                    print("RESULTADO FINAL: El conjunto es INCONSISTENTE.")
                                    return False
            
            if nuevas_esta_ronda:
                clausulas_procesadas.update(nuevas_esta_ronda)
                hubo_cambios = True
                
        print("\nEl motor ha agotado todas las combinaciones posibles.")
        print("No se pudo derivar la cláusula vacía.")
        print("RESULTADO FINAL: El conjunto es CONSISTENTE.")
        return True

# --- Ejecución de tu actividad ---
if __name__ == "__main__":
    motor = MotorResolucion()
    
    # Hechos iniciales
    motor.agregar_hecho('d')  # R5
    motor.agregar_hecho('e')  # R6
    
    # Reglas
    motor.agregar_regla(['b', 'c'], 'a')  # R1
    motor.agregar_regla(['d', 'e'], 'b')  # R2
    motor.agregar_regla(['g', 'e'], 'b')  # R3
    motor.agregar_regla(['e'], 'c')       # R4
    motor.agregar_regla(['a', 'g'], 'f')  # R7
    
    motor.ejecutar()