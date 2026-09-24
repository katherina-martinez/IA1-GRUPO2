import pygame
import sys
import math
import random

# --- CONFIGURACIÓN GENERAL ---
WIDTH, HEIGHT = 800, 600
FPS = 60
MAX_SCORE = 5

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
BLUE = (100, 150, 255)

# --- MOTOR DE LÓGICA DIFUSA ---
class FuzzyMF:
    @staticmethod
    def trimf(x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        if a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        if b < x < c:
            return (c - x) / (c - b) if c != b else 1.0
        return 0.0

    @staticmethod
    def trapmf(x, a, b, c, d):
        if x <= a or x >= d:
            return 0.0
        if a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        if b < x <= c:
            return 1.0
        if c < x < d:
            return (d - x) / (d - c) if d != c else 1.0
        return 0.0

class FuzzyController:
    def __init__(self, max_speed):
        self.max_speed = max_speed
        
    def calculate_speed(self, dx, dy, ball_speed):
        raise NotImplementedError()

    def defuzzify(self, output_sets):
        numerador = 0.0
        denominador = 0.0
        steps = 40
        step_size = (self.max_speed * 2) / steps
        
        for i in range(steps + 1):
            v = -self.max_speed + i * step_size
            max_mu = 0.0
            for set_name, activation in output_sets.items():
                mu_val = self.get_output_mf(v, set_name)
                mu_cut = min(mu_val, activation)
                max_mu = max(max_mu, mu_cut)
                
            numerador += v * max_mu
            denominador += max_mu
            
        if denominador == 0: return 0
        return numerador / denominador

    def get_output_mf(self, v, set_name):
        ms = self.max_speed
        if set_name == 'fast_up': return FuzzyMF.trimf(v, -ms, -ms, -ms/2)
        elif set_name == 'slow_up': return FuzzyMF.trimf(v, -ms, -ms/2, 0)
        elif set_name == 'stop': return FuzzyMF.trimf(v, -ms/4, 0, ms/4)
        elif set_name == 'slow_down': return FuzzyMF.trimf(v, 0, ms/2, ms)
        elif set_name == 'fast_down': return FuzzyMF.trimf(v, ms/2, ms, ms)
        return 0.0

# --- NUEVO AGENTE CONFIGURABLE POR DIFICULTAD ---
class FuzzyAgentConfigurable(FuzzyController):
    def __init__(self, max_speed, difficulty="MEDIUM"):
        super().__init__(max_speed)
        
        # Ajustamos los límites de pertenencia de "Cerca" en base a la dificultad.
        if difficulty == "EASY":
            self.fact_x = 0.25  # Cerca horizontal es solo 1/4
            self.fact_y = 0.75  # Cerca vertical es 3/4 (Se moverá lento la mayoría del tiempo)
        elif difficulty == "HARD":
            self.fact_x = 0.75  # Cerca horizontal es 3/4 (Reacciona rápido antes)
            self.fact_y = 0.25  # Cerca vertical es 1/4 (Se moverá rápido la mayoría del tiempo)
        else: # MEDIUM
            self.fact_x = 0.50
            self.fact_y = 0.50

    def calculate_speed(self, dx, dy, ball_speed):
        dx = abs(dx)
        
        limit_x = WIDTH * self.fact_x
        limit_y = HEIGHT * self.fact_y
        
        # Fuzzificación Horizontal
        dx_near = FuzzyMF.trimf(dx, -1, 0, limit_x)
        dx_far = FuzzyMF.trimf(dx, limit_x / 2, WIDTH, WIDTH * 2)
        
        # Fuzzificación Vertical
        dy_far_up = FuzzyMF.trimf(dy, -HEIGHT*2, -HEIGHT, -limit_y/2)
        dy_near_up = FuzzyMF.trimf(dy, -limit_y, -limit_y/2, 0)
        dy_center = FuzzyMF.trimf(dy, -limit_y/4, 0, limit_y/4)
        dy_near_down = FuzzyMF.trimf(dy, 0, limit_y/2, limit_y)
        dy_far_down = FuzzyMF.trimf(dy, limit_y/2, HEIGHT, HEIGHT*2)

        # Reglas Lógicas (Evaluadas con Mínimo para AND y Máximo para OR)
        rule_stop = dy_center
        
        # 1. RÁPIDO: Si horizontal es Cerca Y vertical es Lejos
        rule_fast_up = min(dx_near, dy_far_up)
        rule_fast_down = min(dx_near, dy_far_down)
        
        # 2. LENTO: Si vertical es Cerca (ajuste fino) O (horizontal es Lejos y vertical es Lejos)
        rule_slow_up = max(dy_near_up, min(dx_far, dy_far_up))
        rule_slow_down = max(dy_near_down, min(dx_far, dy_far_down))
        
        output_sets = {
            'stop': rule_stop,
            'fast_up': rule_fast_up,
            'fast_down': rule_fast_down,
            'slow_up': rule_slow_up,
            'slow_down': rule_slow_down
        }
        
        return self.defuzzify(output_sets)


class FuzzyAgentAdvanced(FuzzyController):
    """Agente modo observar (IA Avanzada)"""
    def calculate_speed(self, dx, dy, ball_speed):
        dx = abs(dx)
        speed_factor = min(ball_speed / 5, 2.5) 
        near_limit = (WIDTH/2) * speed_factor

        dx_near = FuzzyMF.trapmf(dx, 0, 0, near_limit/2, near_limit)
        dx_far = FuzzyMF.trapmf(dx, near_limit/2, near_limit, WIDTH, WIDTH)
        
        dy_far_up = FuzzyMF.trapmf(dy, -HEIGHT, -HEIGHT, -HEIGHT/3, -HEIGHT/6)
        dy_near_up = FuzzyMF.trapmf(dy, -HEIGHT/2, -HEIGHT/4, -HEIGHT/16, 0)
        dy_center = FuzzyMF.trapmf(dy, -HEIGHT/10, -HEIGHT/20, HEIGHT/20, HEIGHT/10)
        dy_near_down = FuzzyMF.trapmf(dy, 0, HEIGHT/16, HEIGHT/4, HEIGHT/2)
        dy_far_down = FuzzyMF.trapmf(dy, HEIGHT/6, HEIGHT/3, HEIGHT, HEIGHT)

        rule_stop = dy_center
        rule_fast_up = max(dy_far_up, min(dx_near, dy_near_up))
        rule_fast_down = max(dy_far_down, min(dx_near, dy_near_down))
        rule_slow_up = min(dx_far, dy_near_up)
        rule_slow_down = min(dx_far, dy_near_down)
        
        output_sets = {
            'stop': rule_stop, 'fast_up': rule_fast_up, 'fast_down': rule_fast_down,
            'slow_up': rule_slow_up, 'slow_down': rule_slow_down
        }
        return self.defuzzify(output_sets)

# --- CLASES DEL JUEGO ---
class Paddle:
    def __init__(self, x, y, is_player=False, fuzzy_agent=None):
        self.rect = pygame.Rect(x, y, 15, 100)
        self.is_player = is_player
        self.speed = 0
        self.max_speed = 15
        self.fuzzy_agent = fuzzy_agent

    def update(self, ball):
        if self.is_player:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: self.speed = -self.max_speed
            elif keys[pygame.K_DOWN]: self.speed = self.max_speed
            else: self.speed = 0
        else:
            dx = ball.rect.centerx - self.rect.centerx
            dy = ball.rect.centery - self.rect.centery
            ball_speed = math.hypot(ball.vx, ball.vy)
            self.speed = self.fuzzy_agent.calculate_speed(dx, dy, ball_speed)
        
        self.rect.y += self.speed
        if self.rect.top < 0: self.rect.top = 0
        if self.rect.bottom > HEIGHT: self.rect.bottom = HEIGHT

    def draw(self, surface):
        pygame.draw.rect(surface, WHITE, self.rect)


class Ball:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH//2 - 7, HEIGHT//2 - 7, 15, 15)
        self.base_speed = 5
        self.reset()

    def reset(self):
        self.rect.center = (WIDTH//2, HEIGHT//2)
        self.vx = self.base_speed * random.choice([1, -1])
        self.vy = self.base_speed * random.choice([1, -1]) * random.uniform(0.5, 1.0)

    def increase_speed(self):
        self.vx *= 1.1
        self.vy *= 1.1
        max_ball_speed = 15
        if self.vx > max_ball_speed: self.vx = max_ball_speed
        if self.vx < -max_ball_speed: self.vx = -max_ball_speed
        if self.vy > max_ball_speed: self.vy = max_ball_speed
        if self.vy < -max_ball_speed: self.vy = -max_ball_speed

    def update(self, left_paddle, right_paddle):
        self.rect.x += int(self.vx)
        self.rect.y += int(self.vy)

        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.vy *= -1
            self.increase_speed()
            if self.rect.top <= 0: self.rect.top = 1
            if self.rect.bottom >= HEIGHT: self.rect.bottom = HEIGHT - 1

        if self.rect.colliderect(left_paddle.rect) and self.vx < 0:
            self.vx *= -1
            self.increase_speed()
            self.rect.left = left_paddle.rect.right
        
        if self.rect.colliderect(right_paddle.rect) and self.vx > 0:
            self.vx *= -1
            self.increase_speed()
            self.rect.right = right_paddle.rect.left

        if self.rect.left <= 0: return "RIGHT"
        elif self.rect.right >= WIDTH: return "LEFT"
        return None

    def draw(self, surface):
        pygame.draw.ellipse(surface, WHITE, self.rect)


class PongGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Pong - Fuzzy Logic Controller")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 36, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 24)
        
    def start_game(self, mode, difficulty="MEDIUM"):
        self.score_left = 0
        self.score_right = 0
        self.ball = Ball()
        
        if mode == "PLAY":
            self.paddle_left = Paddle(30, HEIGHT//2 - 50, is_player=True)
            self.paddle_right = Paddle(WIDTH - 45, HEIGHT//2 - 50, is_player=False, 
                                     fuzzy_agent=FuzzyAgentConfigurable(9, difficulty))
        else: # "OBSERVE"
            self.paddle_left = Paddle(30, HEIGHT//2 - 50, is_player=False, 
                                    fuzzy_agent=FuzzyAgentConfigurable(9, "MEDIUM"))
            self.paddle_right = Paddle(WIDTH - 45, HEIGHT//2 - 50, is_player=False, 
                                     fuzzy_agent=FuzzyAgentAdvanced(9))

        self.run_loop(mode)

    def run_loop(self, mode):
        running = True
        while running:
            self.screen.fill(BLACK)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False 
            
            self.paddle_left.update(self.ball)
            self.paddle_right.update(self.ball)
            goal = self.ball.update(self.paddle_left, self.paddle_right)
            
            if goal == "LEFT":
                self.score_left += 1
                self.ball.reset()
            elif goal == "RIGHT":
                self.score_right += 1
                self.ball.reset()
                
            if self.score_left >= MAX_SCORE or self.score_right >= MAX_SCORE:
                self.show_winner("Jugador (Izquierda)" if self.score_left >= MAX_SCORE else "IA (Derecha)")
                running = False

            pygame.draw.aaline(self.screen, GRAY, (WIDTH//2, 0), (WIDTH//2, HEIGHT))
            self.paddle_left.draw(self.screen)
            self.paddle_right.draw(self.screen)
            self.ball.draw(self.screen)
            
            score_text_L = self.font.render(str(self.score_left), True, WHITE)
            score_text_R = self.font.render(str(self.score_right), True, WHITE)
            self.screen.blit(score_text_L, (WIDTH//4, 20))
            self.screen.blit(score_text_R, (WIDTH*3//4, 20))

            pygame.display.flip()
            self.clock.tick(FPS)

    def show_winner(self, winner):
        waiting = True
        while waiting:
            self.screen.fill(BLACK)
            text = self.font.render(f"¡Ganador: {winner}!", True, WHITE)
            subtext = self.small_font.render("Presiona ESPACIO para volver al menú", True, GRAY)
            
            self.screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - 50))
            self.screen.blit(subtext, (WIDTH//2 - subtext.get_width()//2, HEIGHT//2 + 20))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting = False

    def difficulty_menu(self):
        waiting = True
        while waiting:
            self.screen.fill(BLACK)
            title = self.font.render("SELECCIONA LA DIFICULTAD", True, BLUE)
            op1 = self.small_font.render("1. FÁCIL (1/4 Distancia Cerca)", True, WHITE)
            op2 = self.small_font.render("2. MEDIO (1/2 Distancia Cerca)", True, WHITE)
            op3 = self.small_font.render("3. DIFÍCIL (3/4 Distancia Cerca)", True, WHITE)
            
            self.screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//4))
            self.screen.blit(op1, (WIDTH//2 - op1.get_width()//2, HEIGHT//2 - 20))
            self.screen.blit(op2, (WIDTH//2 - op2.get_width()//2, HEIGHT//2 + 20))
            self.screen.blit(op3, (WIDTH//2 - op3.get_width()//2, HEIGHT//2 + 60))
            
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.start_game("PLAY", "EASY")
                        waiting = False
                    elif event.key == pygame.K_2:
                        self.start_game("PLAY", "MEDIUM")
                        waiting = False
                    elif event.key == pygame.K_3:
                        self.start_game("PLAY", "HARD")
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        waiting = False

    def menu(self):
        while True:
            self.screen.fill(BLACK)
            title = self.font.render("PONG - CONTROLADOR DIFUSO", True, WHITE)
            op1 = self.small_font.render("1. MODO JUGAR (Tú vs IA)", True, WHITE)
            op2 = self.small_font.render("2. MODO OBSERVAR (IA Estándar vs IA Avanzada)", True, WHITE)
            inst = self.small_font.render("Presiona 1 o 2 (ESC en juego para volver)", True, GRAY)

            self.screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//4))
            self.screen.blit(op1, (WIDTH//2 - op1.get_width()//2, HEIGHT//2))
            self.screen.blit(op2, (WIDTH//2 - op2.get_width()//2, HEIGHT//2 + 50))
            self.screen.blit(inst, (WIDTH//2 - inst.get_width()//2, HEIGHT - 50))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.difficulty_menu()
                    elif event.key == pygame.K_2:
                        self.start_game("OBSERVE")

if __name__ == "__main__":
    game = PongGame()
    game.menu()