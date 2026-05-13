import pygame
import cv2
import numpy as np
import random
import time
import math
from collections import deque, Counter
from ultralytics import YOLO

# Inicializar Pygame
pygame.init()

# Configuración de la ventana
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("🤟 Aprende Lenguaje de Señas - Juego Interactivo")

# Colores modernos
COLORS = {
    'background': (20, 25, 40),
    'card_bg': (45, 55, 80),
    'primary': (100, 200, 255),
    'secondary': (150, 100, 255),
    'success': (50, 200, 100),
    'error': (255, 100, 100),
    'warning': (255, 200, 50),
    'white': (255, 255, 255),
    'light_gray': (200, 200, 200),
    'dark_gray': (100, 100, 100),
    'accent': (255, 150, 50)
}

# Fuentes
font_large = pygame.font.Font(None, 48)
font_medium = pygame.font.Font(None, 36)
font_small = pygame.font.Font(None, 24)
font_title = pygame.font.Font(None, 64)

class ParticleEffect:
    def __init__(self):
        self.particles = []
    
    def add_success_particles(self, x, y):
        for _ in range(20):
            self.particles.append({
                'x': x,
                'y': y,
                'vx': random.uniform(-5, 5),
                'vy': random.uniform(-8, -2),
                'life': 60,
                'color': random.choice([COLORS['success'], COLORS['primary'], COLORS['accent']]),
                'size': random.uniform(3, 8)
            })
    
    def add_error_particles(self, x, y):
        for _ in range(15):
            self.particles.append({
                'x': x,
                'y': y,
                'vx': random.uniform(-3, 3),
                'vy': random.uniform(-5, -1),
                'life': 40,
                'color': COLORS['error'],
                'size': random.uniform(2, 5)
            })
    
    def update(self):
        for particle in self.particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.2  # gravity
            particle['life'] -= 1
            particle['size'] *= 0.98
            if particle['life'] <= 0 or particle['size'] < 1:
                self.particles.remove(particle)
    
    def draw(self, surface):
        for particle in self.particles:
            alpha = max(0, particle['life'] / 60.0 * 255)
            color = (*particle['color'], int(alpha))
            
            # Crear superficie temporal para alpha blending
            temp_surface = pygame.Surface((int(particle['size'] * 2), int(particle['size'] * 2)))
            temp_surface.set_alpha(int(alpha))
            pygame.draw.circle(temp_surface, particle['color'], 
                             (int(particle['size']), int(particle['size'])), 
                             int(particle['size']))
            surface.blit(temp_surface, (particle['x'] - particle['size'], particle['y'] - particle['size']))

def draw_rounded_rect(surface, color, rect, radius=20):
    """Dibuja un rectángulo con esquinas redondeadas"""
    x, y, w, h = rect
    pygame.draw.rect(surface, color, (x + radius, y, w - 2 * radius, h))
    pygame.draw.rect(surface, color, (x, y + radius, w, h - 2 * radius))
    pygame.draw.circle(surface, color, (x + radius, y + radius), radius)
    pygame.draw.circle(surface, color, (x + w - radius, y + radius), radius)
    pygame.draw.circle(surface, color, (x + radius, y + h - radius), radius)
    pygame.draw.circle(surface, color, (x + w - radius, y + h - radius), radius)

def draw_gradient_rect(surface, color1, color2, rect):
    """Dibuja un rectángulo con gradiente"""
    x, y, w, h = rect
    for i in range(h):
        ratio = i / h
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        pygame.draw.line(surface, (r, g, b), (x, y + i), (x + w, y + i))

def draw_progress_bar(surface, x, y, width, height, progress, color):
    """Dibuja una barra de progreso moderna"""
    # Fondo
    draw_rounded_rect(surface, COLORS['dark_gray'], (x, y, width, height), 10)
    
    # Progreso
    if progress > 0:
        progress_width = int(width * progress)
        draw_rounded_rect(surface, color, (x, y, progress_width, height), 10)
        
        # Efecto de brillo
        highlight_height = height // 3
        draw_rounded_rect(surface, tuple(min(255, c + 50) for c in color), 
                         (x, y, progress_width, highlight_height), 10)

def opencv_to_pygame(cv_image):
    """Convierte imagen de OpenCV a formato Pygame"""
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    cv_image = np.rot90(cv_image)
    cv_image = pygame.surfarray.make_surface(cv_image)
    return cv_image

class SignLanguageGame:
    def __init__(self):
        # Cargar modelo YOLO
        self.model = YOLO('/home/srm/Documents/hands-pose-estimation/runs/detect/train5/weights/best.pt')
        print("Clases del modelo:", self.model.names)
        
        # Variables del juego
        self.vocales = ['A', 'E', 'I', 'O', 'U']
        self.puntuacion = 0
        self.mejor_puntuacion = 0
        self.tiempo_preparacion = 5.0
        self.vocal_actual = random.choice(self.vocales)
        self.mensaje_feedback = ""
        self.color_feedback = COLORS['success']
        self.tiempo_inicio = time.time()
        self.evaluado = False
        
        # Buffer para detecciones
        self.detections_deque = deque(maxlen=15)
        self.feedback_start = None
        self.feedback_duracion = 3.0
        self.CONF_THRESHOLD = 0.40
        
        # Cámara
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Efectos visuales
        self.particles = ParticleEffect()
        self.pulse_time = 0
        self.shake_intensity = 0
        self.shake_duration = 0
        
        # Estado del juego
        self.game_state = "playing"  # "playing", "paused", "menu"
        
        # Clock para FPS
        self.clock = pygame.time.Clock()
        
    def add_screen_shake(self, intensity=10, duration=500):
        """Añade efecto de vibración de pantalla"""
        self.shake_intensity = intensity
        self.shake_duration = duration
        
    def get_screen_offset(self):
        """Calcula el offset para el efecto de vibración"""
        if self.shake_duration > 0:
            self.shake_duration -= self.clock.get_time()
            offset_x = random.randint(-self.shake_intensity, self.shake_intensity)
            offset_y = random.randint(-self.shake_intensity, self.shake_intensity)
            return offset_x, offset_y
        return 0, 0
    
    def process_frame(self):
        """Procesa un frame de la cámara"""
        ret, frame = self.cap.read()
        if not ret:
            return None, None, 0.0, None
            
        # Inferencia YOLO
        results = self.model(frame)[0]
        
        detected_class = None
        detected_conf = 0.0
        detected_box = None
        
        if hasattr(results, "boxes") and len(results.boxes):
            try:
                data = results.boxes.data.cpu().numpy()
            except:
                data = results.boxes.data.numpy()
                
            best_idx = int(np.argmax(data[:, 4]))
            x1, y1, x2, y2, conf, cls = data[best_idx]
            cls = int(cls)
            detected_conf = float(conf)
            detected_class = self.model.names[cls]
            detected_box = (int(x1), int(y1), int(x2), int(y2))
        
        return frame, detected_class, detected_conf, detected_box
    
    def update_game_logic(self, detected_class, detected_conf):
        """Actualiza la lógica del juego"""
        # Añadir detección al buffer
        if detected_class is not None and detected_conf >= self.CONF_THRESHOLD:
            self.detections_deque.append(detected_class)
        else:
            self.detections_deque.append(None)
        
        # Temporizador
        tiempo_transcurrido = time.time() - self.tiempo_inicio
        tiempo_restante = max(0.0, self.tiempo_preparacion - tiempo_transcurrido)
        
        # Lógica del juego
        if tiempo_restante > 0:
            self.evaluado = False
            self.mensaje_feedback = "¡Prepárate para hacer la seña!"
            self.color_feedback = COLORS['warning']
        else:
            if not self.evaluado:
                # Evaluar por mayoría
                clases_validas = [d for d in self.detections_deque if d is not None]
                if len(clases_validas) == 0:
                    self.mensaje_feedback = "❌ No se detectó la seña claramente"
                    self.color_feedback = COLORS['error']
                    self.particles.add_error_particles(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
                    self.add_screen_shake(5, 300)
                else:
                    counts = Counter(clases_validas)
                    top_class, top_count = counts.most_common(1)[0]
                    
                    if top_class.upper() == self.vocal_actual.upper() and top_count >= 3:
                        self.puntuacion += 1
                        self.mejor_puntuacion = max(self.mejor_puntuacion, self.puntuacion)
                        self.mensaje_feedback = f"🎉 ¡Excelente! Acertaste la {self.vocal_actual}"
                        self.color_feedback = COLORS['success']
                        self.particles.add_success_particles(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
                    else:
                        self.mensaje_feedback = f"❌ Incorrecto. Era la letra {self.vocal_actual}"
                        self.color_feedback = COLORS['error']
                        self.particles.add_error_particles(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
                        self.add_screen_shake(8, 400)
                
                self.evaluado = True
                self.feedback_start = time.time()
        
        # Manejar fin del feedback
        if self.evaluado and self.feedback_start is not None:
            if time.time() - self.feedback_start > self.feedback_duracion:
                self.vocal_actual = random.choice(self.vocales)
                self.tiempo_inicio = time.time()
                self.mensaje_feedback = ""
                self.evaluado = False
                self.feedback_start = None
                self.detections_deque.clear()
        
        return tiempo_restante
    
    def draw_camera_feed(self, surface, frame, detected_box, detected_class, detected_conf):
        """Dibuja el feed de la cámara con detecciones"""
        if frame is None:
            return
            
        # Convertir frame a pygame
        frame_copy = frame.copy()
        
        # Dibujar caja de detección si existe
        if detected_box is not None:
            x1, y1, x2, y2 = detected_box
            color = (0, 255, 0) if detected_conf >= self.CONF_THRESHOLD else (255, 255, 0)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
            
            # Etiqueta con fondo
            label = f"{detected_class} {detected_conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame_copy, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Convertir a pygame y escalar
        pygame_frame = opencv_to_pygame(frame_copy)
        pygame_frame = pygame.transform.scale(pygame_frame, (480, 360))
        
        # Posición de la cámara
        camera_x = WINDOW_WIDTH - 500
        camera_y = 20
        
        # Dibujar marco decorativo
        frame_rect = (camera_x - 10, camera_y - 10, 500, 380)
        draw_rounded_rect(surface, COLORS['card_bg'], frame_rect, 15)
        draw_rounded_rect(surface, COLORS['primary'], (camera_x - 12, camera_y - 12, 504, 384), 15)
        
        surface.blit(pygame_frame, (camera_x, camera_y))
    
    def draw_ui(self, surface, tiempo_restante):
        """Dibuja la interfaz de usuario"""
        offset_x, offset_y = self.get_screen_offset()
        
        # Fondo con gradiente
        draw_gradient_rect(surface, COLORS['background'], 
                          tuple(max(0, c - 20) for c in COLORS['background']), 
                          (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Panel principal izquierdo
        main_panel_rect = (20 + offset_x, 20 + offset_y, 650, 760)
        draw_rounded_rect(surface, COLORS['card_bg'], main_panel_rect, 25)
        
        # Título del juego
        title_text = font_title.render("🤟 Lenguaje de Señas", True, COLORS['primary'])
        title_rect = title_text.get_rect(centerx=main_panel_rect[0] + main_panel_rect[2] // 2, y=60 + offset_y)
        surface.blit(title_text, title_rect)
        
        # Letra objetivo con animación
        self.pulse_time += 0.1
        pulse_scale = 1 + 0.1 * math.sin(self.pulse_time)
        
        # Carta de la letra
        letter_card_rect = (100 + offset_x, 150 + offset_y, 200, 250)
        draw_rounded_rect(surface, COLORS['primary'], letter_card_rect, 20)
        
        # Efecto de brillo en la carta
        highlight_rect = (letter_card_rect[0], letter_card_rect[1], letter_card_rect[2], 50)
        draw_rounded_rect(surface, tuple(min(255, c + 30) for c in COLORS['primary']), highlight_rect, 20)
        
        # Letra grande
        letter_surface = font_title.render(self.vocal_actual, True, COLORS['white'])
        letter_surface = pygame.transform.scale(letter_surface, 
                                               (int(letter_surface.get_width() * pulse_scale),
                                                int(letter_surface.get_height() * pulse_scale)))
        letter_rect = letter_surface.get_rect(center=(letter_card_rect[0] + letter_card_rect[2] // 2,
                                                     letter_card_rect[1] + letter_card_rect[3] // 2))
        surface.blit(letter_surface, letter_rect)
        
        # Instrucciones
        instruction_text = font_medium.render(f"Haz la seña para la letra:", True, COLORS['light_gray'])
        instruction_rect = instruction_text.get_rect(centerx=200 + offset_x, y=120 + offset_y)
        surface.blit(instruction_text, instruction_rect)
        
        # Panel de estadísticas
        stats_panel_rect = (350 + offset_x, 150 + offset_y, 280, 250)
        draw_rounded_rect(surface, COLORS['secondary'], stats_panel_rect, 20)
        
        # Puntuación actual
        score_text = font_large.render(f"Puntuación: {self.puntuacion}", True, COLORS['white'])
        surface.blit(score_text, (stats_panel_rect[0] + 20, stats_panel_rect[1] + 30))
        
        # Mejor puntuación
        best_text = font_medium.render(f"Mejor: {self.mejor_puntuacion}", True, COLORS['accent'])
        surface.blit(best_text, (stats_panel_rect[0] + 20, stats_panel_rect[1] + 80))
        
        # Progreso del tiempo
        if tiempo_restante > 0:
            progress = 1 - (tiempo_restante / self.tiempo_preparacion)
            countdown_text = font_large.render(f"⏱️ {int(tiempo_restante + 1)}", True, COLORS['warning'])
            surface.blit(countdown_text, (stats_panel_rect[0] + 20, stats_panel_rect[1] + 130))
            
            # Barra de progreso del tiempo
            progress_rect = (100 + offset_x, 450 + offset_y, 500, 20)
            draw_progress_bar(surface, progress_rect[0], progress_rect[1], 
                            progress_rect[2], progress_rect[3], progress, COLORS['warning'])
        
        # Mensaje de feedback
        if self.mensaje_feedback:
            feedback_panel_rect = (50 + offset_x, 500 + offset_y, 550, 100)
            draw_rounded_rect(surface, tuple(c // 2 for c in self.color_feedback), feedback_panel_rect, 15)
            draw_rounded_rect(surface, self.color_feedback, 
                            (feedback_panel_rect[0] + 5, feedback_panel_rect[1] + 5,
                             feedback_panel_rect[2] - 10, feedback_panel_rect[3] - 10), 15)
            
            feedback_text = font_medium.render(self.mensaje_feedback, True, COLORS['white'])
            feedback_rect = feedback_text.get_rect(center=(feedback_panel_rect[0] + feedback_panel_rect[2] // 2,
                                                          feedback_panel_rect[1] + feedback_panel_rect[3] // 2))
            surface.blit(feedback_text, feedback_rect)
        
        # Barra de vocales
        vowel_y = 650 + offset_y
        for i, vowel in enumerate(self.vocales):
            vowel_x = 120 + i * 110 + offset_x
            vowel_rect = (vowel_x, vowel_y, 80, 80)
            
            if vowel == self.vocal_actual:
                draw_rounded_rect(surface, COLORS['accent'], vowel_rect, 15)
            else:
                draw_rounded_rect(surface, COLORS['dark_gray'], vowel_rect, 15)
            
            vowel_text = font_large.render(vowel, True, COLORS['white'])
            vowel_text_rect = vowel_text.get_rect(center=(vowel_x + 40, vowel_y + 40))
            surface.blit(vowel_text, vowel_text_rect)
        
        # Dibujar partículas
        self.particles.update()
        self.particles.draw(surface)
    
    def run(self):
        """Ejecuta el juego principal"""
        running = True
        
        while running:
            # Manejar eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        # Reiniciar juego
                        self.puntuacion = 0
                        self.vocal_actual = random.choice(self.vocales)
                        self.tiempo_inicio = time.time()
                        self.detections_deque.clear()
            
            # Procesar frame de cámara
            frame, detected_class, detected_conf, detected_box = self.process_frame()
            
            # Actualizar lógica del juego
            tiempo_restante = self.update_game_logic(detected_class, detected_conf)
            
            # Limpiar pantalla
            screen.fill(COLORS['background'])
            
            # Dibujar UI
            self.draw_ui(screen, tiempo_restante)
            
            # Dibujar feed de cámara
            self.draw_camera_feed(screen, frame, detected_box, detected_class, detected_conf)
            
            # Actualizar pantalla
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
        
        # Limpieza
        self.cap.release()
        pygame.quit()

if __name__ == "__main__":
    game = SignLanguageGame()
    game.run()