"""
game.py - Juego Interactivo de Lenguaje de Señas con YOLO y Pygame.

Este módulo implementa un juego educativo que utiliza visión por computador
para enseñar lenguaje de señas de las vocales (A, E, I, O, U). El juego
captura video de la cámara web del usuario, ejecuta inferencia con un modelo
YOLO entrenado para detectar señas, y evalúa la respuesta del jugador
en tiempo real dentro de una interfaz gráfica construida con Pygame.

Dependencias:
    - pygame: Motor gráfico para la interfaz del juego.
    - cv2 (OpenCV): Captura y procesamiento de video.
    - numpy: Manipulación de arrays de imágenes.
    - ultralytics: Framework para modelos YOLO.
    - torch: Backend de deep learning (instalado como dependencia de ultralytics).

Uso:
    Ejecutar directamente desde la terminal::

        python game.py

Controles:
    - ``Q``: Salir del juego.
    - ``R``: Reiniciar la puntuación.

Autor:
    Equipo HORUS - Semillero de Investigación.
"""

import pygame
import cv2
import numpy as np
import random
import time
import math
from collections import deque, Counter
from core.detector import detector
from core.lstm_detector import lstm_detector
from core.config import COLORS, DETECTION_CONFIG, DYNAMIC_SIGNS, SIGN_DISPLAY_NAMES, LSTM_CONFIG
from core.user_manager import user_manager
from core.draw_utils import draw_lstm_prob_bars

# Inicializar Pygame solo si se ejecuta directamente
if __name__ == "__main__":
    pygame.init()

# ──────────────────────────────────────────────
# Constantes globales de configuración
# ──────────────────────────────────────────────

WINDOW_WIDTH = 1200
"""int: Ancho de la ventana del juego en píxeles."""

WINDOW_HEIGHT = 800
"""int: Alto de la ventana del juego en píxeles."""

screen = None  # se resuelve en run() para evitar captura prematura al importar

if __name__ == "__main__":
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Aprende Lenguaje de Señas - Juego Interactivo")

# Fuentes tipográficas
font_large = pygame.font.Font(None, 48)
"""pygame.font.Font: Fuente grande (48pt) para puntuaciones y títulos secundarios."""

font_medium = pygame.font.Font(None, 36)
"""pygame.font.Font: Fuente mediana (36pt) para instrucciones y feedback."""

font_small = pygame.font.Font(None, 24)
"""pygame.font.Font: Fuente pequeña (24pt) para texto auxiliar."""

font_title = pygame.font.Font(None, 64)
"""pygame.font.Font: Fuente de título (64pt) para el nombre del juego y la letra objetivo."""


class ParticleEffect:
    """Sistema de partículas para efectos visuales de retroalimentación.

    Genera y gestiona partículas animadas que se muestran al acertar
    o fallar una seña. Simula gravedad y desvanecimiento progresivo.

    Attributes:
        particles (list[dict]): Lista de partículas activas. Cada partícula
            es un diccionario con las claves ``x``, ``y``, ``vx``, ``vy``,
            ``life``, ``color`` y ``size``.
    """

    def __init__(self):
        """Inicializa el sistema de partículas con una lista vacía."""
        self.particles = []

    def add_success_particles(self, x, y):
        """Genera partículas de celebración en la posición indicada.

        Crea 20 partículas con colores aleatorios de la paleta de éxito
        (``success``, ``primary``, ``accent``) y velocidades variadas
        para simular una explosión festiva.

        Args:
            x (int): Coordenada X del centro de la explosión.
            y (int): Coordenada Y del centro de la explosión.
        """
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
        """Genera partículas de error en la posición indicada.

        Crea 15 partículas rojas con menor dispersión que las de éxito
        para indicar visualmente un fallo.

        Args:
            x (int): Coordenada X del centro de la explosión.
            y (int): Coordenada Y del centro de la explosión.
        """
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
        """Actualiza la posición y estado de todas las partículas activas.

        Aplica velocidad, gravedad (``0.2`` por frame) y reducción de
        tamaño (``×0.98``). Elimina partículas cuya vida llega a cero
        o cuyo tamaño es menor a 1 píxel.
        """
        for particle in self.particles:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.2
            particle['life'] -= 1
            particle['size'] *= 0.98
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] >= 1]

    # Superficie reutilizable para alpha blending de partículas (tamaño máximo posible)
    _MAX_PARTICLE_SIZE = 6   # ceil(random.uniform(2,5) * 1.0) al momento de creación
    _particle_surf = None

    def draw(self, surface):
        """Dibuja todas las partículas activas sobre la superficie dada.

        Reutiliza una única superficie SRCALPHA para el alpha blending,
        evitando 600 allocations/seg durante efectos de éxito/error.

        Args:
            surface (pygame.Surface): Superficie sobre la cual dibujar
                las partículas.
        """
        if ParticleEffect._particle_surf is None:
            dim = ParticleEffect._MAX_PARTICLE_SIZE * 2 + 2
            ParticleEffect._particle_surf = pygame.Surface((dim, dim), pygame.SRCALPHA)

        ps = ParticleEffect._particle_surf
        for particle in self.particles:
            alpha = max(0, int(particle['life'] / 60.0 * 255))
            r_int = int(particle['size'])
            ps.fill((0, 0, 0, 0))
            pygame.draw.circle(ps, (*particle['color'], alpha), (r_int, r_int), r_int)
            surface.blit(ps, (int(particle['x']) - r_int, int(particle['y']) - r_int))


def draw_rounded_rect(surface, color, rect, radius=20):
    x, y, w, h = rect
    if w > 0 and h > 0:
        r = min(radius, w // 2, h // 2)
        pygame.draw.rect(surface, color, (x, y, w, h), border_radius=r)


_gradient_cache: dict = {}


def draw_gradient_rect(surface, color1, color2, rect):
    """Dibuja un rectángulo con un gradiente vertical de dos colores.

    La superficie del gradiente se genera la primera vez y se reutiliza
    en llamadas posteriores con los mismos parámetros (O(1) por frame).

    Args:
        surface (pygame.Surface): Superficie sobre la cual dibujar.
        color1 (tuple[int, int, int]): Color RGB del borde superior.
        color2 (tuple[int, int, int]): Color RGB del borde inferior.
        rect (tuple[int, int, int, int]): Tupla ``(x, y, ancho, alto)``.
    """
    x, y, w, h = rect
    key = (color1, color2, w, h)
    if key not in _gradient_cache:
        grad_surf = pygame.Surface((w, h))
        for i in range(h):
            ratio = i / h
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            pygame.draw.line(grad_surf, (r, g, b), (0, i), (w, i))
        _gradient_cache[key] = grad_surf
    surface.blit(_gradient_cache[key], (x, y))


def draw_progress_bar(surface, x, y, width, height, progress, color):
    """Dibuja una barra de progreso con estilo moderno.

    Renderiza un fondo gris oscuro y una barra de progreso con un
    efecto de brillo en la parte superior para dar profundidad visual.

    Args:
        surface (pygame.Surface): Superficie sobre la cual dibujar.
        x (int): Coordenada X de la esquina superior izquierda.
        y (int): Coordenada Y de la esquina superior izquierda.
        width (int): Ancho total de la barra en píxeles.
        height (int): Alto de la barra en píxeles.
        progress (float): Valor de progreso normalizado entre ``0.0`` y ``1.0``.
        color (tuple[int, int, int]): Color RGB de la barra de progreso.
    """
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
    """Convierte una imagen de OpenCV (BGR, numpy) a una superficie Pygame (RGB).

    Realiza la conversión de espacio de color BGR→RGB y la rotación
    de 90° necesaria para que ``pygame.surfarray.make_surface``
    interprete correctamente la orientación de la imagen.

    Args:
        cv_image (numpy.ndarray): Imagen en formato OpenCV (BGR, ``H×W×3``).

    Returns:
        pygame.Surface: Superficie Pygame lista para renderizar.
    """
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    cv_image = np.rot90(cv_image)
    cv_image = pygame.surfarray.make_surface(cv_image)
    return cv_image


class SignLanguageGame:
    """Clase principal del juego interactivo de lenguaje de señas.

    Gestiona el ciclo de vida completo del juego: carga del modelo YOLO,
    captura de cámara, lógica de juego (temporizador, evaluación por
    mayoría de detecciones), renderizado de la interfaz gráfica y
    efectos visuales.

    Attributes:
        model (YOLO): Modelo YOLO cargado para detección de señas.
        vocales (list[str]): Lista de vocales que el jugador debe representar.
        puntuacion (int): Puntuación actual acumulada en la sesión.
        mejor_puntuacion (int): Mejor puntuación alcanzada en la sesión.
        tiempo_preparacion (float): Segundos que el jugador tiene para
            preparar la seña antes de la evaluación.
        vocal_actual (str): Vocal que el jugador debe representar actualmente.
        mensaje_feedback (str): Mensaje de retroalimentación mostrado al jugador.
        color_feedback (tuple[int, int, int]): Color RGB del panel de feedback.
        tiempo_inicio (float): Timestamp del inicio de la ronda actual.
        evaluado (bool): Indica si la ronda actual ya fue evaluada.
        detections_deque (deque): Buffer circular de detecciones recientes,
            con capacidad máxima de 15 elementos.
        feedback_start (float | None): Timestamp del inicio del feedback,
            o ``None`` si no hay feedback activo.
        feedback_duracion (float): Duración en segundos del mensaje de feedback.
        cap (cv2.VideoCapture): Objeto de captura de video.
        particles (ParticleEffect): Sistema de partículas para efectos visuales.
        pulse_time (float): Acumulador para la animación de pulso de la letra.
        shake_intensity (int): Intensidad actual del efecto de vibración.
        shake_duration (int): Duración restante del efecto de vibración en ms.
        game_state (str): Estado actual del juego (``"playing"``, ``"paused"``,
            ``"menu"``).
        clock (pygame.time.Clock): Reloj para controlar los FPS.
    """

    def __init__(self):
        """Inicializa el juego: carga modelo, abre cámara y configura estado.

        Busca el modelo ``best.pt`` en varias rutas posibles (relativas y
        absolutas) para garantizar portabilidad. Abre la cámara web con
        resolución 640×480. Inicializa todos los contadores, buffers y
        sistemas de efectos visuales.

        Raises:
            FileNotFoundError: Si el modelo ``best.pt`` no se encuentra
                en ninguna de las rutas buscadas (se imprime un mensaje
                y se intenta cargar como fallback).
        """
        print(f"Usando Detector Global Optimizado en: {detector.device}")
        self.model = detector.model

        # ── Variables del juego ──
        self.all_signs = ['A', 'E', 'I', 'O', 'U'] + DYNAMIC_SIGNS
        self.puntuacion = 0
        self.mejor_puntuacion = user_manager.get_best_score()
        self.tiempo_preparacion = 5.0
        self.vocal_actual = random.choice(self.all_signs)
        self.mensaje_feedback = ""
        self.color_feedback = COLORS['success']
        self.tiempo_inicio = time.time()
        self.evaluado = False

        # ── Buffer para detecciones ──
        self.detections_deque = deque(maxlen=DETECTION_CONFIG['detections_buffer'])
        self.feedback_start = None
        self.feedback_duracion = 3.0
        # ── LSTM ──
        self._lstm_frame_count = 0
        if self.vocal_actual in DYNAMIC_SIGNS:
            lstm_detector.initialize()
            lstm_detector.reset()

        # ── Cámara ──
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            print("Advertencia: no se pudo abrir ninguna cámara.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame_count = 0
        self.last_detected = (None, None, 0.0, None)

        # ── Efectos visuales ──
        self.particles = ParticleEffect()
        self.pulse_time = 0
        self.shake_intensity = 0
        self.shake_duration = 0

        # ── Estado del juego ──
        self.game_state = "playing"

        # ── Clock para FPS ──
        self.clock = pygame.time.Clock()

    def add_screen_shake(self, intensity=10, duration=500):
        """Activa el efecto de vibración de pantalla.

        Este efecto desplaza aleatoriamente toda la interfaz durante
        unos milisegundos, simulando un temblor visual para reforzar
        la retroalimentación negativa.

        Args:
            intensity (int, optional): Amplitud máxima del desplazamiento
                en píxeles. Por defecto es ``10``.
            duration (int, optional): Duración del efecto en milisegundos.
                Por defecto es ``500``.
        """
        self.shake_intensity = intensity
        self.shake_duration = duration

    def get_screen_offset(self):
        """Calcula el desplazamiento actual para el efecto de vibración.

        Reduce la duración restante del efecto en cada frame y genera
        offsets aleatorios dentro del rango de intensidad configurado.

        Returns:
            tuple[int, int]: Desplazamiento ``(offset_x, offset_y)`` a
            aplicar a todos los elementos de la interfaz. Retorna
            ``(0, 0)`` si no hay vibración activa.
        """
        if self.shake_duration > 0:
            self.shake_duration -= self.clock.get_time()
            offset_x = random.randint(-self.shake_intensity, self.shake_intensity)
            offset_y = random.randint(-self.shake_intensity, self.shake_intensity)
            return offset_x, offset_y
        return 0, 0

    def update_game_logic(self, detected_class, detected_conf):
        """Actualiza la lógica del juego basada en la detección actual.

        Gestiona el ciclo de cada ronda:
        1. Añade la detección al buffer circular.
        2. Calcula el tiempo restante de preparación.
        3. Cuando el tiempo se agota, evalúa por **votación de mayoría**
           sobre las detecciones acumuladas en el buffer.
        4. Genera feedback visual (partículas, vibración) y actualiza
           la puntuación.
        5. Tras la duración del feedback, inicia una nueva ronda.

        Args:
            detected_class (str | None): Clase detectada en el frame actual.
            detected_conf (float): Confianza de la detección actual.

        Returns:
            float: Tiempo restante de preparación en segundos.
                Retorna ``0.0`` si el tiempo ya expiró.
        """
        # Añadir detección al buffer (filtrar no_sena en señas dinámicas)
        is_dynamic = self.vocal_actual in DYNAMIC_SIGNS
        threshold = LSTM_CONFIG['confidence_threshold'] if is_dynamic else DETECTION_CONFIG['conf_threshold']

        if (detected_class is not None
                and detected_conf >= threshold
                and detected_class != "no_sena"):
            self.detections_deque.append(detected_class)
        else:
            self.detections_deque.append(None)

        # Temporizador
        tiempo_transcurrido = time.time() - self.tiempo_inicio
        tiempo_restante = max(0.0, self.tiempo_preparacion - tiempo_transcurrido)
        sign_display = SIGN_DISPLAY_NAMES.get(self.vocal_actual, self.vocal_actual)

        # ── Lógica del juego ──
        if tiempo_restante > 0:
            self.evaluado = False
            if is_dynamic and lstm_detector.buffer_progress < 1.0:
                pct = int(lstm_detector.buffer_progress * 100)
                self.mensaje_feedback = f"Acumulando seña... {pct}%"
            else:
                self.mensaje_feedback = "¡Prepárate para hacer la seña!"
            self.color_feedback = COLORS['warning']
        else:
            if not self.evaluado:
                clases_validas = [d for d in self.detections_deque if d is not None]
                min_votes = LSTM_CONFIG['min_votes'] if is_dynamic else DETECTION_CONFIG['min_votes']

                if len(clases_validas) == 0:
                    self.mensaje_feedback = "❌ No se detectó la seña claramente"
                    self.color_feedback = COLORS['error']
                    self.particles.add_error_particles(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
                    self.add_screen_shake(5, 300)
                else:
                    counts = Counter(clases_validas)
                    top_class, top_count = counts.most_common(1)[0]

                    correct = (
                        top_class == self.vocal_actual
                        if is_dynamic
                        else top_class.upper() == self.vocal_actual.upper()
                    )

                    if correct and top_count >= min_votes:
                        self.puntuacion += 1
                        self.mejor_puntuacion = max(self.mejor_puntuacion, self.puntuacion)
                        self.mensaje_feedback = f"🎉 ¡Excelente! Acertaste '{sign_display}'"
                        self.color_feedback = COLORS['success']
                        self.particles.add_success_particles(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
                    else:
                        self.mensaje_feedback = f"❌ Incorrecto. Era '{sign_display}'"
                        self.color_feedback = COLORS['error']
                        self.particles.add_error_particles(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
                        self.add_screen_shake(8, 400)

                self.evaluado = True
                self.feedback_start = time.time()

        # Manejar fin del feedback → nueva ronda
        if self.evaluado and self.feedback_start is not None:
            if time.time() - self.feedback_start > self.feedback_duracion:
                self.vocal_actual = random.choice(self.all_signs)
                # Si la nueva seña es dinámica, inicializar/resetear LSTM
                if self.vocal_actual in DYNAMIC_SIGNS:
                    lstm_detector.initialize()
                    lstm_detector.reset()
                    self._lstm_frame_count = 0
                self.tiempo_inicio = time.time()
                self.mensaje_feedback = ""
                self.evaluado = False
                self.feedback_start = None
                self.detections_deque.clear()

        return tiempo_restante

    def draw_camera_feed(self, surface, frame, detected_box, detected_class, detected_conf):
        """Dibuja el feed de la cámara con las detecciones YOLO superpuestas.

        Renderiza el frame con bounding boxes coloreados según el umbral
        de confianza (verde si supera ``CONF_THRESHOLD``, amarillo si no),
        lo escala a 480×360 y lo posiciona en la esquina superior derecha
        con un marco decorativo.

        Args:
            surface (pygame.Surface): Superficie principal del juego.
            frame (numpy.ndarray | None): Frame de la cámara en formato BGR.
                Si es ``None``, no se dibuja nada.
            detected_box (tuple[int, int, int, int] | None): Coordenadas
                del bounding box de la detección.
            detected_class (str | None): Nombre de la clase detectada.
            detected_conf (float): Nivel de confianza de la detección.
        """
        if frame is None:
            return

        frame_copy = frame.copy()

        is_dynamic = self.vocal_actual in DYNAMIC_SIGNS
        if is_dynamic and lstm_detector._initialized:
            lstm_detector.draw_landmarks(frame_copy)

        # Barras de probabilidad LSTM superpuestas en el frame
        if is_dynamic and lstm_detector._initialized and lstm_detector.buffer_progress >= 1.0:
            draw_lstm_prob_bars(frame_copy, lstm_detector.labels, lstm_detector.last_probs, bar_max_w=200)

        # Dibujar bounding box si existe una detección
        if detected_box is not None:
            x1, y1, x2, y2 = detected_box
            color = (0, 255, 0) if detected_conf >= DETECTION_CONFIG['conf_threshold'] else (255, 255, 0)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)

            # Etiqueta con fondo
            label = f"{detected_class} {detected_conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame_copy, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Convertir a Pygame y escalar
        pygame_frame = opencv_to_pygame(frame_copy)
        pygame_frame = pygame.transform.scale(pygame_frame, (480, 360))

        # Posición de la cámara (esquina superior derecha)
        camera_x = WINDOW_WIDTH - 500
        camera_y = 20

        # Marco decorativo
        frame_rect = (camera_x - 10, camera_y - 10, 500, 380)
        draw_rounded_rect(surface, COLORS['card_bg'], frame_rect, 15)
        draw_rounded_rect(surface, COLORS['primary'], (camera_x - 12, camera_y - 12, 504, 384), 15)

        surface.blit(pygame_frame, (camera_x, camera_y))

    def draw_ui(self, surface, tiempo_restante):
        """Dibuja la interfaz de usuario completa del juego.

        Renderiza todos los elementos de la UI en el siguiente orden:
        1. Fondo con gradiente.
        2. Panel principal izquierdo con la carta de la letra objetivo.
        3. Panel de estadísticas (puntuación actual y mejor).
        4. Barra de progreso del temporizador.
        5. Panel de feedback con mensaje de retroalimentación.
        6. Barra inferior con las 5 vocales.
        7. Efectos de partículas.

        Args:
            surface (pygame.Surface): Superficie principal del juego.
            tiempo_restante (float): Tiempo restante de preparación en segundos.
        """
        offset_x, offset_y = self.get_screen_offset()

        # ── Fondo con gradiente ──
        draw_gradient_rect(surface, COLORS['background'],
                          tuple(max(0, c - 20) for c in COLORS['background']),
                          (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))

        # ── Panel principal izquierdo ──
        main_panel_rect = (20 + offset_x, 20 + offset_y, 650, 760)
        draw_rounded_rect(surface, COLORS['card_bg'], main_panel_rect, 25)

        # ── Título del juego ──
        title_text = font_title.render("Lenguaje de Señas", True, COLORS['primary'])
        title_rect = title_text.get_rect(centerx=main_panel_rect[0] + main_panel_rect[2] // 2, y=60 + offset_y)
        surface.blit(title_text, title_rect)

        # ── Carta de la seña objetivo ──
        self.pulse_time += 0.1
        pulse_scale = 1 + 0.1 * math.sin(self.pulse_time)

        is_dynamic = self.vocal_actual in DYNAMIC_SIGNS
        sign_display = SIGN_DISPLAY_NAMES.get(self.vocal_actual, self.vocal_actual)

        letter_card_rect = (60 + offset_x, 150 + offset_y, 240, 250)
        card_color = COLORS['secondary'] if is_dynamic else COLORS['primary']
        draw_rounded_rect(surface, card_color, letter_card_rect, 20)
        highlight_rect = (letter_card_rect[0], letter_card_rect[1], letter_card_rect[2], 50)
        draw_rounded_rect(surface, tuple(min(255, c + 30) for c in card_color), highlight_rect, 20)

        if is_dynamic:
            # Señas dinámicas: nombre en dos líneas si es largo
            words = sign_display.split()
            font_sign = font_large if len(sign_display) > 8 else font_title
            cy = letter_card_rect[1] + letter_card_rect[3] // 2 - (len(words) - 1) * 25
            for w in words:
                ws = font_sign.render(w, True, COLORS['white'])
                surface.blit(ws, ws.get_rect(centerx=letter_card_rect[0] + letter_card_rect[2] // 2, centery=cy))
                cy += font_sign.get_height() - 10
        else:
            # Vocales: letra grande con pulso
            letter_surface = font_title.render(sign_display, True, COLORS['white'])
            letter_surface = pygame.transform.scale(
                letter_surface,
                (int(letter_surface.get_width() * pulse_scale),
                 int(letter_surface.get_height() * pulse_scale)),
            )
            letter_rect = letter_surface.get_rect(
                center=(letter_card_rect[0] + letter_card_rect[2] // 2,
                        letter_card_rect[1] + letter_card_rect[3] // 2)
            )
            surface.blit(letter_surface, letter_rect)

        # ── Instrucciones ──
        instr = "Realiza la seña:" if is_dynamic else "Haz la seña para la letra:"
        instruction_text = font_medium.render(instr, True, COLORS['light_gray'])
        instruction_rect = instruction_text.get_rect(centerx=180 + offset_x, y=120 + offset_y)
        surface.blit(instruction_text, instruction_rect)

        # ── Barra de buffer LSTM (solo en señas dinámicas, durante preparación) ──
        if is_dynamic and tiempo_restante > 0:
            buf = lstm_detector.buffer_progress
            bar_w, bar_h = 500, 12
            bx = 100 + offset_x
            by = 475 + offset_y
            draw_rounded_rect(surface, COLORS['dark_gray'], (bx, by, bar_w, bar_h), 6)
            if buf > 0:
                draw_rounded_rect(surface, COLORS['secondary'], (bx, by, int(bar_w * buf), bar_h), 6)
            lbl = font_small.render("Acumulando seña...", True, COLORS['light_gray'])
            surface.blit(lbl, (bx, by - 22))

        # ── Mostrar nombre del niño ──
        child_name = user_manager.get_user_name()
            
        name_text = font_medium.render(f"Jugador: {child_name}", True, COLORS['secondary'])
        surface.blit(name_text, (20 + offset_x, main_panel_rect[1] + main_panel_rect[3] - 40))

        # ── Panel de estadísticas ──
        stats_panel_rect = (350 + offset_x, 150 + offset_y, 280, 250)
        draw_rounded_rect(surface, COLORS['secondary'], stats_panel_rect, 20)

        score_text = font_large.render(f"Puntuación: {self.puntuacion}", True, COLORS['white'])
        surface.blit(score_text, (stats_panel_rect[0] + 20, stats_panel_rect[1] + 30))

        best_text = font_medium.render(f"Mejor: {self.mejor_puntuacion}", True, COLORS['accent'])
        surface.blit(best_text, (stats_panel_rect[0] + 20, stats_panel_rect[1] + 80))

        # ── Progreso del tiempo ──
        if tiempo_restante > 0:
            progress = 1 - (tiempo_restante / self.tiempo_preparacion)
            countdown_text = font_large.render(f"⏱️ {int(tiempo_restante + 1)}", True, COLORS['warning'])
            surface.blit(countdown_text, (stats_panel_rect[0] + 20, stats_panel_rect[1] + 130))

            progress_rect = (100 + offset_x, 450 + offset_y, 500, 20)
            draw_progress_bar(surface, progress_rect[0], progress_rect[1],
                            progress_rect[2], progress_rect[3], progress, COLORS['warning'])

        # ── Mensaje de feedback ──
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

        # ── Barra de señas (vocales arriba, dinámicas abajo) ──
        vowels = ['A', 'E', 'I', 'O', 'U']
        vow_y = 630 + offset_y
        vow_w = 70
        vow_start = 80 + offset_x
        for i, v in enumerate(vowels):
            vr = (vow_start + i * 90, vow_y, vow_w, vow_w)
            color = COLORS['accent'] if v == self.vocal_actual else COLORS['dark_gray']
            draw_rounded_rect(surface, color, vr, 12)
            vt = font_large.render(v, True, COLORS['white'])
            surface.blit(vt, vt.get_rect(center=(vr[0] + vow_w // 2, vr[1] + vow_w // 2)))

        dyn_y = 710 + offset_y
        dyn_signs = [('hola', 'Hola'), ('hola_mundo', 'H.Mundo'), ('buenos_dias', 'B.Días')]
        dyn_w = 140
        dyn_start = 80 + offset_x
        for i, (key, short) in enumerate(dyn_signs):
            dr = (dyn_start + i * 160, dyn_y, dyn_w, 50)
            color = COLORS['accent'] if key == self.vocal_actual else COLORS['card_bg']
            draw_rounded_rect(surface, color, dr, 10)
            dt = font_small.render(short, True, COLORS['white'])
            surface.blit(dt, dt.get_rect(center=(dr[0] + dyn_w // 2, dr[1] + 25)))

        # ── Dibujar partículas ──
        self.particles.update()
        self.particles.draw(surface)

    def run(self):
        """Ejecuta el bucle principal del juego.

        Gestiona el game loop completo a 30 FPS:
        1. Procesa eventos de teclado (``Q`` para salir, ``R`` para reiniciar).
        2. Captura y procesa un frame de la cámara.
        3. Actualiza la lógica del juego.
        4. Renderiza la UI y el feed de la cámara.
        5. Al salir, libera la cámara y cierra Pygame.
        """
        global screen
        if screen is None:
            screen = pygame.display.get_surface()

        running = True

        try:
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
                            self.vocal_actual = random.choice(self.all_signs)
                            if self.vocal_actual in DYNAMIC_SIGNS:
                                lstm_detector.initialize()
                                lstm_detector.reset()
                                self._lstm_frame_count = 0
                            self.tiempo_inicio = time.time()
                            self.detections_deque.clear()

                # Capturar frame (siempre necesario para mostrar cámara)
                ret, frame = self.cap.read()
                if not ret:
                    frame = None

                is_dynamic = self.vocal_actual in DYNAMIC_SIGNS

                if is_dynamic:
                    # ── Detección LSTM ──
                    self._lstm_frame_count += 1
                    detected_box = None
                    if frame is not None and self._lstm_frame_count % LSTM_CONFIG['skip_frames'] == 0:
                        detected_class, detected_conf = lstm_detector.process_frame(frame)
                    else:
                        detected_class, detected_conf = None, 0.0
                else:
                    # ── Detección YOLO ──
                    if self.frame_count % DETECTION_CONFIG['skip_frames'] == 0:
                        if frame is not None:
                            detected_class, detected_conf, detected_box = detector.predict(frame)
                        else:
                            detected_class, detected_conf, detected_box = None, 0.0, None
                        self.last_detected = (frame, detected_class, detected_conf, detected_box)
                    else:
                        frame, detected_class, detected_conf, detected_box = self.last_detected

                self.frame_count += 1

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

        finally:
            # Limpieza garantizada incluso si ocurre una excepción
            self.cap.release()
            lstm_detector.close()

            user_manager.update_stats(self.puntuacion)

            if __name__ == "__main__":
                pygame.quit()


if __name__ == "__main__":
    game = SignLanguageGame()
    game.run()