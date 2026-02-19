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
from ultralytics import YOLO
import os

# Inicializar Pygame
pygame.init()

# ──────────────────────────────────────────────
# Constantes globales de configuración
# ──────────────────────────────────────────────

WINDOW_WIDTH = 1200
"""int: Ancho de la ventana del juego en píxeles."""

WINDOW_HEIGHT = 800
"""int: Alto de la ventana del juego en píxeles."""

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("🤟 Aprende Lenguaje de Señas - Juego Interactivo")

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
"""dict[str, tuple[int, int, int]]: Paleta de colores moderna para la interfaz.

Cada clave es un nombre semántico del color y su valor es una tupla RGB.
"""

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
        for particle in self.particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.2  # gravedad
            particle['life'] -= 1
            particle['size'] *= 0.98
            if particle['life'] <= 0 or particle['size'] < 1:
                self.particles.remove(particle)

    def draw(self, surface):
        """Dibuja todas las partículas activas sobre la superficie dada.

        Utiliza alpha blending mediante superficies temporales para
        lograr el efecto de desvanecimiento progresivo.

        Args:
            surface (pygame.Surface): Superficie sobre la cual dibujar
                las partículas.
        """
        for particle in self.particles:
            alpha = max(0, particle['life'] / 60.0 * 255)
            # Crear superficie temporal para alpha blending
            temp_surface = pygame.Surface((int(particle['size'] * 2), int(particle['size'] * 2)))
            temp_surface.set_alpha(int(alpha))
            pygame.draw.circle(temp_surface, particle['color'],
                             (int(particle['size']), int(particle['size'])),
                             int(particle['size']))
            surface.blit(temp_surface, (particle['x'] - particle['size'], particle['y'] - particle['size']))


def draw_rounded_rect(surface, color, rect, radius=20):
    """Dibuja un rectángulo con esquinas redondeadas.

    Utiliza una combinación de rectángulos y círculos para simular
    bordes redondeados, ya que Pygame no soporta esta funcionalidad
    de forma nativa en versiones anteriores a 2.x.

    Args:
        surface (pygame.Surface): Superficie sobre la cual dibujar.
        color (tuple[int, int, int]): Color RGB del rectángulo.
        rect (tuple[int, int, int, int]): Tupla ``(x, y, ancho, alto)``
            que define la posición y tamaño del rectángulo.
        radius (int, optional): Radio de las esquinas redondeadas.
            Por defecto es ``20``.
    """
    x, y, w, h = rect
    pygame.draw.rect(surface, color, (x + radius, y, w - 2 * radius, h))
    pygame.draw.rect(surface, color, (x, y + radius, w, h - 2 * radius))
    pygame.draw.circle(surface, color, (x + radius, y + radius), radius)
    pygame.draw.circle(surface, color, (x + w - radius, y + radius), radius)
    pygame.draw.circle(surface, color, (x + radius, y + h - radius), radius)
    pygame.draw.circle(surface, color, (x + w - radius, y + h - radius), radius)


def draw_gradient_rect(surface, color1, color2, rect):
    """Dibuja un rectángulo con un gradiente vertical de dos colores.

    Interpola linealmente entre ``color1`` (arriba) y ``color2`` (abajo),
    dibujando una línea horizontal por cada fila de píxeles.

    Args:
        surface (pygame.Surface): Superficie sobre la cual dibujar.
        color1 (tuple[int, int, int]): Color RGB del borde superior.
        color2 (tuple[int, int, int]): Color RGB del borde inferior.
        rect (tuple[int, int, int, int]): Tupla ``(x, y, ancho, alto)``.
    """
    x, y, w, h = rect
    for i in range(h):
        ratio = i / h
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        pygame.draw.line(surface, (r, g, b), (x, y + i), (x + w, y + i))


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
        CONF_THRESHOLD (float): Umbral mínimo de confianza para aceptar
            una detección YOLO.
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
        # ── Cargar modelo YOLO ──
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, "..", "..", "train", "best.pt"),
            os.path.join(current_dir, "best.pt"),
            r"C:\Users\Asus\Desktop\U\Semillero\train\best.pt"
        ]

        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            print("Error: No se encontró el archivo del modelo best.pt")
            print(f"Buscado en: {possible_paths}")
            model_path = "best.pt"

        print(f"Cargando modelo desde: {model_path}")
        self.model = YOLO(model_path)
        print("Clases del modelo:", self.model.names)

        # ── Variables del juego ──
        self.vocales = ['A', 'E', 'I', 'O', 'U']
        self.puntuacion = 0
        self.mejor_puntuacion = 0
        self.tiempo_preparacion = 5.0
        self.vocal_actual = random.choice(self.vocales)
        self.mensaje_feedback = ""
        self.color_feedback = COLORS['success']
        self.tiempo_inicio = time.time()
        self.evaluado = False

        # ── Buffer para detecciones ──
        self.detections_deque = deque(maxlen=15)
        self.feedback_start = None
        self.feedback_duracion = 3.0
        self.CONF_THRESHOLD = 0.40

        # ── Cámara ──
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

    def process_frame(self):
        """Captura un frame de la cámara y ejecuta la inferencia YOLO.

        Lee un frame del ``VideoCapture``, ejecuta el modelo YOLO sobre él
        y extrae la detección con mayor confianza (si existe).

        Returns:
            tuple: Una tupla de 4 elementos:
                - **frame** (numpy.ndarray | None): Frame capturado en formato
                  BGR, o ``None`` si la lectura falla.
                - **detected_class** (str | None): Nombre de la clase detectada
                  con mayor confianza, o ``None``.
                - **detected_conf** (float): Confianza de la detección (``0.0``
                  si no hay detección).
                - **detected_box** (tuple[int, int, int, int] | None): Coordenadas
                  del bounding box ``(x1, y1, x2, y2)``, o ``None``.
        """
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
            except Exception:
                data = results.boxes.data.numpy()

            best_idx = int(np.argmax(data[:, 4]))
            x1, y1, x2, y2, conf, cls = data[best_idx]
            cls = int(cls)
            detected_conf = float(conf)
            detected_class = self.model.names[cls]
            detected_box = (int(x1), int(y1), int(x2), int(y2))

        return frame, detected_class, detected_conf, detected_box

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
        # Añadir detección al buffer
        if detected_class is not None and detected_conf >= self.CONF_THRESHOLD:
            self.detections_deque.append(detected_class)
        else:
            self.detections_deque.append(None)

        # Temporizador
        tiempo_transcurrido = time.time() - self.tiempo_inicio
        tiempo_restante = max(0.0, self.tiempo_preparacion - tiempo_transcurrido)

        # ── Lógica del juego ──
        if tiempo_restante > 0:
            self.evaluado = False
            self.mensaje_feedback = "¡Prepárate para hacer la seña!"
            self.color_feedback = COLORS['warning']
        else:
            if not self.evaluado:
                # Evaluar por mayoría de votos
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

        # Manejar fin del feedback → nueva ronda
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

        # Dibujar bounding box si existe una detección
        if detected_box is not None:
            x1, y1, x2, y2 = detected_box
            color = (0, 255, 0) if detected_conf >= self.CONF_THRESHOLD else (255, 255, 0)
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

        # ── Letra objetivo con animación de pulso ──
        self.pulse_time += 0.1
        pulse_scale = 1 + 0.1 * math.sin(self.pulse_time)

        letter_card_rect = (100 + offset_x, 150 + offset_y, 200, 250)
        draw_rounded_rect(surface, COLORS['primary'], letter_card_rect, 20)

        # Efecto de brillo en la carta
        highlight_rect = (letter_card_rect[0], letter_card_rect[1], letter_card_rect[2], 50)
        draw_rounded_rect(surface, tuple(min(255, c + 30) for c in COLORS['primary']), highlight_rect, 20)

        # Letra grande con escala de pulso
        letter_surface = font_title.render(self.vocal_actual, True, COLORS['white'])
        letter_surface = pygame.transform.scale(letter_surface,
                                               (int(letter_surface.get_width() * pulse_scale),
                                                int(letter_surface.get_height() * pulse_scale)))
        letter_rect = letter_surface.get_rect(center=(letter_card_rect[0] + letter_card_rect[2] // 2,
                                                     letter_card_rect[1] + letter_card_rect[3] // 2))
        surface.blit(letter_surface, letter_rect)

        # ── Instrucciones ──
        instruction_text = font_medium.render(f"Haz la seña para la letra:", True, COLORS['light_gray'])
        instruction_rect = instruction_text.get_rect(centerx=200 + offset_x, y=120 + offset_y)
        surface.blit(instruction_text, instruction_rect)

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

        # ── Barra de vocales ──
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

        # Limpieza de recursos
        self.cap.release()
        pygame.quit()


if __name__ == "__main__":
    game = SignLanguageGame()
    game.run()