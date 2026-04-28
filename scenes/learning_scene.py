import pygame
import cv2
import numpy as np
import os
from scenes.base_scene import BaseScene
from core.config import (
    COLORS, WIDTH, HEIGHT, VOWEL_IMAGES, DETECTION_CONFIG,
    DYNAMIC_SIGNS, SIGN_DISPLAY_NAMES, LSTM_CONFIG,
)
from core.detector import detector
from core.lstm_detector import lstm_detector


# ── Constantes de layout ──────────────────────────────────────────────
_TAB_W, _TAB_H = 280, 60
_TAB_Y = 160
_TAB_VOWELS_X = WIDTH // 2 - _TAB_W - 10
_TAB_SIGNS_X = WIDTH // 2 + 10

_BTN_W, _BTN_H = 220, 80
_BACK_RECT = pygame.Rect(50, HEIGHT - 100, 200, 60)


class LearningScene(BaseScene):
    """Escena de aprendizaje con soporte para vocales (YOLO) y señas dinámicas (LSTM)."""

    def __init__(self):
        super().__init__()

        # ── Estado general ─────────────────────────────────────────────
        # "selection" → usuario elige qué aprender
        # "loading"   → cargando detector LSTM (un frame de pantalla de espera)
        # "practice"  → modo práctica activo
        self.state = "selection"
        self.active_tab = "vowels"       # "vowels" | "signs"
        self.sign_type = "vowel"         # "vowel"  | "dynamic"
        self.selected_sign = None        # clave interna, ej. 'A' o 'hola'
        self._loading_step = 0           # control de pantalla de carga

        # ── Cámara ────────────────────────────────────────────────────
        self.cap = None
        self.frame_count = 0
        self.cam_surface = None

        # ── Feedback ──────────────────────────────────────────────────
        self.feedback_msg = ""
        self.feedback_color = COLORS['white']

        # ── Imagen de referencia (vocales) ────────────────────────────
        self.vowel_image = None

        # ── Detección LSTM ────────────────────────────────────────────
        self._lstm_frame_count = 0
        self._last_lstm_class = None
        self._last_lstm_conf = 0.0

        # ── UI — tabs ─────────────────────────────────────────────────
        self.tab_vowels_rect = pygame.Rect(_TAB_VOWELS_X, _TAB_Y, _TAB_W, _TAB_H)
        self.tab_signs_rect = pygame.Rect(_TAB_SIGNS_X, _TAB_Y, _TAB_W, _TAB_H)
        self.tab_vowels_hover = False
        self.tab_signs_hover = False

        # ── UI — botones vocales ───────────────────────────────────────
        btn_w, btn_h = 100, 100
        start_x = WIDTH // 2 - (btn_w * 5 + 40 * 4) // 2
        self.vowel_buttons = [
            {
                "label": v,
                "rect": pygame.Rect(start_x + i * (btn_w + 40), HEIGHT // 2, btn_w, btn_h),
                "hover": False,
            }
            for i, v in enumerate(['A', 'E', 'I', 'O', 'U'])
        ]

        # ── UI — botones señas dinámicas ──────────────────────────────
        signs_data = [
            ('hola',        'Hola'),
            ('hola_mundo',  'Hola\nMundo'),
            ('buenos_dias', 'Buenos\nDías'),
        ]
        total_w = len(signs_data) * _BTN_W + (len(signs_data) - 1) * 40
        sx = WIDTH // 2 - total_w // 2
        self.sign_buttons = [
            {
                "key":   key,
                "label": label,
                "rect":  pygame.Rect(sx + i * (_BTN_W + 40), HEIGHT // 2 - _BTN_H // 2, _BTN_W, _BTN_H),
                "hover": False,
            }
            for i, (key, label) in enumerate(signs_data)
        ]

        self.back_hover = False

    # ══════════════════════════════════════════════════════════════════
    # Eventos
    # ══════════════════════════════════════════════════════════════════

    def process_events(self, events):
        if self.state == "loading":
            return

        mouse = pygame.mouse.get_pos()
        self.back_hover = _BACK_RECT.collidepoint(mouse)

        if self.state == "selection":
            self.tab_vowels_hover = self.tab_vowels_rect.collidepoint(mouse)
            self.tab_signs_hover = self.tab_signs_rect.collidepoint(mouse)

            if self.active_tab == "vowels":
                for btn in self.vowel_buttons:
                    btn["hover"] = btn["rect"].collidepoint(mouse)
            else:
                for btn in self.sign_buttons:
                    btn["hover"] = btn["rect"].collidepoint(mouse)

        for event in events:
            if event.type != pygame.MOUSEBUTTONDOWN:
                continue

            if self.back_hover:
                self._handle_back()
                return

            if self.state == "selection":
                if self.tab_vowels_hover:
                    self.active_tab = "vowels"
                    return
                if self.tab_signs_hover:
                    self.active_tab = "signs"
                    return

                if self.active_tab == "vowels":
                    for btn in self.vowel_buttons:
                        if btn["hover"]:
                            self._start_vowel_practice(btn["label"])
                else:
                    for btn in self.sign_buttons:
                        if btn["hover"]:
                            self._start_sign_loading(btn["key"])

    # ══════════════════════════════════════════════════════════════════
    # Arranque de práctica
    # ══════════════════════════════════════════════════════════════════

    def _start_vowel_practice(self, vowel):
        self.selected_sign = vowel
        self.sign_type = "vowel"
        self._load_vowel_image(vowel)
        self._start_camera()
        self.feedback_msg = ""
        self.frame_count = 0
        self.state = "practice"

    def _start_sign_loading(self, sign_key):
        """Pone estado 'loading' para mostrar la pantalla de espera un frame antes de cargar LSTM."""
        self.selected_sign = sign_key
        self.sign_type = "dynamic"
        self._loading_step = 0
        self.state = "loading"

    # ══════════════════════════════════════════════════════════════════
    # Update
    # ══════════════════════════════════════════════════════════════════

    def update(self):
        if self.state == "loading":
            self._loading_step += 1
            if self._loading_step >= 2:   # al menos un frame de pantalla de carga
                lstm_detector.initialize()
                lstm_detector.reset()
                self._lstm_frame_count = 0
                self._last_lstm_class = None
                self._last_lstm_conf = 0.0
                self._start_camera()
                self.feedback_msg = ""
                self.state = "practice"
            return

        if self.state != "practice" or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_count += 1

        if self.sign_type == "vowel":
            self._update_yolo(frame)
        else:
            self._update_lstm(frame)

        # Convertir frame para mostrar en pantalla
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (320, 240))
        surf = pygame.surfarray.make_surface(np.rot90(rgb))
        self.cam_surface = pygame.transform.flip(surf, True, False)

    def _update_yolo(self, frame):
        if self.frame_count % DETECTION_CONFIG['skip_frames'] != 0:
            return
        detected_class, conf, _ = detector.predict(frame)
        if detected_class is None:
            self.feedback_msg = "Buscando seña..."
            self.feedback_color = COLORS['warning']
        elif detected_class.upper() == self.selected_sign.upper():
            self.feedback_msg = "Correcto ✅"
            self.feedback_color = COLORS['success']
        else:
            self.feedback_msg = "Intenta de nuevo ❌"
            self.feedback_color = COLORS['error']

    def _update_lstm(self, frame):
        self._lstm_frame_count += 1
        if self._lstm_frame_count % LSTM_CONFIG['skip_frames'] != 0:
            return

        detected_class, conf = lstm_detector.process_frame(frame)
        progress = lstm_detector.buffer_progress

        if progress < 1.0:
            pct = int(progress * 100)
            self.feedback_msg = f"Acumulando seña... {pct}%"
            self.feedback_color = COLORS['warning']
            return

        self._last_lstm_class = detected_class
        self._last_lstm_conf = conf

        if detected_class == "no_sena" or conf < LSTM_CONFIG['confidence_threshold']:
            self.feedback_msg = "No se detectó la seña"
            self.feedback_color = COLORS['warning']
        elif detected_class == self.selected_sign:
            self.feedback_msg = f"¡Correcto! ✅  ({conf:.0%})"
            self.feedback_color = COLORS['success']
        else:
            display = SIGN_DISPLAY_NAMES.get(detected_class, detected_class)
            self.feedback_msg = f"Parece '{display}' — intenta de nuevo ❌"
            self.feedback_color = COLORS['error']

    # ══════════════════════════════════════════════════════════════════
    # Draw
    # ══════════════════════════════════════════════════════════════════

    def draw(self, screen):
        screen.fill(COLORS['background'])

        if self.state == "loading":
            self._draw_loading(screen)
            return

        if self.state == "selection":
            self._draw_selection(screen)
        elif self.state == "practice":
            self._draw_practice(screen)

        self.draw_button(screen, _BACK_RECT, "Volver", COLORS['accent'], self.back_hover)

    # ── Loading ───────────────────────────────────────────────────────

    def _draw_loading(self, screen):
        sign_name = SIGN_DISPLAY_NAMES.get(self.selected_sign, self.selected_sign)
        title = self.font_large.render("Cargando detector de señas...", True, COLORS['primary'])
        screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 40)))
        sub = self.font_small.render(
            f"Preparando para practicar: {sign_name}", True, COLORS['light_gray']
        )
        screen.blit(sub, sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 20)))

    # ── Selection ─────────────────────────────────────────────────────

    def _draw_selection(self, screen):
        title = self.font_large.render("¿Qué quieres aprender?", True, COLORS['primary'])
        screen.blit(title, title.get_rect(center=(WIDTH // 2, 90)))

        self._draw_tab(screen, self.tab_vowels_rect, "Vocales",
                       self.active_tab == "vowels", self.tab_vowels_hover)
        self._draw_tab(screen, self.tab_signs_rect, "Señas",
                       self.active_tab == "signs", self.tab_signs_hover)

        if self.active_tab == "vowels":
            for btn in self.vowel_buttons:
                self.draw_button(screen, btn["rect"], btn["label"],
                                 COLORS['secondary'], btn["hover"])
        else:
            self._draw_sign_buttons(screen)

    def _draw_tab(self, screen, rect, label, active, hover):
        color = COLORS['primary'] if active else COLORS['card_bg']
        if hover and not active:
            color = tuple(min(255, c + 20) for c in color)
        pygame.draw.rect(screen, color, rect, border_radius=12)
        border_color = COLORS['primary'] if active else COLORS['dark_gray']
        pygame.draw.rect(screen, border_color, rect, 3, border_radius=12)
        text = self.font_small.render(label, True, COLORS['white'])
        screen.blit(text, text.get_rect(center=rect.center))

    def _draw_sign_buttons(self, screen):
        for btn in self.sign_buttons:
            color = COLORS['secondary'] if not btn["hover"] else tuple(
                min(255, c + 30) for c in COLORS['secondary']
            )
            pygame.draw.rect(screen, color, btn["rect"], border_radius=20)
            pygame.draw.rect(screen, COLORS['white'], btn["rect"], 3, border_radius=20)
            lines = btn["label"].split("\n")
            line_h = self.font_small.get_height()
            total_h = line_h * len(lines) + 6 * (len(lines) - 1)
            y0 = btn["rect"].centery - total_h // 2
            for j, line in enumerate(lines):
                surf = self.font_small.render(line, True, COLORS['white'])
                screen.blit(surf, surf.get_rect(centerx=btn["rect"].centerx, y=y0 + j * (line_h + 6)))

    # ── Practice ──────────────────────────────────────────────────────

    def _draw_practice(self, screen):
        sign_name = SIGN_DISPLAY_NAMES.get(self.selected_sign, self.selected_sign)
        title = self.font_large.render(f"Practicando: {sign_name}", True, COLORS['white'])
        screen.blit(title, title.get_rect(center=(WIDTH // 2, 60)))

        # Columna izquierda — referencia
        ref_rect = pygame.Rect(WIDTH // 4 - 150, HEIGHT // 2 - 150, 300, 300)
        if self.sign_type == "vowel" and self.vowel_image:
            screen.blit(self.vowel_image, ref_rect.topleft)
        else:
            self._draw_placeholder(screen, ref_rect)

        # Columna derecha — cámara
        if self.cam_surface:
            cam_rect = self.cam_surface.get_rect(center=(3 * WIDTH // 4, HEIGHT // 2))
            pygame.draw.rect(screen, COLORS['primary'], cam_rect.inflate(10, 10), 3, border_radius=10)
            screen.blit(self.cam_surface, cam_rect)

        # Barra de progreso de buffer LSTM
        if self.sign_type == "dynamic":
            self._draw_lstm_progress(screen)

        # Feedback
        fb = self.font_large.render(self.feedback_msg, True, self.feedback_color)
        screen.blit(fb, fb.get_rect(center=(WIDTH // 2, HEIGHT - 160)))

    def _draw_placeholder(self, screen, rect):
        pygame.draw.rect(screen, COLORS['card_bg'], rect, border_radius=15)
        pygame.draw.rect(screen, COLORS['dark_gray'], rect, 2, border_radius=15)
        icon = self.font_medium.render("🎬", True, COLORS['dark_gray'])
        screen.blit(icon, icon.get_rect(center=(rect.centerx, rect.centery - 20)))
        lbl = self.font_small.render("Video próximamente", True, COLORS['dark_gray'])
        screen.blit(lbl, lbl.get_rect(center=(rect.centerx, rect.centery + 30)))

    def _draw_lstm_progress(self, screen):
        progress = lstm_detector.buffer_progress
        bar_w, bar_h = 400, 14
        bx = WIDTH // 2 - bar_w // 2
        by = HEIGHT - 220
        pygame.draw.rect(screen, COLORS['card_bg'], (bx, by, bar_w, bar_h), border_radius=7)
        if progress > 0:
            filled = int(bar_w * progress)
            pygame.draw.rect(screen, COLORS['primary'], (bx, by, filled, bar_h), border_radius=7)
        lbl = self.font_small.render("Buffer de seña", True, COLORS['light_gray'])
        screen.blit(lbl, lbl.get_rect(center=(WIDTH // 2, by - 18)))

    # ══════════════════════════════════════════════════════════════════
    # Helpers internos
    # ══════════════════════════════════════════════════════════════════

    def _load_vowel_image(self, vowel):
        path = VOWEL_IMAGES.get(vowel)
        if path and os.path.exists(path):
            try:
                img = pygame.image.load(path)
                self.vowel_image = pygame.transform.scale(img, (300, 300))
                return
            except pygame.error:
                pass
        self.vowel_image = None

    def _start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def _handle_back(self):
        if self.state == "practice":
            if self.sign_type == "dynamic":
                lstm_detector.reset()
            self._stop_camera()
            self.cam_surface = None
            self.feedback_msg = ""
            self.state = "selection"
        else:
            self._stop_camera()
            from scenes.menu_scene import MenuScene
            self.switch_to(MenuScene)

    def __del__(self):
        self._stop_camera()
