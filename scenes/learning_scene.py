"""scenes/learning_scene.py

Escena de aprendizaje con soporte para vocales (YOLO) y senas dinamicas (LSTM).

Cambios v3:
- Inferencia YOLO/LSTM delegada a InferenceWorker para no bloquear el render.
- Conversion OpenCV->Pygame optimizada (escala antes de cvtColor).
- Rediseno visual completo estilo HUD futurista.
"""

from __future__ import annotations

import threading
import cv2
import numpy as np
import pygame
from collections import deque

from scenes.base_scene import BaseScene
from core.config import (
    COLORS, WIDTH, HEIGHT, VOWEL_IMAGES, DETECTION_CONFIG,
    DYNAMIC_SIGNS, SIGN_DISPLAY_NAMES, LSTM_CONFIG,
)
from core.detector import detector
from core.lstm_detector import lstm_detector
from core.inference_worker import InferenceWorker
from core.draw_utils import draw_lstm_prob_bars

# Layout constants
_TAB_W, _TAB_H = 280, 60
_TAB_Y = 160
_TAB_VOWELS_X = WIDTH // 2 - _TAB_W - 10
_TAB_SIGNS_X  = WIDTH // 2 + 10
_BTN_W, _BTN_H = 220, 80
_BACK_RECT = pygame.Rect(50, HEIGHT - 100, 200, 60)


class LearningScene(BaseScene):
    """Escena de aprendizaje con soporte para vocales (YOLO) y senas dinamicas (LSTM)."""

    def __init__(self):
        super().__init__()

        # Estado general
        self.state      = "selection"
        self.active_tab = "vowels"
        self.sign_type  = "vowel"
        self.selected_sign = None
        self._loading_step   = 0
        self._loading_thread = None
        self._loading_done   = False

        # Camara
        self.cap = None
        self.cam_surface = None

        # Feedback
        self.feedback_msg   = ""
        self.feedback_color = COLORS['white']

        # Imagen de referencia (vocales)
        self.vowel_image = None

        # Inference Worker
        self._worker: InferenceWorker | None = None
        self._last_result: tuple = (None, 0.0, None)

        # Suavizado temporal YOLO (ventana de 5 frames)
        self._yolo_vote_buf: deque = deque(maxlen=5)

        # UI tabs
        self.tab_vowels_rect  = pygame.Rect(_TAB_VOWELS_X, _TAB_Y, _TAB_W, _TAB_H)
        self.tab_signs_rect   = pygame.Rect(_TAB_SIGNS_X,  _TAB_Y, _TAB_W, _TAB_H)
        self.tab_vowels_hover = False
        self.tab_signs_hover  = False

        # UI botones vocales
        btn_w, btn_h = 100, 100
        start_x = WIDTH // 2 - (btn_w * 5 + 40 * 4) // 2
        self.vowel_buttons = [
            {"label": v,
             "rect": pygame.Rect(start_x + i * (btn_w + 40), HEIGHT // 2, btn_w, btn_h),
             "hover": False}
            for i, v in enumerate(['A', 'E', 'I', 'O', 'U'])
        ]

        # UI botones senas dinamicas
        signs_data = [('hola', 'Hola'), ('hola_mundo', 'Hola\nMundo'), ('buenos_dias', 'Buenos\nDias')]
        total_w = len(signs_data) * _BTN_W + (len(signs_data) - 1) * 40
        sx = WIDTH // 2 - total_w // 2
        self.sign_buttons = [
            {"key": key, "label": label,
             "rect": pygame.Rect(sx + i * (_BTN_W + 40), HEIGHT // 2 - _BTN_H // 2, _BTN_W, _BTN_H),
             "hover": False}
            for i, (key, label) in enumerate(signs_data)
        ]

        self.back_hover = False

    # ── Eventos ───────────────────────────────────────────────────────────────

    def process_events(self, events):
        if self.state == "loading":
            return

        mouse = pygame.mouse.get_pos()
        self.back_hover = _BACK_RECT.collidepoint(mouse)

        if self.state == "selection":
            self.tab_vowels_hover = self.tab_vowels_rect.collidepoint(mouse)
            self.tab_signs_hover  = self.tab_signs_rect.collidepoint(mouse)
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

    # ── Arranque de practica ──────────────────────────────────────────────────

    def _start_vowel_practice(self, vowel):
        self.selected_sign = vowel
        self.sign_type     = "vowel"
        self._load_vowel_image(vowel)
        self._start_camera()
        self._start_worker("yolo")
        self.feedback_msg = ""
        self._yolo_vote_buf.clear()
        self.state = "practice"

    def _start_sign_loading(self, sign_key):
        self.selected_sign = sign_key
        self.sign_type     = "dynamic"
        self._loading_step = 0
        self.state         = "loading"

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self):
        if self.state == "loading":
            self._loading_step += 1
            if self._loading_step == 2 and self._loading_thread is None:
                self._loading_done = False

                def _load():
                    lstm_detector.initialize()
                    lstm_detector.reset()
                    self._loading_done = True

                self._loading_thread = threading.Thread(target=_load, daemon=True)
                self._loading_thread.start()

            if self._loading_done:
                self._loading_thread = None
                self._start_camera()
                self._start_worker("lstm")
                self.feedback_msg = ""
                self.state = "practice"
            return

        if self.state != "practice" or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        if self._worker is not None:
            self._worker.submit(frame)

        if self._worker is not None:
            self._last_result = self._worker.get_result()
            detected_class, detected_conf, _ = self._last_result
        else:
            detected_class, detected_conf = None, 0.0

        if self.sign_type == "vowel":
            self._update_yolo_feedback(detected_class, detected_conf)
        else:
            self._update_lstm_feedback(detected_class, detected_conf)
            if lstm_detector._initialized:
                lstm_detector.draw_landmarks(frame)

        small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LINEAR)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        if self.sign_type == "dynamic" and lstm_detector.buffer_progress >= 1.0:
            draw_lstm_prob_bars(rgb, lstm_detector.labels, lstm_detector.last_probs, bar_max_w=180)

        rgb_t = np.ascontiguousarray(rgb.transpose(1, 0, 2))
        surf  = pygame.surfarray.make_surface(rgb_t)
        self.cam_surface = pygame.transform.flip(surf, True, False)

    # ── Logica de feedback ─────────────────────────────────────────────────────

    def _update_yolo_feedback(self, detected_class, detected_conf):
        if detected_conf >= DETECTION_CONFIG['conf_threshold'] and detected_class:
            self._yolo_vote_buf.append(detected_class)
        else:
            self._yolo_vote_buf.append(None)

        valid = [c for c in self._yolo_vote_buf if c is not None]
        if not valid:
            self.feedback_msg   = "Buscando sena..."
            self.feedback_color = COLORS['warning']
            return

        smoothed = max(set(valid), key=valid.count)
        if smoothed.upper() == self.selected_sign.upper():
            self.feedback_msg   = "Correcto!"
            self.feedback_color = COLORS['success']
        else:
            self.feedback_msg   = "Intenta de nuevo"
            self.feedback_color = COLORS['error']

    def _update_lstm_feedback(self, detected_class, detected_conf):
        progress = lstm_detector.buffer_progress
        if progress < 1.0:
            pct = int(progress * 100)
            self.feedback_msg   = "Acumulando sena... {}%".format(pct)
            self.feedback_color = COLORS['warning']
            return

        if detected_class is None or detected_class == "no_sena" \
                or detected_conf < LSTM_CONFIG['confidence_threshold']:
            self.feedback_msg   = "No se detecto la sena"
            self.feedback_color = COLORS['warning']
        elif detected_class == self.selected_sign:
            self.feedback_msg   = "Correcto! ({:.0%})".format(detected_conf)
            self.feedback_color = COLORS['success']
        else:
            display = SIGN_DISPLAY_NAMES.get(detected_class, detected_class)
            self.feedback_msg   = "Parece '{}' - intenta de nuevo".format(display)
            self.feedback_color = COLORS['error']

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, screen):
        screen.fill(COLORS['background'])
        self.draw_scan_line(screen, (0, 0, WIDTH, HEIGHT), COLORS['primary'], speed=3.5, alpha=18)

        if self.state == "loading":
            self._draw_loading(screen)
            return

        self._draw_top_bar(screen)

        if self.state == "selection":
            self._draw_selection(screen)
        elif self.state == "practice":
            self._draw_practice(screen)

        self.draw_button(screen, _BACK_RECT, "<- VOLVER",
                         COLORS['primary'], self.back_hover, border_only=True)

    def _draw_top_bar(self, screen):
        lbl = self.font_label.render("HORUS v2.0", True, COLORS['light_gray'])
        screen.blit(lbl, (16, 14))
        mod = self.font_label.render("MODULO DE APRENDIZAJE - ACTIVO", True, COLORS['light_gray'])
        screen.blit(mod, (WIDTH - mod.get_width() - 16, 14))
        pygame.draw.line(screen, COLORS['dark_gray'], (0, 36), (WIDTH, 36), 1)

    def _draw_loading(self, screen):
        import time as _time
        sign_name = SIGN_DISPLAY_NAMES.get(self.selected_sign, self.selected_sign)

        card_w, card_h = 580, 260
        cx = WIDTH  // 2 - card_w // 2
        cy = HEIGHT // 2 - card_h // 2

        pygame.draw.rect(screen, COLORS['card_bg'], (cx, cy, card_w, card_h))
        pygame.draw.rect(screen, COLORS['primary'],  (cx, cy, card_w, card_h), 1)
        self.draw_corner_brackets(screen, (cx, cy, card_w, card_h),
                                  COLORS['primary'], size=22, thickness=2)
        self.draw_scan_line(screen, (cx, cy, card_w, card_h),
                            COLORS['primary'], speed=1.8, alpha=28)

        lbl = self.font_label.render("INICIALIZANDO DETECTOR", True, COLORS['light_gray'])
        screen.blit(lbl, (cx + 20, cy + 14))

        title = self.font_large.render("CARGANDO MODELO LSTM", True, COLORS['primary'])
        screen.blit(title, title.get_rect(centerx=WIDTH // 2, top=cy + 50))

        sub = self.font_small.render(
            "SENAL OBJETIVO:  {}".format(sign_name.upper()), True, COLORS['white'])
        screen.blit(sub, sub.get_rect(centerx=WIDTH // 2, top=cy + 110))

        if self.is_blink(2.0):
            dots = self.font_label.render("PROCESANDO...", True, COLORS['primary'])
            screen.blit(dots, dots.get_rect(centerx=WIDTH // 2, top=cy + 170))

        bar_w = 400
        bx = WIDTH // 2 - bar_w // 2
        by = cy + 210
        pygame.draw.rect(screen, COLORS['dark_gray'], (bx, by, bar_w, 6))
        fill = int(bar_w * ((_time.time() % 1.5) / 1.5))
        pygame.draw.rect(screen, COLORS['primary'], (bx, by, fill, 6))

    def _draw_selection(self, screen):
        title = self.font_large.render("SELECCIONA UNA SENAL", True, COLORS['primary'])
        screen.blit(title, title.get_rect(centerx=WIDTH // 2, top=55))

        pygame.draw.line(screen, COLORS['dark_gray'],
                         (WIDTH // 2 - 280, 100), (WIDTH // 2 + 280, 100), 1)

        self._draw_tab(screen, self.tab_vowels_rect, "01  VOCALES",
                       self.active_tab == "vowels", self.tab_vowels_hover)
        self._draw_tab(screen, self.tab_signs_rect,  "02  SENAS DINAMICAS",
                       self.active_tab == "signs",  self.tab_signs_hover)

        sep_y = _TAB_Y + _TAB_H + 28
        pygame.draw.line(screen, COLORS['dark_gray'], (80, sep_y), (WIDTH - 80, sep_y), 1)

        hint = self.font_label.render(
            "SELECCIONA UNA OPCION PARA INICIAR LA PRACTICA", True, COLORS['light_gray'])
        screen.blit(hint, hint.get_rect(centerx=WIDTH // 2, top=sep_y + 10))

        if self.active_tab == "vowels":
            self._draw_vowel_buttons(screen)
        else:
            self._draw_sign_buttons(screen)

    def _draw_tab(self, screen, rect, label, active, hover):
        if active:
            bg = (0, 42, 55)
            pygame.draw.rect(screen, bg, rect)
            pygame.draw.rect(screen, COLORS['primary'], rect, 1)
            self.draw_corner_brackets(screen,
                                      (rect.x, rect.y, rect.width, rect.height),
                                      COLORS['primary'], size=12, thickness=2)
            tc = COLORS['primary']
        else:
            bg = COLORS['card_bg'] if not hover else (20, 34, 56)
            pygame.draw.rect(screen, bg, rect)
            pygame.draw.rect(screen, COLORS['dark_gray'], rect, 1)
            tc = COLORS['light_gray'] if not hover else COLORS['white']

        text = self.font_small.render(label, True, tc)
        screen.blit(text, text.get_rect(center=rect.center))

    def _draw_vowel_buttons(self, screen):
        for btn in self.vowel_buttons:
            is_active = btn["hover"]
            bg = (0, 42, 55) if is_active else COLORS['card_bg']
            bc = COLORS['primary'] if is_active else COLORS['dark_gray']
            pygame.draw.rect(screen, bg,  btn["rect"])
            pygame.draw.rect(screen, bc,  btn["rect"], 1)
            self.draw_corner_brackets(screen,
                                      (btn["rect"].x, btn["rect"].y,
                                       btn["rect"].width, btn["rect"].height),
                                      COLORS['primary'] if is_active else COLORS['light_gray'],
                                      size=14, thickness=2)
            tc  = COLORS['primary'] if is_active else COLORS['white']
            lbl = self.font_large.render(btn["label"], True, tc)
            screen.blit(lbl, lbl.get_rect(center=btn["rect"].center))

    def _draw_sign_buttons(self, screen):
        for btn in self.sign_buttons:
            is_active = btn["hover"]
            bg = (28, 12, 60) if is_active else COLORS['card_bg']
            bc = COLORS['secondary'] if is_active else COLORS['dark_gray']
            pygame.draw.rect(screen, bg,  btn["rect"])
            pygame.draw.rect(screen, bc,  btn["rect"], 1)
            self.draw_corner_brackets(screen,
                                      (btn["rect"].x, btn["rect"].y,
                                       btn["rect"].width, btn["rect"].height),
                                      COLORS['secondary'] if is_active else COLORS['light_gray'],
                                      size=14, thickness=2)
            lines   = btn["label"].split("\n")
            line_h  = self.font_small.get_height()
            total_h = line_h * len(lines) + 8 * (len(lines) - 1)
            y0  = btn["rect"].centery - total_h // 2
            tc  = COLORS['secondary'] if is_active else COLORS['white']
            for j, line in enumerate(lines):
                surf = self.font_small.render(line, True, tc)
                screen.blit(surf, surf.get_rect(
                    centerx=btn["rect"].centerx, y=y0 + j * (line_h + 8)))

    def _draw_practice(self, screen):
        sign_name = SIGN_DISPLAY_NAMES.get(self.selected_sign, self.selected_sign)

        # Panel izquierdo: referencia
        ref_x, ref_y, ref_w, ref_h = WIDTH // 4 - 160, 110, 320, 320

        rlbl = self.font_label.render("SENAL DE REFERENCIA", True, COLORS['light_gray'])
        screen.blit(rlbl, (ref_x, ref_y - 22))

        if self.sign_type == "vowel" and self.vowel_image:
            screen.blit(self.vowel_image, (ref_x, ref_y))
        else:
            self._draw_placeholder(screen, pygame.Rect(ref_x, ref_y, ref_w, ref_h))

        self.draw_corner_brackets(screen,
                                  (ref_x, ref_y, ref_w, ref_h),
                                  COLORS['primary'], size=22, thickness=2)
        self.draw_scan_line(screen,
                            (ref_x, ref_y, ref_w, ref_h),
                            COLORS['primary'], speed=4.0, alpha=14)

        sn_lbl = self.font_medium.render(sign_name.upper(), True, COLORS['primary'])
        screen.blit(sn_lbl, sn_lbl.get_rect(
            centerx=ref_x + ref_w // 2, top=ref_y + ref_h + 12))

        # Panel derecho: camara
        cam_cx = 3 * WIDTH // 4
        cam_cy = HEIGHT // 2 - 20
        cam_w, cam_h = 320, 240

        clbl = self.font_label.render("CAMARA EN VIVO", True, COLORS['light_gray'])
        screen.blit(clbl, clbl.get_rect(
            centerx=cam_cx, bottom=cam_cy - cam_h // 2 - 8))

        cam_bg = pygame.Rect(cam_cx - cam_w // 2 - 4,
                             cam_cy - cam_h // 2 - 4,
                             cam_w + 8, cam_h + 8)
        pygame.draw.rect(screen, COLORS['card_bg'], cam_bg)
        pygame.draw.rect(screen, COLORS['dark_gray'], cam_bg, 1)

        if self.cam_surface:
            screen.blit(self.cam_surface, (cam_cx - cam_w // 2, cam_cy - cam_h // 2))
        else:
            no_cam = self.font_label.render("SIN SENAL", True, COLORS['light_gray'])
            screen.blit(no_cam, no_cam.get_rect(center=(cam_cx, cam_cy)))

        self.draw_scan_line(screen,
                            (cam_cx - cam_w // 2, cam_cy - cam_h // 2, cam_w, cam_h),
                            COLORS['primary'], speed=2.0, alpha=22)
        self.draw_corner_brackets(screen,
                                  (cam_cx - cam_w // 2, cam_cy - cam_h // 2, cam_w, cam_h),
                                  COLORS['primary'], size=22, thickness=2)

        if self.is_blink(1.4):
            rec = self.font_label.render("REC", True, COLORS['error'])
            screen.blit(rec, (cam_cx - cam_w // 2 + 8, cam_cy - cam_h // 2 + 8))

        if self.is_blink(0.9):
            ana = self.font_label.render("ANALIZANDO...", True, COLORS['primary'])
            screen.blit(ana, ana.get_rect(
                right=cam_cx + cam_w // 2 - 8,
                top=cam_cy - cam_h // 2 + 8))

        if self.sign_type == "dynamic":
            self._draw_lstm_progress(screen,
                                     cx=cam_cx,
                                     by=cam_cy + cam_h // 2 + 20)

        # Feedback
        fb_y = HEIGHT - 120
        fb_surf = self.font_medium.render(self.feedback_msg, True, self.feedback_color)
        fb_rect = fb_surf.get_rect(centerx=WIDTH // 2, centery=fb_y)
        screen.blit(fb_surf, fb_rect)
        pygame.draw.line(screen, self.feedback_color,
                         (fb_rect.left - 20, fb_rect.bottom + 4),
                         (fb_rect.right + 20, fb_rect.bottom + 4), 1)

    def _draw_placeholder(self, screen, rect):
        pygame.draw.rect(screen, COLORS['card_bg'], rect)
        pygame.draw.rect(screen, COLORS['dark_gray'], rect, 1)
        lbl = self.font_label.render("VIDEO", True, COLORS['dark_gray'])
        sub = self.font_label.render("PROXIMAMENTE", True, COLORS['dark_gray'])
        screen.blit(lbl, lbl.get_rect(centerx=rect.centerx, centery=rect.centery - 14))
        screen.blit(sub, sub.get_rect(centerx=rect.centerx, centery=rect.centery + 14))
        mx, my = rect.centerx, rect.centery
        pygame.draw.line(screen, COLORS['dark_gray'], (mx - 30, my - 40), (mx + 30, my - 40), 1)
        pygame.draw.line(screen, COLORS['dark_gray'], (mx, my - 55), (mx, my - 25), 1)

    def _draw_lstm_progress(self, screen, cx=None, by=None):
        progress = lstm_detector.buffer_progress
        if cx is None:
            cx = WIDTH // 2
        if by is None:
            by = HEIGHT - 220

        bar_w = 320
        bx    = cx - bar_w // 2
        bar_h = 10

        lbl = self.font_label.render(
            "BUFFER DE SENA  {:3d}%".format(int(progress * 100)),
            True, COLORS['light_gray'])
        screen.blit(lbl, lbl.get_rect(centerx=cx, bottom=by - 4))

        pygame.draw.rect(screen, COLORS['dark_gray'], (bx, by, bar_w, bar_h))
        if progress > 0:
            color = COLORS['primary'] if progress < 1.0 else COLORS['success']
            pygame.draw.rect(screen, color, (bx, by, int(bar_w * progress), bar_h))
        pygame.draw.rect(screen, COLORS['light_gray'], (bx, by, bar_w, bar_h), 1)

        for i in range(1, 4):
            tx = bx + int(bar_w * i / 4)
            pygame.draw.line(screen, COLORS['background'],
                             (tx, by + 1), (tx, by + bar_h - 1), 1)

    # ── Helpers internos ──────────────────────────────────────────────────────

    _ALLOWED_IMAGE_EXTS = {'.jpeg', '.jpg', '.png', '.bmp', '.gif'}
    _MAX_IMAGE_BYTES    = 10 * 1024 * 1024

    def _load_vowel_image(self, vowel):
        from pathlib import Path as _Path
        path = VOWEL_IMAGES.get(vowel)
        if not path:
            self.vowel_image = None
            return
        p = _Path(path)
        if not p.exists() or p.suffix.lower() not in self._ALLOWED_IMAGE_EXTS:
            self.vowel_image = None
            return
        if p.stat().st_size > self._MAX_IMAGE_BYTES:
            self.vowel_image = None
            return
        try:
            img = pygame.image.load(str(p))
            self.vowel_image = pygame.transform.scale(img, (300, 300))
        except pygame.error:
            self.vowel_image = None

    def _start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                self.cap = None
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def _start_worker(self, mode):
        if self._worker is not None:
            self._worker.stop()
        self._worker = InferenceWorker()
        self._worker.start(mode)
        self._last_result = (None, 0.0, None)

    def _stop_worker(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    def _handle_back(self):
        if self.state == "practice":
            if self.sign_type == "dynamic":
                lstm_detector.reset()
            self._stop_camera()
            self._stop_worker()
            self.cam_surface  = None
            self.feedback_msg = ""
            self.state        = "selection"
        else:
            self._stop_camera()
            self._stop_worker()
            from scenes.menu_scene import MenuScene
            self.switch_to(MenuScene)

    def __del__(self):
        self._stop_camera()
        self._stop_worker()
