"""
game.py - Juego Interactivo de Lenguaje de Señas — v3 (visual futurista).

Cambios v3:
- Diseño HUD/sci-fi completo: tipografía Orbitron, paleta cyan-purple,
  corner brackets, scan line animada, barras de confianza IA en tiempo real.
- Lógica y arquitectura de threading de v2 conservadas intactas.
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
from core.inference_worker import InferenceWorker
from core.config import COLORS, DETECTION_CONFIG, DYNAMIC_SIGNS, SIGN_DISPLAY_NAMES, LSTM_CONFIG, UI_CONFIG
from core.user_manager import user_manager
from core.draw_utils import (draw_lstm_prob_bars, draw_corner_brackets,
                              draw_scan_line, is_blink_visible, draw_hud_label)

WINDOW_WIDTH  = 1200
WINDOW_HEIGHT = 800


# ── Fuentes (compartidas entre helpers de módulo) ──────────────────────────────

def _load_font(path, size):
    if path:
        try:
            return pygame.font.Font(path, size)
        except Exception:
            pass
    return pygame.font.Font(None, size + 10)

_bold    = UI_CONFIG['font_bold']
_regular = UI_CONFIG['font_regular']

font_title  = _load_font(_bold,    UI_CONFIG['font_size_title'])
font_large  = _load_font(_bold,    UI_CONFIG['font_size_large'])
font_medium = _load_font(_regular, UI_CONFIG['font_size_medium'])
font_small  = _load_font(_regular, UI_CONFIG['font_size_small'])
font_label  = _load_font(_regular, UI_CONFIG['font_size_label'])


# ── Partículas ─────────────────────────────────────────────────────────────────

class ParticleEffect:
    def __init__(self):
        self.particles = []

    def add_success_particles(self, x, y):
        for _ in range(20):
            self.particles.append({
                'x': x, 'y': y,
                'vx': random.uniform(-5, 5), 'vy': random.uniform(-8, -2),
                'life': 60,
                'color': random.choice([COLORS['success'], COLORS['primary'], COLORS['secondary']]),
                'size': random.uniform(3, 8),
            })

    def add_error_particles(self, x, y):
        for _ in range(15):
            self.particles.append({
                'x': x, 'y': y,
                'vx': random.uniform(-3, 3), 'vy': random.uniform(-5, -1),
                'life': 40, 'color': COLORS['error'], 'size': random.uniform(2, 5),
            })

    def update(self):
        for p in self.particles:
            p['x'] += p['vx']; p['y'] += p['vy']
            p['vy'] += 0.2; p['life'] -= 1; p['size'] *= 0.98
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] >= 1]

    _surf = None

    def draw(self, surface):
        if ParticleEffect._surf is None:
            ParticleEffect._surf = pygame.Surface((20, 20), pygame.SRCALPHA)
        ps = ParticleEffect._surf
        for p in self.particles:
            alpha = max(0, int(p['life'] / 60.0 * 255))
            r = max(1, int(p['size']))
            ps.fill((0, 0, 0, 0))
            pygame.draw.circle(ps, (*p['color'], alpha), (r, r), r)
            surface.blit(ps, (int(p['x']) - r, int(p['y']) - r))


# ── Helpers de dibujado ────────────────────────────────────────────────────────

_cam_surf_cache: dict = {}


def opencv_to_pygame(cv_image, target_w: int, target_h: int) -> pygame.Surface:
    """Escala → cvtColor → blit_array (optimizado). Flip horizontal para espejo natural."""
    small = cv2.resize(cv_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb_t = np.ascontiguousarray(rgb.transpose(1, 0, 2))
    key   = (target_w, target_h)
    if key not in _cam_surf_cache:
        _cam_surf_cache[key] = pygame.Surface((target_w, target_h))
    surf = _cam_surf_cache[key]
    pygame.surfarray.blit_array(surf, rgb_t)
    return pygame.transform.flip(surf, True, False)


def _draw_conf_bar(surface, x, y, w, h, label, pct, color):
    """Barra de confianza individual estilo HUD."""
    lbl  = font_label.render(label, True, COLORS['white'])
    pct_surf = font_label.render(f"{int(pct * 100)}%", True, color)
    surface.blit(lbl,      (x, y))
    surface.blit(pct_surf, (x + w - pct_surf.get_width(), y))
    by = y + lbl.get_height() + 2
    pygame.draw.rect(surface, COLORS['card_bg'],   (x, by, w, h))
    pygame.draw.rect(surface, COLORS['dark_gray'], (x, by, w, h), 1)
    filled = int(w * min(1.0, pct))
    if filled > 0:
        pygame.draw.rect(surface, color, (x, by, filled, h))


# ── Juego principal ────────────────────────────────────────────────────────────

class SignLanguageGame:
    """Lógica completa del juego. Sin bucle propio — integrado con BaseScene."""

    def __init__(self):
        from core.logger import get_logger
        _log = get_logger("game")
        _log.info("Detector en: %s", detector.device)
        self.model = detector.model

        self.all_signs   = ['A', 'E', 'I', 'O', 'U'] + DYNAMIC_SIGNS
        self.puntuacion  = 0
        self.mejor_puntuacion = user_manager.get_best_score()
        self.tiempo_preparacion = 5.0
        self.vocal_actual = random.choice(self.all_signs)
        self.mensaje_feedback = ""
        self.color_feedback   = COLORS['success']
        self.tiempo_inicio    = time.time()
        self.evaluado         = False

        self.detections_deque  = deque(maxlen=DETECTION_CONFIG['detections_buffer'])
        self.feedback_start    = None
        self.feedback_duracion = 3.0
        self._lstm_frame_count = 0

        if self.vocal_actual in DYNAMIC_SIGNS:
            lstm_detector.initialize()
            lstm_detector.reset()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._current_frame: np.ndarray | None = None
        self._frame_for_draw: np.ndarray | None = None
        self._frame_count  = 0
        self._last_result  = (None, 0.0, None)

        is_dynamic = self.vocal_actual in DYNAMIC_SIGNS
        self._worker = InferenceWorker()
        self._worker.start('lstm' if is_dynamic else 'yolo')

        self.particles       = ParticleEffect()
        self.pulse_time      = 0.0
        self.shake_intensity = 0
        self.shake_duration  = 0
        self._tiempo_restante = self.tiempo_preparacion
        self.should_exit     = False
        self._clock          = pygame.time.Clock()

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def handle_events(self, events) -> bool:
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.should_exit = True
                elif event.key == pygame.K_r:
                    self._reset_round()
        return self.should_exit

    def update(self):
        self._clock.tick()

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self._current_frame = frame

        is_dynamic = self.vocal_actual in DYNAMIC_SIGNS
        self._frame_count += 1
        skip = LSTM_CONFIG['skip_frames'] if is_dynamic else DETECTION_CONFIG['skip_frames']
        if self._frame_count % skip == 0 and self._current_frame is not None:
            new_mode = 'lstm' if is_dynamic else 'yolo'
            if self._worker._mode != new_mode:
                self._worker.set_mode(new_mode)
            self._worker.submit(self._current_frame)

        self._last_result = self._worker.get_result()
        detected_class, detected_conf, detected_box = self._last_result

        if self._current_frame is not None:
            if is_dynamic and lstm_detector._initialized:
                fc = self._current_frame.copy()
                lstm_detector.draw_landmarks(fc)
                self._frame_for_draw = fc
            else:
                self._frame_for_draw = self._current_frame
        else:
            self._frame_for_draw = None

        self._tiempo_restante = self._update_game_logic(detected_class, detected_conf)

    def draw(self, surface: pygame.Surface):
        surface.fill(COLORS['background'])
        self._draw_ui(surface, self._tiempo_restante)
        dc, conf, box = self._last_result
        self._draw_camera_feed(surface, self._frame_for_draw, box, dc, conf)

    def cleanup(self):
        self._worker.stop()
        if self.cap:
            self.cap.release()
        lstm_detector.close()
        user_manager.update_stats(self.puntuacion)

    # ── Lógica interna ────────────────────────────────────────────────────────

    def _reset_round(self):
        self.puntuacion    = 0
        self.vocal_actual  = random.choice(self.all_signs)
        if self.vocal_actual in DYNAMIC_SIGNS:
            lstm_detector.initialize(); lstm_detector.reset()
            self._lstm_frame_count = 0
        self.tiempo_inicio    = time.time()
        self.detections_deque.clear()
        self.evaluado         = False
        self.feedback_start   = None
        self.mensaje_feedback = ""

    def add_screen_shake(self, intensity=10, duration=500):
        self.shake_intensity = intensity
        self.shake_duration  = duration

    def get_screen_offset(self):
        if self.shake_duration > 0:
            self.shake_duration -= self._clock.get_time()
            return (random.randint(-self.shake_intensity, self.shake_intensity),
                    random.randint(-self.shake_intensity, self.shake_intensity))
        return 0, 0

    def _update_game_logic(self, detected_class, detected_conf) -> float:
        is_dynamic = self.vocal_actual in DYNAMIC_SIGNS
        threshold  = LSTM_CONFIG['confidence_threshold'] if is_dynamic else DETECTION_CONFIG['conf_threshold']

        if detected_class and detected_conf >= threshold and detected_class != "no_sena":
            self.detections_deque.append(detected_class)
        else:
            self.detections_deque.append(None)

        elapsed  = time.time() - self.tiempo_inicio
        restante = max(0.0, self.tiempo_preparacion - elapsed)
        sign_display = SIGN_DISPLAY_NAMES.get(self.vocal_actual, self.vocal_actual)

        if restante > 0:
            self.evaluado = False
            if is_dynamic and lstm_detector.buffer_progress < 1.0:
                pct = int(lstm_detector.buffer_progress * 100)
                self.mensaje_feedback = f"ACUMULANDO SENA... {pct}%"
            else:
                self.mensaje_feedback = "PREPARATE -- HAZ LA SENA"
            self.color_feedback = COLORS['warning']
        else:
            if not self.evaluado:
                validos   = [d for d in self.detections_deque if d is not None]
                min_votes = LSTM_CONFIG['min_votes'] if is_dynamic else DETECTION_CONFIG['min_votes']

                if not validos:
                    self.mensaje_feedback = f"[X]  NO SE DETECTO LA SENA"
                    self.color_feedback   = COLORS['error']
                    self.particles.add_error_particles(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
                    self.add_screen_shake(5, 300)
                else:
                    counts = Counter(validos)
                    top_class, top_count = counts.most_common(1)[0]
                    correct = (top_class == self.vocal_actual if is_dynamic
                               else top_class.upper() == self.vocal_actual.upper())
                    if correct and top_count >= min_votes:
                        self.puntuacion += 1
                        self.mejor_puntuacion = max(self.mejor_puntuacion, self.puntuacion)
                        self.mensaje_feedback = f"[OK]  CORRECTO -- '{sign_display}'"
                        self.color_feedback   = COLORS['success']
                        self.particles.add_success_particles(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
                    else:
                        self.mensaje_feedback = f"[X]  INCORRECTO -- ERA '{sign_display}'"
                        self.color_feedback   = COLORS['error']
                        self.particles.add_error_particles(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
                        self.add_screen_shake(8, 400)

                self.evaluado       = True
                self.feedback_start = time.time()

        if self.evaluado and self.feedback_start:
            if time.time() - self.feedback_start > self.feedback_duracion:
                self.vocal_actual = random.choice(self.all_signs)
                if self.vocal_actual in DYNAMIC_SIGNS:
                    lstm_detector.initialize(); lstm_detector.reset()
                    self._lstm_frame_count = 0
                    self._worker.set_mode('lstm')
                else:
                    self._worker.set_mode('yolo')
                self.tiempo_inicio    = time.time()
                self.mensaje_feedback = ""
                self.evaluado         = False
                self.feedback_start   = None
                self.detections_deque.clear()
                self._last_result     = (None, 0.0, None)

        return restante

    # ── Cámara ────────────────────────────────────────────────────────────────

    def _draw_camera_feed(self, surface, frame, detected_box, detected_class, detected_conf):
        CAM_W, CAM_H = 440, 330
        CAM_X = WINDOW_WIDTH - CAM_W - 20
        CAM_Y = 50

        # ── Panel de cámara ───────────────────────────────────────────────────
        panel_rect = (CAM_X - 14, CAM_Y - 14, CAM_W + 28, CAM_H + 28)
        pygame.draw.rect(surface, COLORS['card_bg'], panel_rect)
        pygame.draw.rect(surface, COLORS['dark_gray'], panel_rect, 1)

        if frame is not None:
            # Bounding box en el frame (si hay detección YOLO)
            if detected_box is not None:
                frame = frame.copy()
                x1, y1, x2, y2 = detected_box
                bc = (0, 230, 118) if detected_conf >= DETECTION_CONFIG['conf_threshold'] else (255, 193, 7)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bc, 2)
                lbl = f"{detected_class} {detected_conf:.0%}"
                (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), bc, -1)
                cv2.putText(frame, lbl, (x1 + 2, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (8, 12, 24), 1)

            cam_surf = opencv_to_pygame(frame, CAM_W, CAM_H)
            surface.blit(cam_surf, (CAM_X, CAM_Y))
        else:
            # Placeholder si no hay cámara
            pygame.draw.rect(surface, COLORS['background'], (CAM_X, CAM_Y, CAM_W, CAM_H))
            no_cam = font_label.render("SIN SEÑAL DE CÁMARA", True, COLORS['light_gray'])
            surface.blit(no_cam, no_cam.get_rect(center=(CAM_X + CAM_W // 2, CAM_Y + CAM_H // 2)))

        # Scan line animada sobre el feed
        draw_scan_line(surface, (CAM_X, CAM_Y, CAM_W, CAM_H), COLORS['primary'], speed=1.8, alpha=35)

        # Corner brackets del targeting reticle
        draw_corner_brackets(surface, (CAM_X, CAM_Y, CAM_W, CAM_H), COLORS['primary'], size=22, thickness=2)

        # Etiquetas HUD sobre el feed
        if is_blink_visible(1.5):
            scan_lbl = font_label.render("ANALIZANDO...", True, COLORS['primary'])
            surface.blit(scan_lbl, (CAM_X + 8, CAM_Y + CAM_H - scan_lbl.get_height() - 8))
        rec_lbl = font_label.render("REC", True, COLORS['success'])
        rec_x   = CAM_X + CAM_W - rec_lbl.get_width() - 22
        rec_y   = CAM_Y + CAM_H - rec_lbl.get_height() - 8
        pygame.draw.circle(surface, COLORS['error'],
                           (rec_x - 8, rec_y + rec_lbl.get_height() // 2), 4)
        surface.blit(rec_lbl, (rec_x, rec_y))

        # ── Barras de confianza IA ────────────────────────────────────────────
        is_dynamic = self.vocal_actual in DYNAMIC_SIGNS
        bx  = CAM_X
        by  = CAM_Y + CAM_H + 20
        bw  = CAM_W

        mod_lbl = font_label.render("MÓDULO VISUAL · IA", True, COLORS['light_gray'])
        surface.blit(mod_lbl, (bx, by))
        by += mod_lbl.get_height() + 8

        if is_dynamic and lstm_detector._initialized and len(lstm_detector.last_probs) > 0:
            bar_colors = {
                'hola':        COLORS['primary'],
                'hola_mundo':  COLORS['secondary'],
                'buenos_dias': COLORS['warning'],
                'no_sena':     COLORS['error'],
            }
            labels = lstm_detector.labels
            probs  = lstm_detector.last_probs
            bar_h  = 6
            row_h  = font_label.get_height() + bar_h + 10
            for i, (lbl, prob) in enumerate(zip(labels, probs)):
                display = SIGN_DISPLAY_NAMES.get(lbl, lbl.upper())
                col     = bar_colors.get(lbl, COLORS['white'])
                _draw_conf_bar(surface, bx, by + i * row_h, bw, bar_h, display, float(prob), col)
        else:
            # YOLO: solo la confianza del top-1
            if detected_class and detected_conf > 0:
                col = COLORS['success'] if detected_conf >= DETECTION_CONFIG['conf_threshold'] else COLORS['warning']
                _draw_conf_bar(surface, bx, by, bw, 6, detected_class.upper(), detected_conf, col)
            else:
                no_det = font_label.render("SIN DETECCIÓN", True, COLORS['light_gray'])
                surface.blit(no_det, (bx, by))

    # ── UI principal ──────────────────────────────────────────────────────────

    def _draw_ui(self, surface, tiempo_restante):
        ox, oy = self.get_screen_offset()

        # Scan line de fondo (global, muy sutil)
        draw_scan_line(surface, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT),
                       COLORS['primary'], speed=4.0, alpha=12)

        # HUD bar superior
        top_lbl = font_label.render("HORUS · MODO JUEGO", True, COLORS['light_gray'])
        surface.blit(top_lbl, (16 + ox, 14 + oy))
        pygame.draw.line(surface, COLORS['dark_gray'], (0, 36), (WINDOW_WIDTH, 36), 1)

        # ── Panel izquierdo ───────────────────────────────────────────────────
        PNL_X, PNL_Y, PNL_W, PNL_H = 16 + ox, 50 + oy, 680, 720
        pygame.draw.rect(surface, COLORS['card_bg'], (PNL_X, PNL_Y, PNL_W, PNL_H))
        pygame.draw.rect(surface, COLORS['dark_gray'], (PNL_X, PNL_Y, PNL_W, PNL_H), 1)
        draw_corner_brackets(surface, (PNL_X, PNL_Y, PNL_W, PNL_H),
                             COLORS['dark_gray'], size=14, thickness=1)

        # Label "OBJETIVO DE RONDA"
        obj_lbl = font_label.render("OBJETIVO DE RONDA", True, COLORS['light_gray'])
        surface.blit(obj_lbl, (PNL_X + 16, PNL_Y + 14))

        # ── Carta de la seña ──────────────────────────────────────────────────
        is_dynamic   = self.vocal_actual in DYNAMIC_SIGNS
        sign_display = SIGN_DISPLAY_NAMES.get(self.vocal_actual, self.vocal_actual)
        card_color   = COLORS['secondary'] if is_dynamic else COLORS['primary']

        CARD_X, CARD_Y, CARD_W, CARD_H = PNL_X + 16, PNL_Y + 38, 340, 230
        pygame.draw.rect(surface, COLORS['background'], (CARD_X, CARD_Y, CARD_W, CARD_H))
        pygame.draw.rect(surface, card_color, (CARD_X, CARD_Y, CARD_W, CARD_H), 1)
        draw_corner_brackets(surface, (CARD_X, CARD_Y, CARD_W, CARD_H),
                             card_color, size=18, thickness=2)

        # Tipo de seña
        tipo = "SEÑA DINÁMICA" if is_dynamic else "VOCAL ESTÁTICA"
        tipo_surf = font_label.render(tipo, True, COLORS['light_gray'])
        surface.blit(tipo_surf, tipo_surf.get_rect(centerx=CARD_X + CARD_W // 2, top=CARD_Y + 12))

        # Letra / nombre con pulso
        self.pulse_time += 0.1
        pulse_scale = 1 + 0.07 * math.sin(self.pulse_time)

        if is_dynamic:
            words  = sign_display.split()
            fnt    = font_large if len(sign_display) > 8 else font_title
            cy_txt = CARD_Y + CARD_H // 2 - (len(words) - 1) * 22
            for w in words:
                ws = fnt.render(w, True, card_color)
                surface.blit(ws, ws.get_rect(centerx=CARD_X + CARD_W // 2, centery=cy_txt))
                cy_txt += fnt.get_height() - 8
        else:
            ls = font_title.render(sign_display, True, card_color)
            ls = pygame.transform.scale(
                ls, (int(ls.get_width() * pulse_scale), int(ls.get_height() * pulse_scale)))
            surface.blit(ls, ls.get_rect(
                center=(CARD_X + CARD_W // 2, CARD_Y + CARD_H // 2 + 10)))

        # Instrucción
        instr = "REALIZA ESTA SEÑA:" if is_dynamic else "HAZ LA SEÑA:"
        i_surf = font_label.render(instr, True, COLORS['light_gray'])
        surface.blit(i_surf, i_surf.get_rect(centerx=CARD_X + CARD_W // 2, bottom=CARD_Y - 4))

        # ── 3 cajas HUD: PUNTOS / MEJOR / TIEMPO ─────────────────────────────
        BOX_Y  = CARD_Y
        BOX_X0 = CARD_X + CARD_W + 16
        BOX_W  = (PNL_W - CARD_W - 48) // 3
        BOX_H  = 90

        score_str = f"{self.puntuacion:02d}"
        best_str  = f"{self.mejor_puntuacion:02d}"
        time_str  = f"{int(tiempo_restante + 1)}s" if tiempo_restante > 0 else "0s"

        boxes = [
            ("PUNTOS",  score_str, COLORS['primary']),
            ("MEJOR",   best_str,  COLORS['secondary']),
            ("TIEMPO",  time_str,  COLORS['warning'] if tiempo_restante > 0 else COLORS['error']),
        ]
        for i, (lbl, val, col) in enumerate(boxes):
            draw_hud_label(surface, font_label, font_large,
                           (BOX_X0 + i * (BOX_W + 8), BOX_Y, BOX_W, BOX_H),
                           lbl, val, col)

        # ── Barra de temporizador ─────────────────────────────────────────────
        # TB_Y debe quedar debajo de la carta (CARD_H=230) no de las cajas (BOX_H=90)
        TB_Y  = CARD_Y + CARD_H + 14
        TB_X  = PNL_X + 16
        TB_W  = PNL_W - 32
        TB_H  = 8
        progress = 1 - (tiempo_restante / self.tiempo_preparacion) if tiempo_restante > 0 else 1.0
        pygame.draw.rect(surface, COLORS['background'],  (TB_X, TB_Y, TB_W, TB_H))
        pygame.draw.rect(surface, COLORS['dark_gray'],   (TB_X, TB_Y, TB_W, TB_H), 1)
        if progress > 0:
            t_col = COLORS['warning'] if tiempo_restante > 1.5 else COLORS['error']
            pygame.draw.rect(surface, t_col, (TB_X, TB_Y, int(TB_W * progress), TB_H))

        # ── Barra buffer LSTM ─────────────────────────────────────────────────
        if is_dynamic and tiempo_restante > 0:
            buf = lstm_detector.buffer_progress
            LB_Y = TB_Y + TB_H + 16
            lb_lbl = font_label.render("BUFFER DE SEÑA", True, COLORS['light_gray'])
            surface.blit(lb_lbl, (TB_X, LB_Y))
            LBY2 = LB_Y + lb_lbl.get_height() + 4
            pygame.draw.rect(surface, COLORS['background'], (TB_X, LBY2, TB_W, 6))
            pygame.draw.rect(surface, COLORS['dark_gray'],  (TB_X, LBY2, TB_W, 6), 1)
            if buf > 0:
                pygame.draw.rect(surface, COLORS['secondary'],
                                 (TB_X, LBY2, int(TB_W * buf), 6))

        # ── Mensaje de feedback ───────────────────────────────────────────────
        if self.mensaje_feedback:
            FB_Y = TB_Y + TB_H + (60 if is_dynamic else 30)
            fb_surf = font_medium.render(self.mensaje_feedback, True, self.color_feedback)
            fx = PNL_X + PNL_W // 2 - fb_surf.get_width() // 2
            # Borde de color bajo el texto
            pygame.draw.line(surface, self.color_feedback,
                             (fx, FB_Y + fb_surf.get_height() + 4),
                             (fx + fb_surf.get_width(), FB_Y + fb_surf.get_height() + 4), 1)
            surface.blit(fb_surf, (fx, FB_Y))

        # ── Barra de señas ────────────────────────────────────────────────────
        SIGN_Y = PNL_Y + PNL_H - 72
        pygame.draw.line(surface, COLORS['dark_gray'],
                         (PNL_X + 8, SIGN_Y - 8), (PNL_X + PNL_W - 8, SIGN_Y - 8), 1)

        vowel_lbl = font_label.render("VOCALES", True, COLORS['light_gray'])
        surface.blit(vowel_lbl, (PNL_X + 16, SIGN_Y - vowel_lbl.get_height() - 2))

        vow_sz = 44
        for i, v in enumerate(['A', 'E', 'I', 'O', 'U']):
            vx = PNL_X + 16 + i * (vow_sz + 8)
            is_active = v == self.vocal_actual
            col = COLORS['primary'] if is_active else COLORS['dark_gray']
            pygame.draw.rect(surface, COLORS['background'], (vx, SIGN_Y, vow_sz, vow_sz))
            pygame.draw.rect(surface, col, (vx, SIGN_Y, vow_sz, vow_sz), 2 if is_active else 1)
            vt = font_medium.render(v, True, col)
            surface.blit(vt, vt.get_rect(center=(vx + vow_sz // 2, SIGN_Y + vow_sz // 2)))

        dyn_lbl = font_label.render("SEÑAS DINÁMICAS", True, COLORS['light_gray'])
        surface.blit(dyn_lbl, (PNL_X + 16, SIGN_Y + vow_sz + 4))

        dyn_sz_w, dyn_sz_h = 110, 26
        for i, (key, short) in enumerate([('hola','HOLA'),('hola_mundo','H.MUNDO'),('buenos_dias','B.DÍAS')]):
            dx = PNL_X + 16 + i * (dyn_sz_w + 8)
            dy = SIGN_Y + vow_sz + dyn_lbl.get_height() + 8
            is_act = key == self.vocal_actual
            col    = COLORS['secondary'] if is_act else COLORS['dark_gray']
            pygame.draw.rect(surface, COLORS['background'], (dx, dy, dyn_sz_w, dyn_sz_h))
            pygame.draw.rect(surface, col, (dx, dy, dyn_sz_w, dyn_sz_h), 2 if is_act else 1)
            dt = font_label.render(short, True, col)
            surface.blit(dt, dt.get_rect(center=(dx + dyn_sz_w // 2, dy + dyn_sz_h // 2)))

        # ── Jugador + hints ───────────────────────────────────────────────────
        pname = font_label.render(f"AGENTE: {user_manager.get_user_name()}", True, COLORS['secondary'])
        surface.blit(pname, (PNL_X + 16, PNL_Y + PNL_H - pname.get_height() - 4))

        hints = font_label.render("[ESC] SALIR   [R] REINICIAR", True, COLORS['light_gray'])
        surface.blit(hints, (PNL_X + PNL_W - hints.get_width() - 8,
                              PNL_Y + PNL_H - hints.get_height() - 4))

        # ── Partículas ────────────────────────────────────────────────────────
        self.particles.update()
        self.particles.draw(surface)
