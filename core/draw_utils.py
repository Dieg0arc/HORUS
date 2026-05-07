"""core/draw_utils.py — Utilidades de dibujado compartidas entre game.py y learning_scene.py."""

import cv2
import numpy as np


# ── Constantes de landmarks ──────────────────────────────────────────────────
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
_HAND_KEY_PTS = [0, 4, 8, 12, 16, 20]  # muñeca + puntas de dedos

_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]

_FACE_KEY = [33, 133, 362, 263, 159, 386, 145, 374, 1, 4, 61, 291, 13, 14, 10, 152, 234, 454]
_FACE_CONNECTIONS = [
    (33, 133), (133, 159), (159, 145), (145, 33),
    (362, 263), (263, 386), (386, 374), (374, 362),
    (61, 13), (13, 291),
    (234, 10), (10, 454),
]


def draw_landmarks_bgr(frame: np.ndarray, pose_result, face_result, hand_result) -> None:
    """Dibuja landmarks reducidos (manos, cara clave, esqueleto pose) sobre un frame BGR in-place."""
    h, w = frame.shape[:2]

    def pt(lm):
        return int(lm.x * w), int(lm.y * h)

    if hand_result and hand_result.hand_landmarks:
        for hand_lms in hand_result.hand_landmarks:
            for s, e in _HAND_CONNECTIONS:
                cv2.line(frame, pt(hand_lms[s]), pt(hand_lms[e]), (200, 80, 220), 2)
            for idx in _HAND_KEY_PTS:
                cv2.circle(frame, pt(hand_lms[idx]), 4, (100, 100, 255), -1)

    if face_result and face_result.face_landmarks:
        flms = face_result.face_landmarks[0]
        for s, e in _FACE_CONNECTIONS:
            if s < len(flms) and e < len(flms):
                cv2.line(frame, pt(flms[s]), pt(flms[e]), (160, 80, 200), 1)
        for idx in _FACE_KEY:
            if idx < len(flms):
                cv2.circle(frame, pt(flms[idx]), 2, (150, 100, 255), -1)

    if pose_result and pose_result.pose_landmarks:
        plms = pose_result.pose_landmarks[0]
        for s, e in _POSE_CONNECTIONS:
            if s < len(plms) and e < len(plms):
                cv2.line(frame, pt(plms[s]), pt(plms[e]), (200, 80, 220), 2)

_PROB_COLORS_BGR = {
    'buenos_dias': (100, 200, 255),
    'hola':        (50, 200, 100),
    'hola_mundo':  (150, 100, 255),
    'no_sena':     (255, 100, 100),
}

_PROB_LABELS_DISPLAY = {
    'buenos_dias': 'B.Dias',
    'hola':        'Hola',
    'hola_mundo':  'H.Mundo',
    'no_sena':     'No seña',
}


def draw_lstm_prob_bars(frame, labels, probs, bar_max_w: int = 200):
    """Dibuja barras de probabilidad LSTM sobre un frame RGB (numpy array in-place).

    Args:
        frame: Array numpy RGB H×W×3 sobre el que se dibuja.
        labels: Lista de etiquetas de clase.
        probs: Array de probabilidades (una por clase).
        bar_max_w: Ancho máximo de la barra en píxeles.
    """
    bar_h = max(24, bar_max_w // 6)
    spacing = bar_h + 8
    for i, (label, prob) in enumerate(zip(labels, probs)):
        y0 = 10 + i * spacing
        y1 = y0 + bar_h
        color = _PROB_COLORS_BGR.get(label, (200, 200, 200))
        cv2.rectangle(frame, (0, y0), (bar_max_w, y1), (40, 40, 60), -1)
        filled = int(bar_max_w * float(prob))
        if filled > 0:
            cv2.rectangle(frame, (0, y0), (filled, y1), color, -1)
        cv2.putText(
            frame,
            f"{_PROB_LABELS_DISPLAY.get(label, label)}: {prob:.0%}",
            (4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
