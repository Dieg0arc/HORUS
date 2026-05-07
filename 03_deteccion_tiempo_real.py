"""03_deteccion_tiempo_real.py

Detecta en tiempo real la seña realizada con la cámara web.
Usa los mismos landmarkers (Pose + Face + Hand) que el modelo LSTM fue entrenado.

Requiere:
  - haber entrenado el modelo (`02_entrenar_modelo.py`)
  - modelos MediaPipe en models/ (se descargan automáticamente si faltan)

Ejemplo:
  python 03_deteccion_tiempo_real.py
"""

from pathlib import Path
from collections import deque, Counter
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from core.draw_utils import draw_landmarks_bgr


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "ai" / "sign_language" / "action.h5"
LABELS_FILE = ROOT / "ai" / "sign_language" / "labels.json"
MODELS_DIR = ROOT / "models"

SEQUENCE_LENGTH = 30
PREDICT_HISTORY = 10
THRESHOLD = 0.80

# URLs de los modelos MediaPipe
_MODEL_URLS = {
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

# Colores BGR por clase para las barras de probabilidad
_BAR_COLORS = {
    "hola":        (50,  200, 100),
    "hola_mundo":  (150, 100, 255),
    "buenos_dias": (100, 200, 255),
    "no_sena":     (60,  60,  200),
}


def download_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in _MODEL_URLS.items():
        path = MODELS_DIR / name
        if not path.exists():
            print(f"Descargando {name}...")
            urllib.request.urlretrieve(url, path)
            print(f"  -> {name} descargado.")


def load_labels() -> list:
    import json
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            return json.load(f)
    # Fallback: leer carpetas de keypoints
    kp_dir = ROOT / "data" / "keypoints"
    labels = sorted([d.name for d in kp_dir.iterdir() if d.is_dir()])
    if not labels:
        raise RuntimeError("No se encontraron etiquetas. Ejecuta 01_extraer_keypoints.py y 02_entrenar_modelo.py.")
    return labels


def _landmarks_to_array(landmarks, n_landmarks: int) -> np.ndarray:
    if landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()
    return np.zeros(n_landmarks * 3, dtype=np.float32)


def extract_keypoints(pose_result, face_result, hand_result) -> np.ndarray:
    """Extrae vector de keypoints en el mismo formato que el modelo fue entrenado."""
    pose = _landmarks_to_array(
        pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None, 33
    )
    face = _landmarks_to_array(
        face_result.face_landmarks[0] if face_result.face_landmarks else None, 478
    )
    lh = np.zeros(21 * 3, dtype=np.float32)
    rh = np.zeros(21 * 3, dtype=np.float32)
    if hand_result.hand_landmarks:
        for i, handedness in enumerate(hand_result.handedness):
            arr = _landmarks_to_array(hand_result.hand_landmarks[i], 21)
            if handedness[0].category_name == "Left":
                lh = arr
            else:
                rh = arr
    return np.concatenate([pose, face, lh, rh])



def draw_prob_bars(frame: np.ndarray, labels: list, probs: np.ndarray) -> None:
    """Dibuja barras de probabilidad sobre el frame (in-place, BGR)."""
    bar_max_w = 300
    bar_h = 28
    spacing = bar_h + 6
    x0 = 10
    for i, (label, prob) in enumerate(zip(labels, probs)):
        y0 = 50 + i * spacing
        y1 = y0 + bar_h
        color = _BAR_COLORS.get(label, (180, 180, 180))
        filled = int(bar_max_w * float(prob))
        cv2.rectangle(frame, (x0, y0), (x0 + bar_max_w, y1), (30, 30, 30), -1)
        if filled > 0:
            cv2.rectangle(frame, (x0, y0), (x0 + filled, y1), color, -1)
        cv2.rectangle(frame, (x0, y0), (x0 + bar_max_w, y1), (80, 80, 80), 1)
        cv2.putText(
            frame,
            f"{label}: {prob:.0%}",
            (x0 + 6, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main():
    if not MODEL_FILE.exists():
        raise RuntimeError(f"No se encontró el modelo: {MODEL_FILE}. Ejecuta 02_entrenar_modelo.py primero.")

    labels = load_labels()
    model = tf.keras.models.load_model(str(MODEL_FILE))
    download_models()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    # ── Crear landmarkers individuales (misma API que en entrenamiento) ──
    pose_lm = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODELS_DIR / "pose_landmarker.task")),
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
    )
    face_lm = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODELS_DIR / "face_landmarker.task")),
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
    )
    hand_lm = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODELS_DIR / "hand_landmarker.task")),
            min_hand_detection_confidence=0.1,
            min_hand_presence_confidence=0.1,
            num_hands=2,
        )
    )

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    predictions = deque(maxlen=PREDICT_HISTORY)
    last_probs = None

    try:
        with pose_lm, face_lm, hand_lm:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                pose_res = pose_lm.detect(mp_img)
                face_res = face_lm.detect(mp_img)
                hand_res = hand_lm.detect(mp_img)

                keypoints = extract_keypoints(pose_res, face_res, hand_res)
                sequence.append(keypoints)

                if len(sequence) == SEQUENCE_LENGTH:
                    res = model.predict(
                        np.expand_dims(np.array(sequence), axis=0), verbose=0
                    )[0]
                    last_probs = res
                    predicted_idx = int(np.argmax(res))
                    predictions.append(predicted_idx)

                    most_common = Counter(predictions).most_common(1)[0][0]
                    stability = list(predictions).count(most_common) / len(predictions)

                    if (
                        most_common == predicted_idx
                        and stability >= THRESHOLD
                        and res[predicted_idx] > THRESHOLD
                    ):
                        detected_label = labels[predicted_idx]
                    else:
                        detected_label = "no_sena"
                else:
                    pct = int(len(sequence) / SEQUENCE_LENGTH * 100)
                    detected_label = f"cargando... {pct}%"

                # ── Dibujar landmarks ──
                draw_landmarks_bgr(frame, pose_res, face_res, hand_res)

                # ── Banner superior ──
                is_sign = detected_label not in ("no_sena",) and not detected_label.startswith("cargando")
                banner_color = (40, 140, 40) if is_sign else (30, 30, 140)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), banner_color, -1)
                cv2.putText(
                    frame, detected_label,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )

                # ── Barras de probabilidad ──
                if last_probs is not None:
                    draw_prob_bars(frame, labels, last_probs)

                cv2.imshow("Detector de Lengua de Senas", frame)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
