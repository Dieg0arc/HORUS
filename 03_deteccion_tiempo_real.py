"""03_deteccion_tiempo_real.py

Detecta en tiempo real la seña realizada con la cámara web.

Requiere:
  - haber entrenado el modelo (`02_entrenar_modelo.py`)
  - tener videos procesados con `01_extraer_keypoints.py` para conocer las etiquetas

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


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "ai" / "sign_language" / "action.h5"
KEYPOINTS_DIR = ROOT / "data" / "keypoints"

SEQUENCE_LENGTH = 30
PREDICT_HISTORY = 10
THRESHOLD = 0.7

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
MODEL_PATH = ROOT / "models" / "holistic_landmarker.task"


def download_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print("Descargando modelo Holistic Landmarker...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo descargado.")


def extract_keypoints(result) -> np.ndarray:
    def _landmarks_to_array(landmarks, n_landmarks):
        if landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return np.zeros(n_landmarks * 3, dtype=np.float32)

    pose = _landmarks_to_array(result.pose_landmarks, 33)
    lh = _landmarks_to_array(result.left_hand_landmarks, 21)
    rh = _landmarks_to_array(result.right_hand_landmarks, 21)
    face = _landmarks_to_array(result.face_landmarks, 468)

    return np.concatenate([pose, lh, rh, face], axis=0)


def load_labels():
    labels = sorted([d.name for d in KEYPOINTS_DIR.iterdir() if d.is_dir()])
    if not labels:
        raise RuntimeError(f"No se encontraron etiquetas en {KEYPOINTS_DIR}. Ejecuta 01_extraer_keypoints.py primero.")
    return labels


def main():
    if not MODEL_FILE.exists():
        raise RuntimeError(f"No se encontró el modelo: {MODEL_FILE}. Ejecuta 02_entrenar_modelo.py primero.")

    labels = load_labels()
    model = tf.keras.models.load_model(str(MODEL_FILE))

    download_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Prueba cambiando el índice de VideoCapture (0, 1, 2...).")

    # Crear el HolisticLandmarker
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = vision.HolisticLandmarker.create_from_options(options)

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    predictions = deque(maxlen=PREDICT_HISTORY)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect(mp_image)

            keypoints = extract_keypoints(result)
            sequence.append(keypoints)

            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(np.array(sequence), axis=0), verbose=0)[0]
                predicted_idx = int(np.argmax(res))
                predictions.append(predicted_idx)

                most_common = Counter(predictions).most_common(1)[0][0]
                stability = list(predictions).count(most_common) / len(predictions)

                if most_common == predicted_idx and stability >= THRESHOLD and res[predicted_idx] > THRESHOLD:
                    detected_label = labels[predicted_idx]
                else:
                    detected_label = "no_sena"
            else:
                detected_label = "cargando..."

            cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(
                frame,
                f"{detected_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Detector de Lengua de Senas", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
