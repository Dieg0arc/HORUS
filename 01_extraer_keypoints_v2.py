"""01_extraer_keypoints_v2.py

Extrae keypoints de los videos de señas usando MediaPipe Holistic (API antigua).

Estructura esperada:
  videos/<clase>/*.mp4

Salida:
  data/keypoints/<clase>/<video_name>.npy

Cada .npy contiene una secuencia de 30 frames con los keypoints extraídos.

Ejemplo de uso:
  python 01_extraer_keypoints_v2.py
"""

import os
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "videos"
OUTPUT_DIR = ROOT / "data" / "keypoints"
SEQUENCE_LENGTH = 30


mp_holistic = mp.solutions.holistic


def extract_keypoints(results) -> np.ndarray:
    """Extrae un vector 1D de keypoints de MediaPipe Holistic."""

    def _landmarks_to_array(landmarks, n_landmarks):
        if landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return np.zeros(n_landmarks * 3, dtype=np.float32)

    pose = _landmarks_to_array(results.pose_landmarks.landmark if results.pose_landmarks else None, 33)
    lh = _landmarks_to_array(results.left_hand_landmarks.landmark if results.left_hand_landmarks else None, 21)
    rh = _landmarks_to_array(results.right_hand_landmarks.landmark if results.right_hand_landmarks else None, 21)
    face = _landmarks_to_array(results.face_landmarks.landmark if results.face_landmarks else None, 468)

    return np.concatenate([pose, lh, rh, face], axis=0)


def process_video(video_path: Path, output_path: Path):
    """Extrae keypoints de un video y guarda un .npy con la secuencia de largo fijo."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    seq = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            keypoints = extract_keypoints(results)
            seq.append(keypoints)

    cap.release()

    if len(seq) == 0:
        raise RuntimeError(f"No se extrajeron frames de {video_path}")

    # Asegurar longitud fija (30), truncando o rellenando con ceros.
    if len(seq) >= SEQUENCE_LENGTH:
        seq = seq[-SEQUENCE_LENGTH:]
    else:
        padding = [np.zeros_like(seq[0]) for _ in range(SEQUENCE_LENGTH - len(seq))]
        seq = padding + seq

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), np.array(seq, dtype=np.float32))


def main():
    print("Iniciando extraccion de keypoints...")
    if not VIDEOS_DIR.exists():
        raise RuntimeError(f"No existe la carpeta de videos: {VIDEOS_DIR}")

    labels = [d.name for d in sorted(VIDEOS_DIR.iterdir()) if d.is_dir()]
    if not labels:
        raise RuntimeError(f"No se encontraron subcarpetas en {VIDEOS_DIR}. Debes crear videos/<clase>/...")

    print("Clases detectadas:", labels)

    for label in labels:
        class_dir = VIDEOS_DIR / label
        out_dir = OUTPUT_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)

        videos = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')])
        if not videos:
            print(f"  Advertencia: No hay videos en {class_dir}, omitiendo...")
            continue

        print(f"Procesando clase: {label}")
        for video_path in videos:
            out_file = out_dir / (video_path.stem + ".npy")
            print(f"  -> Procesando {video_path.name} -> {out_file.relative_to(ROOT)}")
            process_video(video_path, out_file)

    print("OK Extraccion de keypoints completada.")


if __name__ == "__main__":
    main()