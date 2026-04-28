"""01_extraer_keypoints.py

Extrae keypoints de los videos de señas usando MediaPipe Landmark (nueva API).

Estructura esperada:
  videos/<clase>/*.mp4

Salida:
  data/keypoints/<clase>/<video_name>.npy

Cada .npy contiene una secuencia de 30 frames con los keypoints extraídos.

Ejemplo de uso:
  python 01_extraer_keypoints.py
"""

import os
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "videos"
OUTPUT_DIR = ROOT / "data" / "keypoints"
SEQUENCE_LENGTH = 30

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

POSE_MODEL_PATH = ROOT / "models" / "pose_landmarker.task"
FACE_MODEL_PATH = ROOT / "models" / "face_landmarker.task"
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"


def download_models():
    for url, path in [(POSE_MODEL_URL, POSE_MODEL_PATH), (FACE_MODEL_URL, FACE_MODEL_PATH), (HAND_MODEL_URL, HAND_MODEL_PATH)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            print(f"Descargando modelo {path.name}...")
            urllib.request.urlretrieve(url, path)
            print(f"Modelo {path.name} descargado.")


def extract_keypoints(pose_result, face_result, left_hand_result, right_hand_result) -> np.ndarray:
    """Extrae un vector 1D de keypoints de MediaPipe Landmark."""

    def _landmarks_to_array(landmarks, n_landmarks):
        if landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return np.zeros(n_landmarks * 3, dtype=np.float32)

    pose = _landmarks_to_array(pose_result.pose_landmarks[0], 33) if pose_result.pose_landmarks else np.zeros(33 * 3, dtype=np.float32)
    face = _landmarks_to_array(face_result.face_landmarks[0], 468) if face_result.face_landmarks else np.zeros(468 * 3, dtype=np.float32)
    left_hand = _landmarks_to_array(left_hand_result.hand_landmarks, 21) if left_hand_result and left_hand_result.hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
    right_hand = _landmarks_to_array(right_hand_result.hand_landmarks, 21) if right_hand_result and right_hand_result.hand_landmarks else np.zeros(21 * 3, dtype=np.float32)

    return np.concatenate([pose, face, left_hand, right_hand], axis=0)


def process_video(video_path: Path, output_path: Path):
    """Extrae keypoints de un video y guarda un .npy con la secuencia de largo fijo."""

    download_models()

    # Crear los landmarkers
    pose_base_options = python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    face_base_options = python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH))
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

    hand_base_options = python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        num_hands=2
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    seq = []

    with pose_landmarker, face_landmarker, hand_landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convertir a RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detectar
            pose_result = pose_landmarker.detect(mp_image)
            face_result = face_landmarker.detect(mp_image)
            hand_result = hand_landmarker.detect(mp_image)

            # Asumir left y right basados en handedness o posición
            left_hand_result = None
            right_hand_result = None
            if hand_result.hand_landmarks:
                for i, handedness in enumerate(hand_result.handedness):
                    if handedness[0].category_name == 'Left':
                        left_hand_result = type('Result', (), {'hand_landmarks': hand_result.hand_landmarks[i]})()
                    elif handedness[0].category_name == 'Right':
                        right_hand_result = type('Result', (), {'hand_landmarks': hand_result.hand_landmarks[i]})()

            keypoints = extract_keypoints(pose_result, face_result, left_hand_result, right_hand_result)
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
            print(f"  -> No hay videos en {class_dir}, omitiendo...")
            continue

        for video_path in videos:
            out_file = out_dir / (video_path.stem + ".npy")
            print(f"  → Procesando {video_path.name} -> {out_file.relative_to(ROOT)}")
            process_video(video_path, out_file)

    print("OK Extraccion de keypoints completada.")


if __name__ == "__main__":
    main()
