"""01_extraer_keypoints.py

Extrae keypoints de videos usando MediaPipe y los guarda como archivos .npy.

Requiere:
  - videos organizados en carpetas por clase (ej: videos/hola/, videos/buenos_dias/)
  - cada video debe tener al menos 30 frames

Ejemplo:
  python 01_extraer_keypoints.py
"""

from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp


ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "videos"
KEYPOINTS_DIR = ROOT / "data" / "keypoints"

SEQUENCE_LENGTH = 30

# Nueva API MediaPipe 0.10 - Usar landmarkers individuales
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


def extract_keypoints(pose_result, left_hand_result, right_hand_result) -> np.ndarray:
    def _landmarks_to_array(landmarks, n_landmarks):
        if landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return np.zeros(n_landmarks * 3, dtype=np.float32)

    pose = _landmarks_to_array(pose_result.pose_landmarks if pose_result else None, 33)
    lh = _landmarks_to_array(left_hand_result.hand_landmarks[0] if left_hand_result and left_hand_result.hand_landmarks else None, 21)
    rh = _landmarks_to_array(right_hand_result.hand_landmarks[0] if right_hand_result and right_hand_result.hand_landmarks else None, 21)
    # Face omitido por ahora - usar ceros
    face = np.zeros(468 * 3, dtype=np.float32)

    return np.concatenate([pose, lh, rh, face], axis=0)


def process_video(video_path: Path, keypoints_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < SEQUENCE_LENGTH:
        print(f"Video demasiado corto ({total_frames} frames): {video_path}")
        cap.release()
        return

    # Configurar opciones para cada landmarker
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE
    )
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2
    )

    keypoints_sequence = []

    with PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
         HandLandmarker.create_from_options(hand_options) as hand_landmarker:

        frame_count = 0
        while len(keypoints_sequence) < SEQUENCE_LENGTH and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            pose_result = pose_landmarker.detect(mp_image)
            hand_result = hand_landmarker.detect(mp_image)

            # Separar manos izquierda y derecha
            left_hand_result = None
            right_hand_result = None
            if hand_result and hand_result.hand_landmarks:
                for i, handedness in enumerate(hand_result.handedness):
                    if handedness[0].category_name == 'Left':
                        left_hand_result = type('HandResult', (), {'hand_landmarks': [hand_result.hand_landmarks[i]]})()
                    elif handedness[0].category_name == 'Right':
                        right_hand_result = type('HandResult', (), {'hand_landmarks': [hand_result.hand_landmarks[i]]})()

            keypoints = extract_keypoints(pose_result, left_hand_result, right_hand_result)
            keypoints_sequence.append(keypoints)
            frame_count += 1

    cap.release()

    if len(keypoints_sequence) == SEQUENCE_LENGTH:
        output_file = keypoints_dir / f"{video_path.stem}.npy"
        np.save(output_file, np.array(keypoints_sequence))
        print(f"Procesado: {video_path} -> {output_file}")
    else:
        print(f"No se pudieron extraer suficientes keypoints de: {video_path}")


def main():
    if not VIDEOS_DIR.exists():
        raise RuntimeError(f"No se encontró el directorio de videos: {VIDEOS_DIR}")

    KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Verificar que los modelos existen
    models = ['pose_landmarker.task', 'hand_landmarker.task']
    missing_models = [m for m in models if not Path(m).exists()]

    if missing_models:
        print("Modelos faltantes:", missing_models)
        print("Por favor descarga los modelos manualmente desde:")
        print("- Pose: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
        print("- Hand: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        print("Guárdalos en el directorio raíz del proyecto.")
        return

    video_dirs = [d for d in VIDEOS_DIR.iterdir() if d.is_dir()]
    if not video_dirs:
        raise RuntimeError(f"No se encontraron carpetas de videos en {VIDEOS_DIR}")

    for video_dir in video_dirs:
        label = video_dir.name
        print(f"Procesando clase: {label}")

        keypoints_label_dir = KEYPOINTS_DIR / label
        keypoints_label_dir.mkdir(exist_ok=True)

        video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
        if not video_files:
            print(f"No se encontraron videos en {video_dir}")
            continue

        for video_path in video_files:
            process_video(video_path, keypoints_label_dir)

    print("Extracción de keypoints completada.")


if __name__ == "__main__":
    main()