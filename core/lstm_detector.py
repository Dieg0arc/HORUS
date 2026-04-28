import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
LABELS_PATH = ROOT / "ai" / "sign_language" / "labels.json"
MODEL_PATH = ROOT / "ai" / "sign_language" / "action.h5"

SEQUENCE_LENGTH = 30
# Dimensiones del vector según como fue extraído en entrenamiento:
# pose(33*3=99) + face(478*3=1434, zeros) + lh(21*3=63) + rh(21*3=63) = 1659
POSE_DIMS = 33 * 3
FACE_DIMS = 478 * 3
HAND_DIMS = 21 * 3


class LSTMDetector:
    """Detector de señas dinámicas usando secuencias MediaPipe + modelo LSTM.

    Singleton de inicialización diferida: no carga ningún modelo hasta que
    se llame a initialize() explícitamente, evitando costo en import-time.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def initialize(self):
        """Carga el modelo LSTM y los landmarkers de MediaPipe (pose + manos).

        Se omite el FaceLandmarker por velocidad; el vector de face se rellena
        con ceros — consistente con la forma en que el modelo fue entrenado
        cuando no había cara detectada.
        """
        if self._initialized:
            return

        import tensorflow as tf  # importación diferida

        with open(LABELS_PATH) as f:
            self.labels = json.load(f)

        self.model = tf.keras.models.load_model(str(MODEL_PATH))

        pose_opts = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(MODELS_DIR / "pose_landmarker.task")
            ),
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self._pose = vision.PoseLandmarker.create_from_options(pose_opts)

        hand_opts = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(MODELS_DIR / "hand_landmarker.task")
            ),
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            num_hands=2,
        )
        self._hand = vision.HandLandmarker.create_from_options(hand_opts)

        self._sequence = deque(maxlen=SEQUENCE_LENGTH)
        self._initialized = True
        print("LSTMDetector inicializado.")

    def close(self):
        """Libera los recursos de MediaPipe."""
        if not self._initialized:
            return
        self._pose.close()
        self._hand.close()
        self._sequence.clear()
        self._initialized = False

    def reset(self):
        """Vacía el buffer de secuencia (usar al cambiar de seña objetivo)."""
        if self._initialized:
            self._sequence.clear()

    # ------------------------------------------------------------------
    # Extracción de keypoints
    # ------------------------------------------------------------------

    @staticmethod
    def _to_array(landmarks, n_landmarks: int) -> np.ndarray:
        if landmarks:
            return np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
            ).flatten()
        return np.zeros(n_landmarks * 3, dtype=np.float32)

    def _extract(self, pose_result, hand_result) -> np.ndarray:
        pose = self._to_array(
            pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None,
            33,
        )
        face = np.zeros(FACE_DIMS, dtype=np.float32)

        lh = np.zeros(HAND_DIMS, dtype=np.float32)
        rh = np.zeros(HAND_DIMS, dtype=np.float32)
        if hand_result.hand_landmarks:
            for i, handedness in enumerate(hand_result.handedness):
                arr = self._to_array(hand_result.hand_landmarks[i], 21)
                if handedness[0].category_name == "Left":
                    lh = arr
                else:
                    rh = arr

        # Mismo orden que el entrenamiento: [pose, face, lh, rh]
        return np.concatenate([pose, face, lh, rh])

    # ------------------------------------------------------------------
    # Inferencia frame a frame
    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray) -> tuple:
        """Procesa un frame BGR y retorna la predicción si el buffer está lleno.

        Returns:
            (class_name, confidence) cuando hay 30 frames acumulados.
            (None, 0.0) mientras el buffer se está llenando.
        """
        if not self._initialized:
            return None, 0.0

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_result = self._pose.detect(mp_img)
        hand_result = self._hand.detect(mp_img)

        keypoints = self._extract(pose_result, hand_result)
        self._sequence.append(keypoints)

        if len(self._sequence) < SEQUENCE_LENGTH:
            return None, 0.0

        seq = np.array(self._sequence, dtype=np.float32)
        pred = self.model.predict(np.expand_dims(seq, 0), verbose=0)[0]
        idx = int(np.argmax(pred))
        return self.labels[idx], float(pred[idx])

    @property
    def buffer_progress(self) -> float:
        """Fracción de llenado del buffer [0.0 – 1.0]."""
        if not self._initialized:
            return 0.0
        return len(self._sequence) / SEQUENCE_LENGTH


lstm_detector = LSTMDetector()
