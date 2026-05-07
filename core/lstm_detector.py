import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from pathlib import Path
from core.logger import get_logger

_log = get_logger("lstm_detector")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
LABELS_PATH = ROOT / "ai" / "sign_language" / "labels.json"
MODEL_PATH = ROOT / "ai" / "sign_language" / "action.h5"

SEQUENCE_LENGTH = 30
VOTING_WINDOW = 10
STABILITY_MIN = 0.7

# Dimensiones del vector — mismo orden que en entrenamiento: [pose, face, lh, rh]
# pose: 33 landmarks × 3 (x,y,z)         =   99
# face: 478 landmarks × 3 (x,y,z)         = 1434
# lh:   21 landmarks × 3 (x,y,z)          =   63
# rh:   21 landmarks × 3 (x,y,z)          =   63
# Total                                    = 1659
POSE_DIMS = 33 * 3
FACE_DIMS = 478 * 3
HAND_DIMS = 21 * 3


class LSTMDetector:
    """Detector de señas dinámicas usando secuencias MediaPipe + modelo LSTM.

    Singleton de inicialización diferida: no carga ningún modelo hasta que
    se llame a initialize() explícitamente, evitando costo en import-time.

    Mejoras respecto a versión anterior:
    - FaceLandmarker activo en inferencia (consistente con datos de entrenamiento).
    - Votación por mayoría sobre últimas VOTING_WINDOW predicciones para
      estabilizar resultados ruidosos, siguiendo la guía de referencia.
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
        if self._initialized:
            return

        import tensorflow as tf  # importación diferida

        with open(LABELS_PATH) as f:
            self.labels = json.load(f)

        self.model = tf.keras.models.load_model(str(MODEL_PATH))
        expected_shape = (None, SEQUENCE_LENGTH, POSE_DIMS + FACE_DIMS + HAND_DIMS * 2)
        if self.model.input_shape != expected_shape:
            _log.warning(
                "Shape del modelo %s no coincide con el esperado %s. "
                "Puede haber incompatibilidad entre el modelo y el extractor.",
                self.model.input_shape,
                expected_shape,
            )

        pose_opts = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(MODELS_DIR / "pose_landmarker.task")
            ),
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self._pose = vision.PoseLandmarker.create_from_options(pose_opts)

        face_opts = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(MODELS_DIR / "face_landmarker.task")
            ),
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self._face = vision.FaceLandmarker.create_from_options(face_opts)

        hand_opts = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(MODELS_DIR / "hand_landmarker.task")
            ),
            min_hand_detection_confidence=0.1,
            min_hand_presence_confidence=0.1,
            num_hands=2,
        )
        self._hand = vision.HandLandmarker.create_from_options(hand_opts)

        self._sequence = deque(maxlen=SEQUENCE_LENGTH)
        self._vote_history = deque(maxlen=VOTING_WINDOW)
        self.last_probs = np.zeros(len(self.labels), dtype=np.float32)
        self._initialized = True
        _log.info("LSTMDetector inicializado.")

    def close(self):
        if not self._initialized:
            return
        self._pose.close()
        self._face.close()
        self._hand.close()
        self._sequence.clear()
        self._vote_history.clear()
        self._initialized = False

    def reset(self):
        if self._initialized:
            self._sequence.clear()
            self._vote_history.clear()
            self.last_probs = np.zeros(len(self.labels), dtype=np.float32)

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

    def _extract(self, pose_result, face_result, hand_result) -> np.ndarray:
        pose = self._to_array(
            pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None,
            33,
        )
        face = self._to_array(
            face_result.face_landmarks[0] if face_result.face_landmarks else None,
            478,
        )
        lh = np.zeros(HAND_DIMS, dtype=np.float32)
        rh = np.zeros(HAND_DIMS, dtype=np.float32)
        if hand_result.hand_landmarks:
            for i, handedness in enumerate(hand_result.handedness):
                arr = self._to_array(hand_result.hand_landmarks[i], 21)
                if handedness[0].category_name == "Left":
                    lh = arr
                else:
                    rh = arr
        return np.concatenate([pose, face, lh, rh])

    # ------------------------------------------------------------------
    # Inferencia frame a frame
    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray) -> tuple:
        """Procesa un frame BGR y retorna la predicción estabilizada.

        Una vez que el buffer de 30 frames está lleno, emite predicciones en
        cada llamada. Tras acumular VOTING_WINDOW predicciones, aplica votación
        por mayoría: si el voto más frecuente alcanza STABILITY_MIN de acuerdo,
        devuelve ese resultado en lugar de la predicción cruda del frame actual.

        Returns:
            (class_name, confidence) cuando el buffer está lleno.
            (None, 0.0) mientras el buffer se está llenando o ante cualquier error.
        """
        if not self._initialized:
            return None, 0.0

        if frame_bgr is None or frame_bgr.size == 0:
            return None, 0.0

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            pose_result = self._pose.detect(mp_img)
            face_result = self._face.detect(mp_img)
            hand_result = self._hand.detect(mp_img)

            keypoints = self._extract(pose_result, face_result, hand_result)
            self._sequence.append(keypoints)

            if len(self._sequence) < SEQUENCE_LENGTH:
                return None, 0.0

            seq = np.array(self._sequence, dtype=np.float32)
            pred = self.model.predict(np.expand_dims(seq, 0), verbose=0)[0]
            self.last_probs = pred
            idx = int(np.argmax(pred))
            conf = float(pred[idx])

            self._vote_history.append(idx)

            # Votación por mayoría una vez que hay suficiente historial
            if len(self._vote_history) >= VOTING_WINDOW:
                last = list(self._vote_history)
                most_common_idx = max(set(last), key=last.count)
                stability = last.count(most_common_idx) / VOTING_WINDOW
                if stability >= STABILITY_MIN:
                    return self.labels[most_common_idx], stability

            # Predicción cruda mientras se construye el historial de votos
            return self.labels[idx], conf

        except Exception as e:
            _log.error("Error en predict: %s", e, exc_info=True)
            return None, 0.0

    @property
    def buffer_progress(self) -> float:
        """Fracción de llenado del buffer de secuencia [0.0 – 1.0]."""
        if not self._initialized:
            return 0.0
        return len(self._sequence) / SEQUENCE_LENGTH


lstm_detector = LSTMDetector()
