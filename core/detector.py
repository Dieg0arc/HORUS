import atexit
import torch
from ultralytics import YOLO
from collections import deque
from pathlib import Path
from core.logger import get_logger

_log = get_logger("detector")

_SMOOTHING_WINDOW = 5  # frames para suavizado temporal
_BASE_DIR = Path(__file__).resolve().parent.parent


class Detector:
    """Clase global para el detector YOLO optimizado."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Detector, cls).__new__(cls)
            cls._instance._init_detector()
        return cls._instance

    def _init_detector(self):
        from core.config import DETECTION_CONFIG

        possible_paths = [
            _BASE_DIR / "runs" / "segment" / "vocales-2" / "weights" / "best.pt",
            _BASE_DIR / "upload" / "best.pt",
            Path(__file__).resolve().parent / "best.pt",
        ]

        model_path = next((p for p in possible_paths if p.exists()), None)

        if model_path is None:
            _log.error("No se encontró best.pt en las rutas esperadas")
            model_path = Path("best.pt")

        _log.info("Cargando modelo YOLO desde: %s", model_path)
        self.model = YOLO(str(model_path))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        _log.info("Detector YOLO cargado en: %s", self.device)

        self.img_size = DETECTION_CONFIG['img_size']
        self.conf_threshold = DETECTION_CONFIG['conf_threshold']

        self._smooth_buf = deque(maxlen=_SMOOTHING_WINDOW)
        atexit.register(self._cleanup)

    def _cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, frame):
        """Inferencia en un frame, retorna la mejor detección sin suavizado."""
        results = self.model.predict(
            source=frame,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )

        detected_class = None
        detected_conf = 0.0
        detected_box = None

        if results and len(results[0].boxes):
            boxes = results[0].boxes
            best_idx = torch.argmax(boxes.conf)
            detected_conf = float(boxes.conf[best_idx])
            detected_class = results[0].names[int(boxes.cls[best_idx])]
            box = boxes.xyxy[best_idx].cpu().numpy()
            detected_box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

        return detected_class, detected_conf, detected_box

    def predict_smoothed(self, frame):
        """Inferencia con suavizado temporal por ventana deslizante.

        Acumula las últimas _SMOOTHING_WINDOW predicciones y retorna
        la clase más frecuente. Ideal para feedback en tiempo real
        donde las predicciones frame a frame son ruidosas.
        """
        detected_class, detected_conf, detected_box = self.predict(frame)
        self._smooth_buf.append(detected_class)

        valid = [c for c in self._smooth_buf if c is not None]
        if not valid:
            return None, 0.0, detected_box

        smoothed_class = max(set(valid), key=valid.count)
        smoothed_conf = valid.count(smoothed_class) / len(self._smooth_buf)
        return smoothed_class, smoothed_conf, detected_box


# Instancia global
detector = Detector()
