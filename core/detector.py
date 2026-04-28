import cv2
import torch
from ultralytics import YOLO
import os

class Detector:
    """Clase global para el detector YOLO optimizado."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Detector, cls).__new__(cls)
            cls._instance._init_detector()
        return cls._instance
    
    def _init_detector(self):
        # Encontrar el modelo en rutas posibles
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        possible_paths = [
            os.path.join(base_dir, "runs", "segment", "train_gpu", "weights", "best.pt"),
            os.path.join(base_dir, "runs", "segment", "train_gpu2", "weights", "best.pt"),
            os.path.join(base_dir, "upload", "best.pt"),
            os.path.join(current_dir, "best.pt"),
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print("Error: No se encontró best.pt en las rutas esperadas")
            model_path = "best.pt"
            
        print(f"Buscando modelo en: {model_path}")
        self.model = YOLO(model_path)
        
        # Seleccionar dispositivo (GPU si está disponible)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Detector cargado en: {self.device}")
        
        # Parámetros de optimización
        self.img_size = 320 # Resolución reducida para velocidad
        self.conf_threshold = 0.50
        
    def predict(self, frame):
        """Realiza la inferencia en un frame."""
        # Reducir resolución antes de pasar al modelo para máxima velocidad
        results = self.model.predict(
            source=frame,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            verbose=False, # Silenciar consola
            device=self.device
        )
        
        # Extraer mejor detección
        detected_class = None
        detected_conf = 0.0
        detected_box = None
        
        if results and len(results[0].boxes):
            boxes = results[0].boxes
            # Obtener la de mayor confianza
            best_idx = torch.argmax(boxes.conf)
            detected_conf = float(boxes.conf[best_idx])
            detected_class = results[0].names[int(boxes.cls[best_idx])]
            
            # Coordenadas (x1, y1, x2, y2)
            box = boxes.xyxy[best_idx].cpu().numpy()
            detected_box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            
        return detected_class, detected_conf, detected_box

# Instancia global
detector = Detector()
