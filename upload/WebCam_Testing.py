"""
WebCam_Testing.py - Script de prueba rápida de la webcam con modelo YOLO.

Este script abre la cámara web, ejecuta inferencia YOLO en tiempo real
sobre cada frame capturado y muestra los resultados anotados en una
ventana de OpenCV. Es útil para validar rápidamente que el modelo
funciona correctamente antes de usarlo en el juego principal.

Dependencias:
    - cv2 (OpenCV): Captura y visualización de video.
    - ultralytics: Framework para modelos YOLO.

Uso:
    Ejecutar directamente desde la terminal::

        python WebCam_Testing.py

Controles:
    - ``Q``: Salir del programa y cerrar la ventana.

Autor:
    Equipo HORUS - Semillero de Investigación.
"""

import cv2
from ultralytics import YOLO
import os

# ── Abrir cámara web ──
cap = cv2.VideoCapture(0)
"""cv2.VideoCapture: Objeto de captura de video desde la cámara web (índice 0)."""

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara web.")
    exit()

# ── Cargar modelo YOLO ──
# Intentar localizar el modelo relativo al script
current_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.join(current_dir, "..", "runs", "segment", "vocales-2", "weights", "best.pt"),
    os.path.join(current_dir, "..", "..", "train", "best.pt"),
    os.path.join(current_dir, "best.pt"),
]
"""list[str]: Rutas posibles donde buscar el modelo ``best.pt``."""

model_path = None
for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    print("Error: No se encontró el archivo del modelo best.pt")
    # Fallback
    model_path = "best.pt"

model = YOLO(model_path)
"""YOLO: Modelo YOLO cargado para realizar predicciones en tiempo real."""

# ── Bucle principal de captura y predicción ──
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: No se pudo leer el frame de la cámara.")
        break

    # Ejecutar predicción YOLO sobre el frame capturado
    result = model.predict(frame)

    # Mostrar el frame anotado con las detecciones
    cv2.imshow('Webcam Video Stream', result[0].plot())

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Limpieza de recursos ──
cap.release()
cv2.destroyAllWindows()
