

import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)

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

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: No se pudo leer el frame de la cámara.")
        break

    result = model.predict(frame)

    cv2.imshow('Webcam Video Stream', result[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

