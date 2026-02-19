"""
test.py - Script de detección y segmentación de señas con visualización OpenCV.

Este módulo captura video de la cámara web en tiempo real y ejecuta inferencia
con un modelo YOLO entrenado para segmentación de señas de lenguaje de señas.
A diferencia de ``WebCam_Testing.py``, este script procesa manualmente los
resultados de YOLO, dibujando polígonos de segmentación cuando están disponibles
o bounding boxes como fallback.

El script soporta dos modos de visualización:
    - **Segmentación**: Dibuja los contornos (polígonos) de las manos detectadas.
    - **Detección**: Dibuja bounding boxes si no hay máscaras disponibles.

Dependencias:
    - cv2 (OpenCV): Captura y visualización de video.
    - numpy: Manipulación de coordenadas de polígonos.
    - ultralytics: Framework para modelos YOLO.

Uso:
    Ejecutar directamente desde la terminal::

        python test.py

Controles:
    - ``Q``: Salir del programa y cerrar la ventana.

Autor:
    Equipo HORUS - Semillero de Investigación.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os

# ── Cargar el modelo entrenado ──
current_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.join(current_dir, "..", "..", "train", "best.pt"),
    os.path.join(current_dir, "best.pt"),
    r"C:\Users\Asus\Desktop\U\Semillero\train\best.pt"
]
"""list[str]: Rutas posibles donde buscar el modelo ``best.pt``."""

model_path = None
for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    print("Error: No se encontró el archivo del modelo best.pt")
    model_path = "best.pt"

model = YOLO(model_path)
"""YOLO: Modelo YOLO cargado para realizar inferencia de segmentación."""

# ── Iniciar la captura de la cámara web ──
cap = cv2.VideoCapture(0)
"""cv2.VideoCapture: Objeto de captura de video desde la cámara web (índice 0)."""

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara web.")
    exit()

# Configurar la resolución de la cámara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ── Bucle principal de captura e inferencia ──
while True:
    # Leer un cuadro de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el cuadro de la cámara.")
        break

    # Realizar inferencia YOLO en el cuadro capturado
    results = model(frame)

    # Procesar cada resultado de la inferencia
    for result in results:
        img = result.orig_img  # Imagen original sin anotaciones

        # Verificar si hay detecciones
        if len(result.boxes) == 0:
            print("No se detectaron objetos en este cuadro.")

        # ── Modo Segmentación: Dibujar polígonos ──
        if result.masks is not None:
            print("Máscaras de segmentación detectadas.")
            for mask, box in zip(result.masks.xy, result.boxes):
                # Convertir las coordenadas del polígono a formato OpenCV
                polygon = np.array(mask, dtype=np.int32)
                polygon = polygon.reshape((-1, 1, 2))

                # Obtener la clase detectada
                cls = int(box.cls[0])
                label = model.names[cls]
                print(f"Clase detectada: {label}")

                # Dibujar el contorno del polígono sobre la imagen
                cv2.polylines(img, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

                # Calcular posición de la etiqueta desde las coordenadas xywh
                x, y, w, h = box.xywh[0]
                x, y = int(x - w / 2), int(y - h / 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # ── Modo Detección: Dibujar solo bounding boxes ──
            print("No hay máscaras de segmentación. Usando solo cajas delimitadoras.")
            for box in result.boxes:
                # Obtener la clase detectada
                cls = int(box.cls[0])
                label = model.names[cls]
                print(f"Clase detectada: {label}")

                # Calcular coordenadas del bounding box
                x, y, w, h = box.xywh[0]
                x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)

                # Dibujar el bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Dibujar la etiqueta
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Mostrar el cuadro con las detecciones/segmentaciones
        cv2.imshow('Deteccion de Lenguaje de Senas', img)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Limpieza de recursos ──
cap.release()
cv2.destroyAllWindows()