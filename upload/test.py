from ultralytics import YOLO
import cv2
import numpy as np

# ── Cargar el modelo entrenado ──
current_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.join(current_dir, "..", "runs", "segment", "vocales-2", "weights", "best.pt"),
    os.path.join(current_dir, "..", "..", "train", "best.pt"),
    os.path.join(current_dir, "best.pt"),
]
"""list[str]: Rutas posibles donde buscar el modelo ``best.pt``."""

# Iniciar la captura de la cámara web
cap = cv2.VideoCapture(1)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara web.")
    exit()

# Configurar la resolución de la cámara (opcional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Leer un cuadro de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el cuadro de la cámara.")
        break

    # Realizar inferencia en el cuadro
    results = model(frame)

    # Procesar los resultados
    for result in results:
        img = result.orig_img  # Imagen original

        # Verificar si hay detecciones
        if len(result.boxes) == 0:
            print("No se detectaron objetos en este cuadro.")
        
        # Verificar si hay máscaras de segmentación (polígonos)
        if result.masks is not None:
            print("Máscaras de segmentación detectadas.")
            for mask, box in zip(result.masks.xy, result.boxes):
                # Convertir las coordenadas del polígono a formato para OpenCV
                polygon = np.array(mask, dtype=np.int32)
                polygon = polygon.reshape((-1, 1, 2))

                # Obtener la clase (A o E)
                cls = int(box.cls[0])
                label = model.names[cls]  # Solo el nombre de la clase ("A" o "E")
                print(f"Clase detectada: {label}")

                # Dibujar el polígono en la imagen
                cv2.polylines(img, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

                # Obtener coordenadas para la etiqueta
                x, y, w, h = box.xywh[0]
                x, y = int(x - w/2), int(y - h/2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            print("No hay máscaras de segmentación. Usando solo cajas delimitadoras.")
            # Si no hay máscaras, dibujar solo las cajas delimitadoras
            for box in result.boxes:
                # Obtener la clase (A o E)
                cls = int(box.cls[0])
                label = model.names[cls]  # Solo el nombre de la clase ("A" o "E")
                print(f"Clase detectada: {label}")

                # Obtener coordenadas de la caja delimitadora
                x, y, w, h = box.xywh[0]
                x, y, w, h = int(x - w/2), int(y - h/2), int(w), int(h)

                # Dibujar la caja en la imagen
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Dibujar la etiqueta
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Mostrar el cuadro con las detecciones
        cv2.imshow('Deteccion de Lenguaje de Senas', img)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()