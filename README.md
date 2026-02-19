<!-- @format -->

# 🤟 HORUS - Juego Interactivo de Lenguaje de Señas

Proyecto educativo que utiliza visión por computador y modelos YOLO para enseñar lenguaje de señas de las vocales (A, E, I, O, U) a través de un juego interactivo con Pygame.

## 📋 Requisitos Previos

- **Python 3.10+** instalado y configurado en el PATH del sistema.
- **Cámara web** funcional conectada al equipo.
- **Modelo YOLO entrenado** (`best.pt`) ubicado en `../train/best.pt` (relativo a la carpeta `upload/`).

## 🚀 Instalación

1. Abre una terminal en la carpeta raíz del proyecto (`HORUS/`).

2. Instala las dependencias:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Verifica que el modelo YOLO esté disponible:
   ```
   C:\Users\Asus\Desktop\U\Semillero\train\best.pt
   ```

## ▶️ Ejecución

### Juego principal

```bash
python upload/game.py
```

- **Q**: Salir del juego.
- **R**: Reiniciar puntuación.

### Test de cámara con YOLO

```bash
python upload/test.py
```

- Muestra detecciones y segmentaciones en tiempo real.
- **Q**: Salir.

### Test rápido de webcam

```bash
python upload/WebCam_Testing.py
```

- Visualización simple de predicciones YOLO.
- **Q**: Salir.

### Entrenamiento del modelo

```bash
python upload/train.py
```

- Entrena un modelo YOLOv11s-seg con el dataset configurado.
- Los resultados se guardan en `runs/segment/train_gpu/`.

## 📁 Estructura del Proyecto

```
HORUS/
├── upload/
│   ├── game.py              # Juego principal con Pygame + YOLO
│   ├── train.py             # Script de entrenamiento YOLO
│   ├── test.py              # Test con segmentación y bounding boxes
│   ├── WebCam_Testing.py    # Test rápido de webcam
│   ├── yolo11n.pt           # Modelo YOLO nano preentrenado
│   └── yolo11s-seg.pt       # Modelo YOLO small segmentación
├── runs/                    # Resultados de entrenamiento
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Este archivo
```

## 📦 Dependencias

| Paquete         | Versión | Uso                                     |
| --------------- | ------- | --------------------------------------- |
| `pygame`        | 2.5.2   | Motor gráfico del juego                 |
| `opencv-python` | 4.9+    | Captura y procesamiento de video        |
| `ultralytics`   | Latest  | Framework YOLO (detección/segmentación) |
| `numpy`         | 1.26+   | Manipulación de imágenes/arrays         |
| `torch`         | 2.0+    | Backend de deep learning                |

## 👥 Equipo

Equipo HORUS - Semillero de Investigación.
