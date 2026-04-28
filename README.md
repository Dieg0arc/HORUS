<!-- @format -->

# 🤟 HORUS - Juego Interactivo de Lenguaje de Señas

Proyecto educativo que utiliza visión por computador y modelos YOLO para enseñar lenguaje de señas de las vocales (A, E, I, O, U) a través de un juego interactivo con Pygame.

También incluye un flujo alternativo de reconocimiento de señas usando **MediaPipe Holistic + LSTM**, que permite reconocer señales basadas en keypoints extraídos de videos.

## 📋 Requisitos Previos

- **Python 3.10+** instalado y configurado en el PATH del sistema.
- **Cámara web** funcional conectada al equipo.
- **Modelo YOLO entrenado** (si se usa el flujo YOLO) ubicado en `runs/segment/train_gpu/weights/best.pt`.

## 🚀 Instalación

1. Abre una terminal en la carpeta raíz del proyecto (`HORUS/`).

2. Instala las dependencias:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. (Opcional para el flujo LSTM) Instala dependencias adicionales:

   ```bash
   python -m pip install mediapipe tensorflow-cpu
   ```

## ▶️ Ejecución

### Juego principal (Nueva versión con escenas)

```bash
python main.py
```

- Navegación entre escenas: Login, Menú, Aprendizaje, Juego.
- **Q**: Salir del juego.
- **R**: Reiniciar puntuación (en escena de juego).

### Juego anterior (Legacy)

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

---

## 🧠 Flujo de entrenamiento MediaPipe + LSTM

Este flujo está basado en el PDF de guía y permite reconocer varias clases de señas usando una red LSTM.

### 1) Preparar videos (clases)

Coloca tus videos en carpetas dentro de `videos/`:

- `videos/a/`
- `videos/e/`
- `videos/i/`
- `videos/o/`
- `videos/u/`
- `videos/hola/`
- `videos/hola_mundo/`
- `videos/buenos_dias/`
- `videos/no_sena/`

> Para agregar más clases, crea una carpeta nueva en `videos/` con el nombre de la clase.

### 2) Extraer keypoints

```bash
python 01_extraer_keypoints.py
```

- Genera `.npy` en `data/keypoints/<clase>/` con secuencias de 30 frames.

### 3) Entrenar el modelo LSTM

```bash
python 02_entrenar_modelo.py
```

- Entrena un modelo LSTM y guarda los pesos en `ai/sign_language/action.h5`.

### 4) Detección en tiempo real

```bash
python 03_deteccion_tiempo_real.py
```

- Abre la cámara y muestra en pantalla la clasificación de la seña.
- Presiona **q** para salir.

---

## 📁 Estructura del Proyecto

```
HORUS/
├── main.py                  # Punto de entrada principal (nueva versión con escenas)
├── core/
│   ├── config.py            # Configuraciones globales (dimensiones, rutas, colores)
│   ├── detector.py          # Clase singleton para detector YOLO optimizado
│   └── user_manager.py      # Gestión de usuarios y puntuaciones
├── scenes/
│   ├── base_scene.py        # Clase base para escenas
│   ├── login_scene.py       # Escena de login
│   ├── menu_scene.py        # Escena de menú principal
│   ├── learning_scene.py    # Escena de aprendizaje de señas
│   └── game_scene.py        # Escena de juego interactivo
├── assets/
│   ├── images/              # Imágenes de vocales (A.jpeg, E.jpeg, etc.)
│   └── videos/              # Videos de demostración (A-E-I-O-U.mp4)
├── videos/                  # Videos de entrenamiento para el flujo LSTM
├── data/                    # Keypoints (numpy) generados por 01_extraer_keypoints
├── ai/                      # Modelos y datos de IA adicionales
│   └── sign_language/
│       └── action.h5        # Modelo LSTM entrenado
├── upload/                  # Scripts legacy
│   ├── game.py              # Juego principal legacy
│   ├── train.py             # Script de entrenamiento YOLO
│   ├── test.py              # Test con segmentación y bounding boxes
│   ├── WebCam_Testing.py    # Test rápido de webcam
│   ├── yolo11n.pt           # Modelo YOLO nano preentrenado
│   └── yolo11s-seg.pt       # Modelo YOLO small segmentación
├── runs/                    # Resultados de entrenamiento YOLO
├── users.json               # Archivo de usuarios y puntuaciones
├── requirements.txt         # Dependencias del proyecto
├── .gitignore               # Archivos ignorados por Git
└── README.md                # Este archivo
```

## 📦 Dependencias

| Paquete          | Versión | Uso                                     |
| ---------------- | ------- | --------------------------------------- |
| `pygame`         | 2.5.2   | Motor gráfico del juego                 |
| `opencv-python`  | 4.9+    | Captura y procesamiento de video        |
| `ultralytics`    | Latest  | Framework YOLO (detección/segmentación) |
| `numpy`          | 1.26+   | Manipulación de imágenes/arrays         |
| `torch`          | 2.0+    | Backend de deep learning                |
| `mediapipe`      | Latest  | Extracción de keypoints (LSTM)          |
| `tensorflow-cpu` | Latest  | Entrenamiento/inferencia del LSTM       |

## 👥 Equipo

Equipo HORUS - Semillero de Investigación.

## 🚀 Instalación

1. Abre una terminal en la carpeta raíz del proyecto (`HORUS/`).

2. Instala las dependencias:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Verifica que el modelo YOLO esté disponible en la ruta configurada (por defecto: `runs/segment/train_gpu/weights/best.pt`).

## ▶️ Ejecución

### Juego principal (Nueva versión con escenas)

```bash
python main.py
```

- Navegación entre escenas: Login, Menú, Aprendizaje, Juego.
- **Q**: Salir del juego.
- **R**: Reiniciar puntuación (en escena de juego).

### Juego anterior (Legacy)

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
├── main.py                  # Punto de entrada principal (nueva versión con escenas)
├── core/
│   ├── config.py            # Configuraciones globales (dimensiones, rutas, colores)
│   ├── detector.py          # Clase singleton para detector YOLO optimizado
│   └── user_manager.py      # Gestión de usuarios y puntuaciones
├── scenes/
│   ├── base_scene.py        # Clase base para escenas
│   ├── login_scene.py       # Escena de login
│   ├── menu_scene.py        # Escena de menú principal
│   ├── learning_scene.py    # Escena de aprendizaje de señas
│   └── game_scene.py        # Escena de juego interactivo
├── assets/
│   ├── images/              # Imágenes de vocales (A.jpeg, E.jpeg, etc.)
│   └── videos/              # Videos de demostración (A-E-I-O-U.mp4)
├── upload/                  # Scripts legacy
│   ├── game.py              # Juego principal legacy
│   ├── train.py             # Script de entrenamiento YOLO
│   ├── test.py              # Test con segmentación y bounding boxes
│   ├── WebCam_Testing.py    # Test rápido de webcam
│   ├── yolo11n.pt           # Modelo YOLO nano preentrenado
│   └── yolo11s-seg.pt       # Modelo YOLO small segmentación
├── runs/                    # Resultados de entrenamiento YOLO
├── ai/                      # Modelos y datos de IA adicionales
├── users.json               # Archivo de usuarios y puntuaciones
├── requirements.txt         # Dependencias del proyecto
├── .gitignore               # Archivos ignorados por Git
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
