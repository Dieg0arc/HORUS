<!-- @format -->

    # HORUS — Juego Interactivo de Lenguaje de Señas

    Proyecto educativo que usa visión por computador para enseñar lenguaje de señas a niños mediante un juego interactivo con Pygame.

    - **Vocales (A, E, I, O, U):** detectadas con un modelo YOLO de segmentación en tiempo real.
    - **Señas dinámicas (hola, hola mundo, buenos días):** reconocidas con secuencias MediaPipe + LSTM.

    ---

    ## Requisitos Previos

    - Python 3.10+
    - Cámara web funcional
    - Modelos entrenados (ver sección [Modelos](#modelos))

    ---

    ## Instalación

    ```bash
    # 1. Clonar el repositorio
    git clone https://github.com/Dieg0arc/HORUS.git
    cd HORUS

    # 2. Crear entorno virtual
    python -m venv venv

    # 3. Activar entorno
    # En Windows:
    .\venv\Scripts\activate
    # En Linux/Mac:j
    source venv/bin/activate

    # 4. Instalar dependencias
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```

    ---

    ## Modelos

    Los archivos de modelo **no están incluidos en el repositorio** (están en `.gitignore`).
    Necesitas dos archivos antes de poder ejecutar el proyecto:

    | Archivo | Ruta esperada | Descripción |
    |---------|--------------|-------------|
    | `best.pt` | `runs/segment/vocales-2/weights/best.pt` | Modelo YOLO para vocales |
    | `action.h5` | `ai/sign_language/action.h5` | Modelo LSTM para señas dinámicas |

    ### Opción A — Entrenar desde cero

    #### 1) Entrenar el modelo YOLO (vocales)

    Coloca tu dataset en el formato YOLO y ejecuta:

    ```bash
    python upload/train.py
    ```

    El modelo se guarda automáticamente en `runs/segment/vocales-2/weights/best.pt`.

    #### 2) Extraer keypoints para el LSTM

    Coloca videos de cada seña en carpetas dentro de `data/videos/`:

    ```
    data/videos/
    ├── hola/
    ├── hola_mundo/
    ├── buenos_dias/
    └── no_sena/
    ```

    Luego extrae los keypoints:

    ```bash
    python 01_extraer_keypoints.py
    ```

    Genera archivos `.npy` en `data/keypoints/<clase>/`.

    #### 3) Entrenar el modelo LSTM

    ```bash
    python 02_entrenar_modelo.py
    ```

    Guarda el modelo en `ai/sign_language/action.h5` y las etiquetas en `ai/sign_language/labels.json`.

    ### Opción B — Recibir los modelos del equipo

    Solicita al equipo los archivos `best.pt` y `action.h5` y colócalos en las rutas de la tabla anterior.

    ---

    ## Ejecución

    ### Juego principal

    ```bash
    python main.py
    ```

    Flujo: Login → Menú → Aprendizaje / Juego.
    Teclas: **Q** salir, **R** reiniciar puntuación (en escena de juego).

    ### Detección LSTM en tiempo real (standalone)

    ```bash
    python 03_deteccion_tiempo_real.py
    ```

    Abre la cámara y clasifica señas dinámicas. Presiona **q** para salir.

    ### Scripts de prueba

    ```bash
    python upload/test.py           # YOLO con segmentación y bounding boxes
    python upload/WebCam_Testing.py # Vista rápida de predicciones YOLO
    ```

    ---

    ## Estructura del Proyecto

    ```
    HORUS/
    ├── main.py                        # Punto de entrada
    ├── core/
    │   ├── config.py                  # Constantes globales (colores, rutas, umbrales)
    │   ├── detector.py                # Singleton YOLO
    │   ├── lstm_detector.py           # Singleton LSTM + MediaPipe
    │   └── user_manager.py            # Persistencia de usuarios en users.json
    ├── scenes/
    │   ├── base_scene.py
    │   ├── login_scene.py
    │   ├── menu_scene.py
    │   ├── learning_scene.py          # Modo Aprender (vocales + señas)
    │   └── game_scene.py              # Modo Juego
    ├── models/                        # Modelos MediaPipe (.task)
    ├── ai/sign_language/
    │   ├── action.h5                  # Modelo LSTM (no en git)
    │   └── labels.json                # Etiquetas de clases
    ├── runs/segment/vocales-2/
    │   └── weights/best.pt            # Modelo YOLO (no en git)
    ├── assets/
    │   └── images/                    # Imágenes de referencia A-U
    ├── data/keypoints/                # Keypoints .npy por clase
    ├── upload/                        # Scripts legacy y de entrenamiento
    ├── users.json                     # Progreso de usuarios
    └── requirements.txt
    ```

    ---

    ## Dependencias

    | Paquete | Versión | Uso |
    |---------|---------|-----|
    | `pygame` | >=2.5.2 | Motor gráfico |
    | `opencv-python` | >=4.9.0 | Captura y procesamiento de video |
    | `ultralytics` | >=8.3.0 | Framework YOLO |
    | `numpy` | >=1.26.0 | Arrays y keypoints |
    | `torch` | >=2.0.0 | Backend deep learning |
    | `mediapipe` | >=0.10.0 | Extracción de landmarks |
    | `tensorflow-cpu` | >=2.15.0 | Inferencia LSTM |

    ---

    ## Troubleshooting

    **El juego abre pero no detecta nada:**
    - Verifica que `best.pt` existe en `runs/segment/vocales-2/weights/`.
    - Verifica que la cámara no esté siendo usada por otra aplicación.

    **Error al iniciar el modo de señas dinámicas:**
    - Verifica que `action.h5` existe en `ai/sign_language/`.
    - Verifica que `labels.json` existe en `ai/sign_language/`.
    - Verifica que los modelos MediaPipe `.task` existen en `models/`.

    **Conflicto TensorFlow + PyTorch:**
    - Usa `tensorflow-cpu` (no `tensorflow`) para evitar conflictos de CUDA con PyTorch.
    - Si persiste, crea un entorno virtual separado.

    **Cámara no encontrada:**
    - En Windows, verifica permisos de cámara en Configuración → Privacidad → Cámara.

    ---

    ## Equipo

    Equipo HORUS — Semillero de Investigación.
