<div align="center">

# HORUS

### Juego Interactivo de Lenguaje de Señas

*Enseñando lenguaje de señas a niños mediante visión por computador*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyGame](https://img.shields.io/badge/PyGame-2.5%2B-green?logo=python)](https://www.pygame.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-orange)](https://ultralytics.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-CPU-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Landmarks-blueviolet)](https://mediapipe.dev/)

</div>

---

## ¿Qué es HORUS?

HORUS es un proyecto educativo que combina **visión por computador** e **inteligencia artificial** para enseñar lenguaje de señas colombiano a niños a través de un juego interactivo. El sistema detecta señas en tiempo real usando la cámara web y proporciona retroalimentación inmediata.

| Tipo de seña | Señas soportadas | Tecnología |
|---|---|---|
| Estáticas | Vocales: **A, E, I, O, U** | YOLO Segmentación |
| Dinámicas | **Hola · Hola mundo · Buenos días** | MediaPipe + LSTM |

---

## Tabla de Contenidos

- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Modelos](#modelos)
- [Ejecución](#ejecución)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Dependencias](#dependencias)
- [Troubleshooting](#troubleshooting)
- [Equipo](#equipo)

---

## Requisitos

- Python **3.10** o superior
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
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Linux / Mac

# 4. Instalar dependencias
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Modelos

> Los archivos de modelo **no están incluidos en el repositorio** (están en `.gitignore`).
> Necesitas los siguientes archivos antes de ejecutar el proyecto:

| Archivo | Ruta esperada | Descripción |
|---------|--------------|-------------|
| `best.pt` | `runs/segment/vocales-2/weights/best.pt` | Modelo YOLO — vocales |
| `action.h5` | `ai/sign_language/action.h5` | Modelo LSTM — señas dinámicas |

Tienes dos opciones para obtenerlos:

<details>
<summary><strong>Opción A — Entrenar desde cero</strong></summary>

### 1) Modelo YOLO (vocales)

Coloca tu dataset en formato YOLO y ejecuta:

```bash
python upload/train.py
```

El modelo se guarda en `runs/segment/vocales-2/weights/best.pt`.

---

### 2) Extraer keypoints para el LSTM

Organiza los videos de cada seña en carpetas dentro de `data/videos/`:

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

---

### 3) Entrenar el modelo LSTM

```bash
python 02_entrenar_modelo.py
```

Guarda el modelo en `ai/sign_language/action.h5` y las etiquetas en `ai/sign_language/labels.json`.

</details>

<details>
<summary><strong>Opción B — Recibir los modelos del equipo</strong></summary>

Solicita al equipo los archivos `best.pt` y `action.h5` y colócalos en las rutas indicadas en la tabla anterior.

</details>

---

## Ejecución

### Juego principal

```bash
python main.py
```

**Flujo:** Login → Menú → Aprendizaje / Juego

| Tecla | Acción |
|-------|--------|
| `Q` | Salir |
| `R` | Reiniciar puntuación *(solo en escena de juego)* |

---

### Detección LSTM en tiempo real *(standalone)*

```bash
python 03_deteccion_tiempo_real.py
```

Abre la cámara y clasifica señas dinámicas en tiempo real. Presiona `Q` para salir.

---

### Scripts de prueba

```bash
python upload/test.py            # YOLO con segmentación y bounding boxes
python upload/WebCam_Testing.py  # Vista rápida de predicciones YOLO
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
│   ├── learning_scene.py          # Modo Aprender (vocales + señas dinámicas)
│   └── game_scene.py              # Modo Juego
├── models/                        # Modelos MediaPipe (.task)
├── ai/sign_language/
│   ├── action.h5                  # Modelo LSTM (no en git)
│   └── labels.json                # Etiquetas de clases
├── runs/segment/vocales-2/
│   └── weights/best.pt            # Modelo YOLO (no en git)
├── assets/
│   └── images/                    # Imágenes de referencia A–U
├── data/keypoints/                # Keypoints .npy por clase
├── upload/                        # Scripts de entrenamiento y prueba
├── users.json                     # Progreso de usuarios
└── requirements.txt
```

---

## Dependencias

| Paquete | Versión mínima | Uso |
|---------|---------------|-----|
| `pygame` | 2.5.2 | Motor gráfico |
| `opencv-python` | 4.9.0 | Captura y procesamiento de video |
| `ultralytics` | 8.3.0 | Framework YOLO |
| `numpy` | 1.26.0 | Arrays y keypoints |
| `torch` | 2.0.0 | Backend deep learning |
| `mediapipe` | 0.10.0 | Extracción de landmarks |
| `tensorflow-cpu` | 2.15.0 | Inferencia LSTM |

---

## Troubleshooting

<details>
<summary><strong>El juego abre pero no detecta nada</strong></summary>

- Verifica que `best.pt` existe en `runs/segment/vocales-2/weights/`.
- Verifica que la cámara no esté siendo usada por otra aplicación.

</details>

<details>
<summary><strong>Error al iniciar el modo de señas dinámicas</strong></summary>

- Verifica que `action.h5` existe en `ai/sign_language/`.
- Verifica que `labels.json` existe en `ai/sign_language/`.
- Verifica que los modelos MediaPipe `.task` existen en `models/`.

</details>

<details>
<summary><strong>Conflicto TensorFlow + PyTorch</strong></summary>

- Usa `tensorflow-cpu` (no `tensorflow`) para evitar conflictos de CUDA con PyTorch.
- Si el conflicto persiste, crea un entorno virtual separado para cada modelo.

</details>

<details>
<summary><strong>Cámara no encontrada (Windows)</strong></summary>

- Ve a **Configuración → Privacidad → Cámara** y verifica que la aplicación tenga permisos.

</details>

---

## Equipo

Desarrollado por el **Equipo HORUS** — Semillero de Investigación.
