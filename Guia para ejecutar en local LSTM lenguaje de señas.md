<!-- @format -->

Detector de Lenguaje de Señas — Guía Local con Cámara en Tiempo Real
Resumen del Flujo
Tú grabas videos de cada seña y de "no seña" por tu cuenta
Los pones en carpetas organizadas
Ejecutas un script que extrae los keypoints de esos videos
Ejecutas un script que entrena el modelo LSTM
Ejecutas el script final que abre la cámara y clasifica en tiempo real

Paso 1: Preparar el Entorno
1.1 Crear carpeta y entorno virtual
Tener python 3.10

mkdir sign_language_detection
cd sign_language_detection

python -m venv venv

Activar:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

1.2 Instalar dependencias
pip install tensorflow==2.15.0 mediapipe==0.10.9 opencv-python numpy scikit-learn

Paso 2: Organizar tus Videos
Crea manualmente la carpeta videos/ con una subcarpeta por cada clase. Dentro de cada subcarpeta, pon tus videos grabados (.mp4, .avi, .mov, etc.).
sign_language_detection/
├── venv/
├── videos/ ← CREA ESTA CARPETA TÚ
│ ├── hola/ ← Videos haciendo la seña "hola"
│ │ ├── video1.mp4
│ │ ├── video2.mp4
│ │ └── ...
│ └── no_sena/ ← Videos haciendo movimientos que NO son señas
│ ├── video1.mp4
│ ├── video2.mp4
│ └── ...
├── utils.py
├── 01_extraer_keypoints.py
├── 02_entrenar_modelo.py
└── 03_deteccion_tiempo_real.py

Tips para los videos:
Graba entre 10 y 30 videos por clase
Cada video debe durar entre 1 y 3 segundos
Intenta variar: distancia a la cámara, ángulo, velocidad, ropa
Para no_sena/: graba gestos comunes como rascarte la cabeza, cruzar brazos, saludar con la mano genéricamente, estar quieto, etc.

Paso 3: Crear los Archivos

Estructura de carpetas:

3.1 utils.py — Funciones compartidas
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image.flags.writeable = False
results = model.process(image)
image.flags.writeable = True
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
return image, results

def draw_styled_landmarks(image, results):
mp_drawing.draw_landmarks(
image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
)
mp_drawing.draw_landmarks(
image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
)
mp_drawing.draw_landmarks(
image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
)

def extract_keypoints(results):
pose = np.array([[res.x, res.y, res.z, res.visibility]
for res in results.pose_landmarks.landmark]).flatten() \
 if results.pose_landmarks else np.zeros(33 _ 4)
face = np.array([[res.x, res.y, res.z]
for res in results.face_landmarks.landmark]).flatten() \
 if results.face_landmarks else np.zeros(468 _ 3)
lh = np.array([[res.x, res.y, res.z]
for res in results.left_hand_landmarks.landmark]).flatten() \
 if results.left_hand_landmarks else np.zeros(21 _ 3)
rh = np.array([[res.x, res.y, res.z]
for res in results.right_hand_landmarks.landmark]).flatten() \
 if results.right_hand_landmarks else np.zeros(21 _ 3)
return np.concatenate([pose, face, lh, rh])

3.2 01_extraer_keypoints.py — Procesa tus videos y extrae keypoints
import os
import cv2
import numpy as np
import mediapipe as mp
from utils import mediapipe_detection, extract_keypoints

# ============================================================

# CONFIGURACIÓN

# ============================================================

actions = np.array(['hola', 'no_sena', 'buenos_dias', 'hola_mundo']) # Debe coincidir con nombres de carpetas en videos/
sequence_length = 30 # Frames a extraer por video
VIDEO_PATH = 'videos'
DATA_PATH = 'MP_Data'

# ============================================================

# PROCESAR VIDEOS

# ============================================================

mp_holistic = mp.solutions.holistic

for action in actions:
action_video_path = os.path.join(VIDEO_PATH, action)
action_data_path = os.path.join(DATA_PATH, action)

# Listar videos disponibles

video_files = [f for f in os.listdir(action_video_path)
if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
video_files.sort()

print(f"\n{'='*50}")
print(f"Accion: {action} — {len(video_files)} videos encontrados")
print(f"{'='*50}")

for video_idx, video_file in enumerate(video_files):
full_video_path = os.path.join(action_video_path, video_file)

       # Crear carpeta para esta secuencia
       sequence_folder = os.path.join(action_data_path, str(video_idx))
       os.makedirs(sequence_folder, exist_ok=True)


       # Abrir video
       cap = cv2.VideoCapture(full_video_path)
       total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


       if total_frames == 0:
           print(f"  ⚠ {video_file} — No se pudo leer, saltando...")
           cap.release()
           continue


       # Seleccionar 30 frames distribuidos uniformemente
       frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)


       print(f"  ▶ {video_file} ({total_frames} frames) → extrayendo {sequence_length} keypoints...")


       with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
           for i, frame_index in enumerate(frame_indices):
               cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
               ret, frame = cap.read()


               if not ret:
                   print(f"    ⚠ No se pudo leer frame {frame_index}, guardando zeros")
                   keypoints = np.zeros(1662)
               else:
                   _, results = mediapipe_detection(frame, holistic)
                   keypoints = extract_keypoints(results)


               npy_path = os.path.join(sequence_folder, f"{i}.npy")
               np.save(npy_path, keypoints)


       cap.release()
       print(f"    ✅ Guardado en {sequence_folder}/")

print(f"\n{'='*50}")
print("¡Extraccion completada!")
print(f"Datos guardados en: {os.path.abspath(DATA_PATH)}")
print(f"{'='*50}")

3.3 02_entrenar_modelo.py — Entrena el modelo LSTM
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, accuracy_score

# ============================================================

# CONFIGURACIÓN

# ============================================================

actions = np.array(['hola', 'no_sena', 'buenos_dias', 'hola_mundo'])
sequence_length = 30
DATA_PATH = 'MP_Data'

label_map = {label: num for num, label in enumerate(actions)}
print("Mapa de etiquetas:", label_map)

# ============================================================

# CARGAR DATOS

# ============================================================

sequences, labels = [], []

for action in actions:
action_path = os.path.join(DATA_PATH, action)
sequence_dirs = sorted([d for d in os.listdir(action_path) if d.isdigit()], key=int)

print(f" {action}: {len(sequence_dirs)} secuencias encontradas")

for seq_dir in sequence_dirs:
window = []
seq_path = os.path.join(action_path, seq_dir)

       for frame_num in range(sequence_length):
           npy_file = os.path.join(seq_path, f"{frame_num}.npy")
           if os.path.exists(npy_file):
               res = np.load(npy_file)
           else:
               print(f"    ⚠ Falta {npy_file}, usando zeros")
               res = np.zeros(1662)
           window.append(res)


       sequences.append(window)
       labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

# ============================================================

# DIVIDIR DATOS

# ============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# ============================================================

# CONSTRUIR MODELO

# ============================================================

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# ============================================================

# ENTRENAR

# ============================================================

tb_callback = TensorBoard(log_dir='Logs')
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# ============================================================

# EVALUAR

# ============================================================

y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1).tolist()
y_pred_labels = np.argmax(y_pred, axis=1).tolist()

acc = accuracy_score(y_true, y_pred_labels)
print(f"\n--- RESULTADOS ---")
print(f"Accuracy: {acc:.2%}")

cm = confusion_matrix(y_true, y_pred_labels)
print(f"\nMatriz de confusion:")
print(f"{'':>15} | Pred hola | Pred no_sena | Pred buenos_dias | Pred hola_mundo")
print(f"{'Real hola':>15} | {cm[0][0]:>9} | {cm[0][1]:>12} | {cm[0][2]:>16} | {cm[0][3]:>15}")
print(f"{'Real no_sena':>15} | {cm[1][0]:>9} | {cm[1][1]:>12} | {cm[1][2]:>16} | {cm[1][3]:>15}")
print(f"{'Real buenos_dias':>15} | {cm[2][0]:>9} | {cm[2][1]:>12} | {cm[2][2]:>16} | {cm[2][3]:>15}")
print(f"{'Real hola_mundo':>15} | {cm[3][0]:>9} | {cm[3][1]:>12} | {cm[3][2]:>16} | {cm[3][3]:>15}")

# ============================================================

# GUARDAR MODELO

# ============================================================

model.save('my_model.keras')
print(f"\n✅ Modelo guardado como 'my_model.keras'")

3.4 03_clasificar video.py

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# ============================================================

# CONFIGURACIÓN

# ============================================================

actions = np.array(['hola', 'no_sena', 'buenos_dias', 'hola_mundo'])
model = load_model('my_model.keras')
threshold = 0.7

# Colores BGR para cada acción

colors = [
(0, 255, 0), # hola → verde
(0, 0, 255), # no_sena → rojo
(255, 255, 0), # buenos_dias → cian
(255, 0, 255), # hola_mundo → magenta
]

# Carpetas

TEST_VIDEOS_PATH = 'test_videos'
OUTPUT_VIDEOS_PATH = 'test_videos/output'
os.makedirs(TEST_VIDEOS_PATH, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS_PATH, exist_ok=True)

# ============================================================

# SELECCIONAR VIDEO

# ============================================================

# Listar videos disponibles

video_files = [f for f in os.listdir(TEST_VIDEOS_PATH)
if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
and os.path.isfile(os.path.join(TEST_VIDEOS_PATH, f))]
video_files.sort()

if not video_files:
print(f"No hay videos en la carpeta '{TEST_VIDEOS_PATH}/'")
print(f"Pon tus videos de prueba ahi y vuelve a ejecutar.")
sys.exit()

print("=" _ 50)
print("VIDEOS DISPONIBLES")
print("=" _ 50)
for i, video in enumerate(video_files):
print(f" [{i}] {video}")
print("=" \* 50)

# Pedir selección

while True:
try:
choice = input(f"\nElige un video (0-{len(video_files)-1}): ").strip()
idx = int(choice)
if 0 <= idx < len(video_files):
break
print(f" Numero fuera de rango. Elige entre 0 y {len(video_files)-1}")
except ValueError:
print(" Ingresa un numero valido.")

selected_video = video_files[idx]
input_path = os.path.join(TEST_VIDEOS_PATH, selected_video)

# Nombre del archivo de salida

name, ext = os.path.splitext(selected_video)
output_path = os.path.join(OUTPUT_VIDEOS_PATH, f"{name}\_clasificado{ext}")

print(f"\nProcesando: {selected_video}")
print(f"Salida: {output_path}")

# ============================================================

# VISUALIZACIÓN

# ============================================================

def prob_viz(res, actions, input_frame, colors):
output_frame = input_frame.copy()
for num, prob in enumerate(res):
color = colors[num]
bar_width = int(prob _ 300)
cv2.rectangle(output_frame, (0, 60 + num _ 40),
(bar_width, 90 + num _ 40), color, -1)
cv2.putText(output_frame, f'{actions[num]}: {prob:.0%}',
(0, 85 + num _ 40), cv2.FONT_HERSHEY_SIMPLEX,
0.7, (255, 255, 255), 2, cv2.LINE_AA)
return output_frame

# ============================================================

# PROCESAR VIDEO

# ============================================================

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
print(f"Error: No se pudo abrir '{input_path}'")
sys.exit()

# Propiedades del video

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Resolucion: {width}x{height} | FPS: {fps:.1f} | Frames: {total_frames}")

# Configurar video de salida

fourcc = cv2.VideoWriter_fourcc(\*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Variables de detección

sequence = []
sentence = []
predictions = []
frame_count = 0

mp_holistic = mp.solutions.holistic
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

while cap.isOpened():
ret, frame = cap.read()
if not ret:
break

       frame_count += 1


       # Detección de landmarks
       image, results = mediapipe_detection(frame, holistic)
       draw_styled_landmarks(image, results)


       # Extraer keypoints y acumular
       keypoints = extract_keypoints(results)
       sequence.append(keypoints)
       sequence = sequence[-30:]


       # Predecir cuando tengamos 30 frames
       if len(sequence) == 30:
           res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
           predicted_class = np.argmax(res)
           predictions.append(predicted_class)


           # Estabilización
           if len(predictions) >= 10:
               last_10 = predictions[-10:]
               most_common = max(set(last_10), key=last_10.count)
               stability = last_10.count(most_common) / len(last_10)


               if most_common == predicted_class and stability >= 0.7:
                   if res[predicted_class] > threshold:
                       detected_action = actions[predicted_class]
                       if detected_action != 'no_sena':
                           if len(sentence) == 0 or detected_action != sentence[-1]:
                               sentence.append(detected_action)


           if len(sentence) > 5:
               sentence = sentence[-5:]


           # Dibujar barras de probabilidad
           image = prob_viz(res, actions, image, colors)


       # Dibujar la oración detectada
       cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
       display_text = ' '.join(sentence) if sentence else 'Esperando sena...'
       cv2.putText(image, display_text, (3, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


       # Guardar frame en video de salida
       out.write(image)


       # Mostrar progreso
       progress = (frame_count / total_frames) * 100
       print(f"\r  Progreso: {progress:.1f}% ({frame_count}/{total_frames})", end="")

cap.release()
out.release()

print(f"\n\n{'='*50}")
print(f"Video clasificado guardado en: {output_path}")
if sentence:
print(f"Senas detectadas: {' '.join(sentence)}")
else:
print(f"No se detectaron senas en el video.")
print(f"{'='*50}")

3.5 04_deteccion_tiempo_real.py — Cámara en vivo
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# ============================================================

# CONFIGURACIÓN

# ============================================================

actions = np.array(['hola', 'no_sena', 'buenos_dias', 'hola_mundo'])
model = load_model('my_model.keras')
threshold = 0.7

# Colores BGR para cada acción

colors = [
(0, 255, 0), # hola → verde
(0, 0, 255), # no_sena → rojo
(255, 255, 0), # buenos_dias → cian
(255, 0, 255), # hola_mundo → magenta
]

# ============================================================

# VISUALIZACIÓN

# ============================================================

def prob_viz(res, actions, input_frame, colors):
output_frame = input_frame.copy()
for num, prob in enumerate(res):
color = colors[num]
bar_width = int(prob _ 300)
cv2.rectangle(output_frame, (0, 60 + num _ 40),
(bar_width, 90 + num _ 40), color, -1)
cv2.putText(output_frame, f'{actions[num]}: {prob:.0%}',
(0, 85 + num _ 40), cv2.FONT_HERSHEY_SIMPLEX,
0.7, (255, 255, 255), 2, cv2.LINE_AA)
return output_frame

# ============================================================

# DETECCIÓN EN TIEMPO REAL

# ============================================================

sequence = []
sentence = []
predictions = []

cap = cv2.VideoCapture(1)

if not cap.isOpened():
print("No se pudo abrir la camara. Intenta cambiar el 0 por 1 o 2.")
exit()

print("Camara abierta. Presiona Q para salir.")

mp_holistic = mp.solutions.holistic
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

while cap.isOpened():
ret, frame = cap.read()
if not ret:
break

       # Detección de landmarks
       image, results = mediapipe_detection(frame, holistic)
       draw_styled_landmarks(image, results)


       # Extraer keypoints y acumular
       keypoints = extract_keypoints(results)
       sequence.append(keypoints)
       sequence = sequence[-30:]


       # Predecir cuando tengamos 30 frames
       if len(sequence) == 30:
           res = model(np.expand_dims(sequence, axis=0), training=False)[0].numpy()
           predicted_class = np.argmax(res)
           predictions.append(predicted_class)


           # Estabilización: voto mayoritario de las últimas 10 predicciones
           if len(predictions) >= 10:
               last_10 = predictions[-10:]
               most_common = max(set(last_10), key=last_10.count)
               stability = last_10.count(most_common) / len(last_10)


               if most_common == predicted_class and stability >= 0.7:
                   if res[predicted_class] > threshold:
                       detected_action = actions[predicted_class]
                       # Solo agregar si es una seña real (no "no_sena")
                       if detected_action != 'no_sena':
                           if len(sentence) == 0 or detected_action != sentence[-1]:
                               sentence.append(detected_action)


           if len(sentence) > 5:
               sentence = sentence[-5:]


           # Dibujar barras de probabilidad
           image = prob_viz(res, actions, image, colors)


       # Dibujar la oración detectada arriba
       cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
       display_text = ' '.join(sentence) if sentence else 'Esperando sena...'
       cv2.putText(image, display_text, (3, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


       # Mostrar ventana
       cv2.imshow('Detector de Lengua de Senas', image)


       if cv2.waitKey(10) & 0xFF == ord('q'):
           break

cap.release()
cv2.destroyAllWindows()
print("Camara cerrada.")

Resumen: Ejecutar Todo

# 1. Activar entorno virtual

# Windows:

venv\Scripts\activate

# Mac/Linux:

source venv/bin/activate

# 2. Poner tus videos en:

videos/hola/
videos/no_sena/
videos/buenos_dias/
videos/hola_mundo/

# 3. Extraer keypoints de los videos

python 01_extraer_keypoints.py

# 4. Entrenar el modelo

python 02_entrenar_modelo.py

# 5. Abrir cámara y detectar en tiempo real

python 04_deteccion_tiempo_real.py

Agregar más señas en el futuro
Si quieres agregar una nueva seña (por ejemplo gracias):
Crea la carpeta videos/gracias/ y pon tus videos ahí
En los 3 scripts, cambia la línea de actions:
actions = np.array(['hola', 'gracias', 'no_sena'])

En 03_deteccion_tiempo_real.py, agrega un color más:

colors = [(0, 255, 0), (255, 255, 0), (0, 0, 255)]
Vuelve a ejecutar los 3 scripts en orden

Solución de Problemas
La cámara no abre: Cambia cv2.VideoCapture(0) por 1 o 2
MediaPipe no instala: Usa Python 3.8–3.10
Predicciones inestables: Sube threshold a 0.8 o graba más videos
Poca precisión: Graba más videos variados (diferente ropa, ángulo, distancia, iluminación)
Rendimiento lento: Agrega cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) después de abrir la cámara
