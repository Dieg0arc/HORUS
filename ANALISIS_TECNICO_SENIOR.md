# HORUS — Análisis Técnico Completo
### Por: Ingeniero Senior (30 años de experiencia en Python, AI y Vision por Computador)
### Fecha: Mayo 2026

---

## 1. ¿QUÉ ES HORUS Y PARA QUÉ SIRVE?

HORUS es un **juego educativo interactivo de lenguaje de señas** que usa inteligencia artificial para detectar lo que el usuario hace con sus manos frente a la cámara. El juego tiene dos modos:

- **Modo Aprender**: El jugador practica señas estáticas (vocales A/E/I/O/U) y dinámicas (Hola, Hola Mundo, Buenos Días) con retroalimentación inmediata de la cámara.
- **Modo Jugar**: El juego elige una seña al azar, da 5 segundos para ejecutarla y evalúa si fue correcta o no. Lleva puntaje y efectos visuales.

La idea central es poderosa: que el jugador entienda que hay una IA que *realmente* lo está mirando y evaluando, combinado con un juego que lo entretenga. Muy buen concepto educativo.

---

## 2. STACK TECNOLÓGICO COMPLETO

### Python y versiones
- **Python 3.11** (inferido del venv: `pip3.11.exe`)
- **Pygame ≥ 2.5.2**: Motor gráfico del juego (ventanas, eventos, dibujado)
- **OpenCV 4.9+**: Captura de cámara y pre-procesamiento de frames
- **Ultralytics (YOLO) ≥ 8.3**: Detección de señas estáticas fotograma a fotograma
- **TensorFlow CPU ≥ 2.15**: Inferencia del modelo LSTM de señas dinámicas
- **MediaPipe ≥ 0.10**: Extracción de landmarks corporales (pose, cara, manos)
- **PyTorch ≥ 2.0**: Usado como backend de YOLO (Ultralytics lo requiere)
- **NumPy ≥ 1.26**: Manipulación de arrays de keypoints e imágenes

### Modelos de IA usados
| Modelo | Archivo | Para qué | Por qué esta elección |
|--------|---------|----------|----------------------|
| YOLO (segmentación) | `best.pt` | Vocales A,E,I,O,U — señas estáticas | Rápido, frame-by-frame, excelente para detección de objetos/gestos fijos |
| LSTM (secuencial) | `action.h5` | Hola, Hola Mundo, Buenos Días — señas dinámicas | Las señas dinámicas son temporales (requieren secuencia de frames), LSTM captura esa dependencia temporal |
| MediaPipe Pose | `pose_landmarker.task` | 33 puntos del cuerpo | Contexto corporal para señas que implican movimiento del torso/brazos |
| MediaPipe Face | `face_landmarker.task` | 478 puntos de la cara | Incluido en entrenamiento — problema grave (ver sección 4) |
| MediaPipe Hands | `hand_landmarker.task` | 21+21 puntos por mano | Las manos son el núcleo de las señas |

### Arquitectura del LSTM
```
Input: 30 frames × 1659 features
  → LSTM(64, return_sequences=True) + Dropout(0.2)
  → LSTM(128, return_sequences=True) + Dropout(0.2)
  → LSTM(64, return_sequences=False) + Dropout(0.2)
  → Dense(64, relu)
  → Dense(32, relu)
  → Dense(4, softmax)  ← [buenos_dias, hola, hola_mundo, no_sena]
```

---

## 3. ARQUITECTURA DEL JUEGO (lo que hace bien)

### Sistema de escenas
El código tiene un sistema limpio de escenas:
- `BaseScene` → clase abstracta con `process_events`, `update`, `draw`
- `LoginScene` → entrada de nombre
- `MenuScene` → Aprender / Jugar / Salir
- `LearningScene` → práctica con cámara en tiempo real
- `GameScene` → lanzador del juego
- `SignLanguageGame` (en `game.py`) → lógica completa del juego

El patrón de `switch_to(SceneClass)` está bien pensado. Los singletons de `Detector` y `LSTMDetector` evitan recargar modelos pesados múltiples veces. El sistema de caché de gradientes (`_gradient_cache`) y la reutilización de superficie para partículas son optimizaciones genuinamente buenas.

---

## 4. PROBLEMAS TÉCNICOS IDENTIFICADOS (diagnóstico de por qué traba)

### PROBLEMA #1 — CRÍTICO: Loop de juego anidado (causa principal del lag)

Este es el problema más serio. En `game_scene.py`:

```python
def update(self):
    if self.loading:
        # ... muestra pantalla de carga ...
        game_instance = self._GameClass()
        game_instance.run()   # ← AQUÍ ESTÁ EL PROBLEMA
```

Y el `main.py` tiene:
```python
while self.running:
    self.active_scene.process_events(events)
    self.active_scene.update()    # ← esto llama a game_instance.run()
    self.active_scene.draw(self.screen)   # ← ESTO NUNCA SE EJECUTA DURANTE EL JUEGO
    pygame.display.flip()
```

Cuando empieza el juego, `update()` llama a `game_instance.run()` que tiene **su propio bucle while** con `clock.tick(30)`. Esto significa:
- El loop principal del `main.py` se queda BLOQUEADO en `update()`
- El `draw()` del sistema de escenas NUNCA se ejecuta mientras se juega
- El juego tiene que manejar TODO él solo (eventos, render, lógica)
- El primer frame visible tarda en aparecer porque primero carga YOLO + TF + cámara **todo en el hilo principal**, después de ya haber mostrado "Iniciando Juego..." un solo frame

**Esto genera el "trabado" inicial que describes.** La carga del modelo YOLO (que ya está en memoria como singleton), la inicialización de la cámara, y en modo LSTM la carga de TensorFlow + 3 modelos MediaPipe, todo ocurre antes del primer frame jugable.

### PROBLEMA #2 — CRÍTICO: Inferencia ML en el hilo principal (causa del lag continuo)

En el bucle de juego (ejecutándose a 30 FPS):

**Modo YOLO** (cada 2 frames):
- `detector.predict(frame)` → inferencia YOLO completa, bloquea el hilo ~30-80ms en CPU

**Modo LSTM** (cada 2 frames):
- `lstm_detector.process_frame(frame)` que internamente hace:
  1. `self._pose.detect(mp_img)` → MediaPipe Pose en CPU
  2. `self._face.detect(mp_img)` → MediaPipe Face **478 landmarks** en CPU ← el más caro
  3. `self._hand.detect(mp_img)` → MediaPipe Hands en CPU
  4. `self.model.predict(seq, verbose=0)` → TensorFlow LSTM en CPU

Son **4 inferencias ML en el mismo hilo que renderiza Pygame**, bloqueando el render cada 2 frames. Con `tensorflow-cpu` (sin GPU), cada `model.predict` puede tomar 50-150ms. El resultado es un FPS real de 8-15 en modo LSTM en lugar de los 30 prometidos.

### PROBLEMA #3 — IMPORTANTE: Face Landmarks 478 puntos es innecesario y costoso

De los 1659 features del vector LSTM:
- Pose: 99 features (6%)
- **Face: 1434 features (86.4% del total)**
- Hands: 126 features (7.6%)

Para detectar "Hola", "Buenos Días" y "Hola Mundo", **las manos y el movimiento del cuerpo son lo que importa**. Los 478 puntos de la cara aportan poco o nada al reconocimiento de estas señas específicas. Esto significa que:
1. MediaPipe Face es el detector más lento de los tres
2. Genera el feature vector más grande (1434 dims)
3. El LSTM tiene que procesar una entrada innecesariamente enorme
4. El entrenamiento tardó más de lo necesario y el modelo es más grande de lo necesario

### PROBLEMA #4 — IMPORTANTE: `tensorflow-cpu` sin GPU

El `requirements.txt` especifica `tensorflow-cpu`. Aunque en muchas laptops no hay GPU CUDA disponible, esto significa que TODA la inferencia del LSTM corre en CPU, lo que es sustancialmente más lento que GPU. Si el hardware tiene GPU, se pierde esa aceleración.

### PROBLEMA #5 — MODERADO: Conversión de frame OpenCV→Pygame ineficiente

```python
def opencv_to_pygame(cv_image):
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    cv_image = np.rot90(cv_image)                          # crea array nuevo
    cv_image = pygame.surfarray.make_surface(cv_image)     # otra copia
    return cv_image
```

Esto crea 2-3 copias del array de imagen por frame. Una alternativa más eficiente es hacer el `cvtColor` y usar `pygame.surfarray.blit_array` directamente, o pre-escalar el frame antes de convertir (procesar 640×480 y luego convertir es más costoso que convertir a 480×360 primero).

### PROBLEMA #6 — VISUAL: Diseño tosco con fuentes por defecto

- `pygame.font.Font(None, tamaño)` usa la fuente **por defecto de Pygame**, que es la PyGame default font (una sans-serif genérica y muy básica)
- No hay fuentes modernas, no hay imágenes de fondo, no hay animaciones de transición entre escenas
- El menú es 3 rectángulos apilados. El juego es paneles de colores planos
- No hay efectos de neón/futurista, no hay animaciones de "AI scan" que hagan sentir que una IA te está analizando

---

## 5. LO QUE FUNCIONA BIEN (no tocar)

Estos componentes son sólidos y deben conservarse en cualquier escenario:

- **Los datos de entrenamiento** (130+ videos en `videos/`): Esto es trabajo humano irremplazable
- **El pipeline de extracción** (`01_extraer_keypoints.py`): Bien estructurado
- **El pipeline de entrenamiento** (`02_entrenar_modelo.py`): Uso correcto de EarlyStopping, ModelCheckpoint, gradient clipping
- **El modelo YOLO** (`best.pt`): Ya entrenado y funcional para vocales
- **La lógica de votación/estabilización** (deque con mayoría de votos): Excelente enfoque para reducir falsos positivos
- **El sistema de escenas** (`BaseScene` + `switch_to`): Patrón limpio
- **El sistema de partículas** y el cache de gradientes: Buenas optimizaciones
- **La arquitectura Singleton** de detectores: Correcta

---

## 6. ¿REHACER DESDE CERO O REFACTORIZAR?

### Razones PARA rehacer desde cero

1. **El loop anidado es un defecto arquitectónico profundo** que requiere mover `game.py` a una escena real
2. **El diseño visual necesita cambiar completamente** — no es un ajuste, es una reconstrucción
3. **La inferencia en hilo principal** requiere añadir threading en varios puntos
4. **Los features de cara** deberían eliminarse y el modelo LSTM reentrenarse (requiere re-extractar keypoints y re-entrenar)
5. A veces es más rápido y limpio empezar fresh que deshacer decisiones profundamente entrelazadas

### Razones EN CONTRA de rehacer desde cero

1. **Los modelos ya están entrenados**. El YOLO `best.pt` para vocales representa horas de recolección de datos y entrenamiento. Eso no se pierde.
2. **130+ videos de datos de entrenamiento** ya están recolectados. Son el activo más valioso.
3. **La lógica de negocio** (sistema de puntos, feedback, timer, votación) está bien pensada y funcionando.
4. Rehacer desde cero tiene riesgo de introducir nuevos bugs en cosas que ya funcionan.
5. El tiempo de rehacerlo todo desde 0 es 3-5× mayor que refactorizar.

---

## 7. DECISIÓN FINAL

### **NO rehacer desde cero. SÍ hacer un refactor profundo y dirigido.**

La razón es simple: el código tiene bugs de arquitectura (loop anidado, inferencia en hilo principal) pero la **lógica de IA y los datos son correctos**. Reescribir implicaría recrear lo que ya funciona bien sin ganancia real. El refactor bien ejecutado resuelve exactamente los problemas que describes (lag, visual tosco).

---

## 8. PLAN DE REFACTOR CONCRETO (en orden de impacto)

### Fase 1 — Eliminar el lag (1-2 días)

**1a. Mover inferencia ML a un hilo de fondo**

Crear un `InferenceThread` que corra en paralelo al render loop:
```python
import threading, queue

class InferenceWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.frame_queue = queue.Queue(maxsize=1)  # solo el frame más reciente
        self.result_queue = queue.Queue(maxsize=1)
    
    def run(self):
        while True:
            frame = self.frame_queue.get()
            result = detector.predict(frame)  # corre en segundo plano
            # reemplaza el resultado anterior si no fue consumido
            try: self.result_queue.get_nowait()
            except: pass
            self.result_queue.put(result)
```

Esto desbloquea completamente el render loop. Pygame sigue dibujando a 30 FPS aunque la inferencia tarde 100ms.

**1b. Convertir GameScene en una escena real (sin loop anidado)**

```python
class GameScene(BaseScene):
    def __init__(self):
        super().__init__()
        self.game = SignLanguageGame()  # inicializa en hilo
        
    def process_events(self, events):
        self.game.handle_events(events)
        
    def update(self):
        self.game.update()
        
    def draw(self, screen):
        self.game.draw(screen)
```

**1c. Eliminar face landmarks del LSTM**

Reentrenar con solo Pose (99) + Hands (126) = 225 features en lugar de 1659. El modelo sería 7× más pequeño y más rápido. Requiere re-ejecutar `01_extraer_keypoints.py` modificado y re-entrenar, pero los **videos originales no se tocan**.

### Fase 2 — Diseño futurista (2-3 días)

**2a. Fuentes personalizadas**

Descargar Orbitron o Rajdhani (Google Fonts, gratis) para dar el look futurista:
```python
font_title = pygame.font.Font("assets/fonts/Orbitron-Bold.ttf", 64)
```

**2b. Efecto "AI Scanning" sobre el feed de cámara**

Añadir una línea de scan animada sobre el feed de la cámara:
```python
scan_y = (scan_y + 3) % camera_height
pygame.draw.line(cam_surf, (0, 255, 200, 100), (0, scan_y), (cam_width, scan_y), 2)
```

Con texto "ANALIZANDO..." parpadeante en verde neón sobre el borde de la cámara.

**2c. Borde de cámara con animación**

En lugar del rectángulo estático actual, hacer esquinas animadas tipo "targeting reticle":
```python
# Cuatro esquinas en L que se mueven levemente (efecto de lock-on)
corner_size = 20 + int(5 * math.sin(time.time() * 3))
# dibujar líneas en L en cada esquina del frame
```

**2d. Fondo con partículas de fondo (subtle)**

Un campo de puntos pequeños que se mueven lentamente en el fondo, tipo "matrix rain" suave, para darle profundidad sin distraer.

**2e. Barra de confianza animada**

Mostrar la confianza de la IA en tiempo real como una barra tipo "signal strength" pulsante, para que el jugador *vea* que la IA está trabajando.

**2f. Transiciones entre escenas**

Un fade-in/fade-out de 200ms entre escenas usando una superficie con alpha:
```python
fade_surface = pygame.Surface((WIDTH, HEIGHT))
fade_surface.fill((0, 0, 0))
for alpha in range(0, 255, 15):
    fade_surface.set_alpha(alpha)
    screen.blit(fade_surface, (0, 0))
    pygame.display.flip()
```

### Fase 3 — Más interactividad (1-2 días)

- **Racha de aciertos**: Contador de racha (streak) con multiplicador de puntos
- **Nivel de dificultad**: Ajustar el tiempo de preparación (fácil: 8s, medio: 5s, difícil: 3s)
- **Sonidos**: `pygame.mixer` con sonidos de éxito/error (archivos `.wav` cortos)
- **Pantalla de resultados**: Al terminar, mostrar estadísticas con gráfico simple de cuántas veces acertó cada seña
- **Modo libre**: Sin timer, solo detectar y mostrar qué seña ves en tiempo real (como un "intérprete" en vivo)

---

## 9. RESUMEN EJECUTIVO

| Aspecto | Estado actual | Causa del problema | Solución |
|---------|--------------|-------------------|---------|
| Lag al iniciar | Grave | Loop anidado + carga en hilo principal | Refactorizar GameScene a escena real |
| Lag continuo en juego | Moderado/Grave | Inferencia ML en hilo principal | Worker thread para inferencia |
| LSTM lento | Moderado | 1659 features (86% cara innecesaria) | Reentrenar sin face (225 features) |
| Visual tosca | Grave | Fuente default + rectángulos planos | Fuentes custom + efectos futuristas |
| Poca interactividad | Moderado | Solo puntuación básica | Racha, dificultad, sonidos, modo libre |

**La lógica de IA, los datos y la estructura base son buenos. Los problemas son de rendimiento y presentación, no de concepto. Refactoriza, no reescribas.**

---

*Análisis generado con revisión completa de: `main.py`, `game.py`, `core/detector.py`, `core/lstm_detector.py`, `core/config.py`, `core/draw_utils.py`, `scenes/*.py`, `01_extraer_keypoints.py`, `02_entrenar_modelo.py`, `requirements.txt`*
