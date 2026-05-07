# Auditoría Técnica — Proyecto HORUS

**Fecha:** 2026-05-06
**Auditor:** Claude Sonnet 4.6
**Rama:** `dev`

---

## Resumen Ejecutivo

El proyecto tiene **33 problemas identificados** distribuidos en 4 niveles de prioridad. Los problemas críticos incluyen pérdida silenciosa de datos de usuario, código muerto que enmascara bugs reales, y un crash potencial en la UI. Los problemas altos afectan la experiencia de juego de forma visible.

| Prioridad | Cantidad | Impacto |
|-----------|----------|---------|
| P1 Crítico | 5 | Crash / pérdida de datos |
| P2 Alto | 8 | Funcionalidad rota / visible para el usuario |
| P3 Medio | 7 | Rendimiento / deuda técnica acumulada |
| P4 Bajo | 13 | Estilo / UX menor / inconsistencias |

**Orden de ataque recomendado:** C1 → C2 → C3 → A1 → A2 → A8 → C4 → A4, luego P2 restantes, luego P3.

---

## P1 — CRÍTICO

> Pueden causar crash de la aplicación o pérdida de datos.

---

### C1 · `mejor_puntuacion` nunca carga desde `UserManager`

**Archivo:** [`game.py:339`](game.py#L339)

```python
self.mejor_puntuacion = 0  # SIEMPRE empieza en 0
```

`user_manager.get_best_score()` existe y funciona, pero nunca se llama al inicializar el juego. Cada partida muestra "Mejor: 0" aunque el usuario tenga partidas previas guardadas. El score sí se persiste al final (línea 845), pero nunca se recupera al inicio.

**Fix:**
```python
self.mejor_puntuacion = user_manager.get_best_score()
```

---

### C2 · Retry de cámara es un no-op

**Archivo:** [`game.py:361-362`](game.py#L361-L362)

```python
self.cap = cv2.VideoCapture(0)
if not self.cap.isOpened():
    self.cap = cv2.VideoCapture(0)  # ← exactamente lo mismo, nunca funciona
```

Si la cámara falla la primera vez, el "retry" abre exactamente el mismo dispositivo `0` con los mismos parámetros. El juego continúa sin cámara ni mensaje de error al usuario.

---

### C3 · `draw_rounded_rect` produce rects de ancho negativo → crash visual

**Archivos:** [`game.py:206`](game.py#L206), [`game.py:256`](game.py#L256)

```python
# En draw_rounded_rect:
pygame.draw.rect(surface, color, (x + radius, y, w - 2 * radius, h))
#                                                    ↑ puede ser negativo si w < 20
```

`draw_progress_bar` llama a `draw_rounded_rect` con `radius=10`. Cuando `progress` es bajo (ej. 1% de 500px → `progress_width=5`), `w - 2*radius = 5 - 20 = -15`. Pygame puede generar comportamiento indefinido o crash al renderizar un rect con ancho negativo. Ocurre en cada ronda durante los primeros segundos del temporizador.

---

### C4 · Excepciones del LSTM se tragan silenciosamente sin log

**Archivo:** [`core/lstm_detector.py:212-213`](core/lstm_detector.py#L212-L213)

```python
except Exception:
    return None, 0.0  # ← sin logging, sin diagnóstico posible
```

Cualquier error de MediaPipe (modelo corrupto, shape mismatch, out of memory) se silencia completamente. Desde el punto de vista del juego, parece que "no se detectó seña", cuando en realidad hay un fallo interno. Imposible diagnosticar en producción.

---

### C5 · `screen` global en `game.py` capturado en tiempo de importación

**Archivo:** [`game.py:59`](game.py#L59)

```python
else:
    screen = pygame.display.get_surface()  # ← ejecuta al importar el módulo
```

Este código corre cuando `GameScene.__init__` hace `from game import SignLanguageGame`. Si por algún motivo el display no está inicializado en ese momento, `get_surface()` retorna `None`. Luego en `draw_ui`, `pygame.draw.line(None, ...)` crashea. Es frágil por depender del orden de inicialización externo.

---

## P2 — ALTO

> Afectan la funcionalidad de manera visible o crean deuda técnica severa.

---

### A1 · `process_frame()` es código muerto — lógica duplicada en `run()`

**Archivos:** [`game.py:415-437`](game.py#L415-L437) vs [`game.py:795-818`](game.py#L795-L818)

El método `process_frame()` hace exactamente lo que el loop de `run()` hace inline: lee la cámara, corre YOLO, retorna el resultado. Nunca es llamado. Hay dos implementaciones paralelas que pueden divergir. Cualquier bug corregido en una no se corrige en la otra.

---

### A2 · `COLORS` definido dos veces de forma independiente

**Archivos:** [`core/config.py:24-36`](core/config.py#L24-L36) vs [`game.py:62-74`](game.py#L62-L74)

`game.py` no importa `COLORS` de `config.py`, sino que declara un diccionario idéntico. Son actualmente iguales, pero cualquier cambio de color en `config.py` no afecta a `game.py`. La paleta tiene dos fuentes de verdad que se pueden desincronizar silenciosamente.

---

### A3 · Cámara no validada con `isOpened()` en `LearningScene`

**Archivo:** [`scenes/learning_scene.py:432-436`](scenes/learning_scene.py#L432-L436)

```python
def _start_camera(self):
    if self.cap is None:
        self.cap = cv2.VideoCapture(0)  # ← nunca se verifica si abrió
```

Si la cámara falla, `self.cap` no es `None` pero tampoco funciona. El guard `if self.cap is None` impide un retry. El feed de cámara queda en negro sin mensaje de error.

---

### A4 · Pantalla de carga LSTM solo dura 2 frames (~66ms)

**Archivo:** [`scenes/learning_scene.py:172`](scenes/learning_scene.py#L172)

```python
if self._loading_step >= 2:        # ← solo muestra loading 2 frames
    lstm_detector.initialize()     # ← carga TF + MediaPipe: puede tardar 3-10 seg
```

La pantalla "Cargando detector..." aparece por ~66ms (2 frames × 33ms), luego la app se congela mientras TensorFlow y MediaPipe cargan los modelos. El usuario ve un freeze sin feedback. Debería mover la carga a un hilo secundario o usar un spinner animado con `_loading_step` mayor.

---

### A5 · `last_probs` no se resetea al cambiar de seña

**Archivo:** [`core/lstm_detector.py:120-123`](core/lstm_detector.py#L120-L123)

```python
def reset(self):
    if self._initialized:
        self._sequence.clear()
        self._vote_history.clear()
        # ← last_probs NO se limpia
```

Cuando el juego cambia a una nueva seña dinámica, las barras de probabilidad muestran los valores de la seña anterior durante ~30 frames mientras el buffer se recarga. El usuario puede interpretar barras estales como retroalimentación real.

---

### A6 · `predict_smoothed()` en `detector.py` es código muerto

**Archivo:** [`core/detector.py:83-99`](core/detector.py#L83-L99)

`predict_smoothed()` implementa suavizado temporal con ventana deslizante, pero ni `game.py` ni `learning_scene.py` la usan. Ambos implementan su propio buffer de votos encima de `predict()` raw. El método existe pero no se llama. Confunde el API: ¿cuál es la forma correcta de llamar al detector?

---

### A7 · LSTM entrenado con `activation='relu'` en capas LSTM

**Archivo:** [`02_entrenar_modelo.py:67-71`](02_entrenar_modelo.py#L67-L71)

```python
layers.LSTM(64, return_sequences=True, activation="relu"),
```

Las capas LSTM usan `activation='relu'` explícito. El estándar es `activation='tanh'` (default de Keras). ReLU en LSTM puede causar explosión de gradiente durante entrenamiento. El modelo actual ya está entrenado, pero si se reentrena, la precisión puede ser subóptima.

---

### A8 · Entrenamiento no guarda `labels.json` automáticamente

**Archivo:** [`02_entrenar_modelo.py`](02_entrenar_modelo.py)

El script carga clases desde las subcarpetas de `data/keypoints/`, entrena el modelo, pero nunca serializa el orden de clases a `ai/sign_language/labels.json`. Si se añade o elimina una seña y se reentrena, el archivo `labels.json` queda desfasado del modelo. Los índices de predicción apuntarán a clases incorrectas.

---

## P3 — MEDIO

> Problemas de rendimiento y calidad de código que generan deuda técnica.

---

### M1 · Gradiente de fondo: 800 llamadas `draw.line` por frame

**Archivo:** [`game.py:226-232`](game.py#L226-L232)

```python
for i in range(h):  # h = 800
    pygame.draw.line(surface, (r, g, b), (x, y + i), (x + w, y + i))
```

Cada frame renderiza 800 líneas individuales para el fondo. Con `requirements.txt` requiriendo `pygame>=2.5.2`, se puede usar una textura pre-generada como `pygame.Surface`. Este loop es O(height) por frame.

---

### M2 · Partículas crean ~600 `pygame.Surface` nuevas por segundo

**Archivo:** [`game.py:182`](game.py#L182)

```python
for particle in self.particles:
    temp_surface = pygame.Surface(...)  # ← nuevo objeto por partícula por frame
    temp_surface.set_alpha(int(alpha))
```

Con 20 partículas × 30 FPS = 600 allocations/seg durante efectos de éxito. Las superficies deberían pre-allocarse con los tamaños posibles y reutilizarse.

---

### M3 · Inconsistencia de tamaños de fuente entre `game.py` y `BaseScene`

**Archivos:** [`game.py:81-91`](game.py#L81-L91), [`scenes/base_scene.py:9-11`](scenes/base_scene.py#L9-L11)

| Nombre | `game.py` | `BaseScene` (via `UI_CONFIG`) |
|--------|-----------|-------------------------------|
| `font_large` | 48 pt | 72 pt |
| `font_medium` | 36 pt | 48 pt |
| `font_small` | 24 pt | 32 pt |

El mismo nombre tiene tamaño diferente según el contexto. `game.py` usa sus propias fuentes globales, no las del sistema de escenas.

---

### M4 · Lógica de barras de probabilidad LSTM duplicada

**Archivos:** [`game.py:563-577`](game.py#L563-L577), [`scenes/learning_scene.py:364-389`](scenes/learning_scene.py#L364-L389)

El código que dibuja las barras de probabilidad OpenCV sobre el frame es virtualmente idéntico en ambos archivos, con pequeñas diferencias en `bar_max_w` (200 vs 180) y offsets de texto. Una función utilitaria compartida en `core/` eliminaría la duplicación.

---

### M5 · `UserManager` silencia errores de parseo JSON

**Archivo:** [`core/user_manager.py:21-23`](core/user_manager.py#L21-L23)

```python
except Exception:
    pass  # ← sin log, datos perdidos silenciosamente
```

Un `users.json` corrupto (truncado, codificación incorrecta, etc.) resulta en `{}` sin ninguna advertencia. Todos los scores del usuario se pierden silenciosamente. Debería al menos loguear el error con `_log.error(...)`.

---

### M6 · `GameScene` silencia excepciones del juego

**Archivo:** [`scenes/game_scene.py:35-36`](scenes/game_scene.py#L35-L36)

```python
except Exception as e:
    print(f"Error al iniciar el juego: {e}")  # ← solo a consola
```

Cualquier crash dentro de `game_instance.run()` se silencia para el usuario. La app vuelve al menú como si nada. En producción (sin consola visible), el problema es completamente invisible.

---

### M7 · `download_models()` se llama una vez por video, no una vez en total

**Archivo:** [`01_extraer_keypoints.py:70`](01_extraer_keypoints.py#L70)

```python
def process_video(video_path, output_path):
    download_models()  # ← una verificación de filesystem por video
```

`download_models()` se invoca para cada video del dataset. Aunque el check `if not path.exists()` es rápido, es innecesario hacerlo N veces. Debería llamarse una sola vez en `main()`.

---

## P4 — BAJO

> Deuda técnica menor, inconsistencias de estilo y problemas de UX triviales.

---

### B1 · `draw_rounded_rect` reimplementa lo que pygame 2.x ya ofrece

**Archivo:** [`game.py:190-211`](game.py#L190-L211)

El comentario dice "pygame no soporta `border_radius` nativamente en versiones anteriores a 2.x", pero `requirements.txt` exige `pygame>=2.5.2`. Las 12 llamadas de la función custom pueden reemplazarse por un solo `pygame.draw.rect(..., border_radius=r)`.

---

### B2 · `game_scene.py` añade `upload/` a `sys.path` sin necesidad

**Archivo:** [`scenes/game_scene.py:10-12`](scenes/game_scene.py#L10-L12)

Reliquia de arquitectura anterior donde `game.py` vivía en `upload/`. Confunde el árbol de imports sin aportar nada.

---

### B3 · `SIGN_DISPLAY_NAMES` tiene entradas redundantes para vocales

**Archivo:** [`core/config.py:62-64`](core/config.py#L62-L64)

`'A': 'A'` es redundante porque `dict.get('A', 'A')` retorna `'A'` de todos modos. Las 5 entradas de vocales pueden eliminarse sin cambiar el comportamiento.

---

### B4 · `type('Result', (), {...})()` para wrappear resultados de mano

**Archivo:** [`01_extraer_keypoints.py:125-127`](01_extraer_keypoints.py#L125-L127)

Creación de clases anónimas inline es un code smell. Debería usarse `types.SimpleNamespace` o un `dataclass`.

---

### B5 · Posición del cursor de texto incorrecta cuando `user_name` está vacío

**Archivo:** [`scenes/login_scene.py:68`](scenes/login_scene.py#L68)

El cursor aparece centrado cuando el campo está vacío. Debería aparecer en la posición inicial del área de texto (izquierda del input, no el centro de la pantalla).

---

### B6 · Umbral `min_votes=2` de LSTM es demasiado permisivo

**Archivo:** [`core/config.py:72`](core/config.py#L72)

Con `deque(maxlen=15)`, solo 2 detecciones de 15 frames (13%) son suficientes para activar una seña. Puede generar falsos positivos frecuentes. Comparar con YOLO que requiere 3/15 (20%).

---

### B7 · `02_entrenar_modelo.py` sin semilla aleatoria fija

**Archivo:** [`02_entrenar_modelo.py:105`](02_entrenar_modelo.py#L105)

Sin `random_seed` en `validation_split`, los resultados no son reproducibles entre ejecuciones. Añadir `tf.random.set_seed(42)` y `numpy.random.seed(42)` al inicio.

---

### B8 · `_update_yolo` en `LearningScene` ignora la confianza de detección

**Archivo:** [`scenes/learning_scene.py:211`](scenes/learning_scene.py#L211)

```python
detected_class, _, _ = detector.predict(frame)  # descarta detected_conf
```

La confianza no se usa para filtrar detecciones dudosas en modo aprendizaje. Una detección con 71% de confianza se trata igual que una con 99%.

---

### B9 · Título de ventana en `game.py` usa emoji

**Archivo:** [`game.py:57`](game.py#L57)

Puede no renderizarse en todos los sistemas operativos dependiendo del sistema de fuentes del SO.

---

### B10 · `ParticleEffect.update()` usa `list.remove()` con copia

**Archivo:** [`game.py:160-167`](game.py#L160-L167)

Itera sobre `self.particles[:]` pero llama `self.particles.remove(particle)`, que es O(n) search por cada eliminación. Debería filtrarse con list comprehension al final del update.

---

### B11 · `self.CONF_THRESHOLD` es un atributo redundante

**Archivo:** [`game.py:351`](game.py#L351)

`self.CONF_THRESHOLD = DETECTION_CONFIG['conf_threshold']` duplica el valor de config en un atributo de instancia. Algunas comparaciones usan `self.CONF_THRESHOLD`, otras `DETECTION_CONFIG['conf_threshold']` directamente. Fuente de verdad inconsistente.

---

### B12 · `menu_scene.py` no tiene navegación por teclado

**Archivo:** [`scenes/menu_scene.py`](scenes/menu_scene.py)

Sin teclas de acceso rápido ni soporte para `K_RETURN` / `K_ESCAPE`. Solo funciona con mouse.

---

### B13 · Diccionario `_colors` en `draw_camera_feed` se recrea cada frame

**Archivo:** [`game.py:563-564`](game.py#L563-L564)

Un dict con 4 entradas constantes es recreado en cada llamada al método (30 veces/seg). Debería ser una constante a nivel de módulo.

---

## Apéndice — Mapa de Archivos Afectados

| Archivo | Problemas |
|---------|-----------|
| [`game.py`](game.py) | C1, C2, C3, C5, A1, A2, M1, M2, M3, M4, B1, B9, B10, B11, B13 |
| [`core/lstm_detector.py`](core/lstm_detector.py) | C4, A5 |
| [`scenes/learning_scene.py`](scenes/learning_scene.py) | A3, A4, M4, B8 |
| [`core/detector.py`](core/detector.py) | A6 |
| [`02_entrenar_modelo.py`](02_entrenar_modelo.py) | A7, A8, B7 |
| [`core/user_manager.py`](core/user_manager.py) | M5 |
| [`scenes/game_scene.py`](scenes/game_scene.py) | M6, B2 |
| [`01_extraer_keypoints.py`](01_extraer_keypoints.py) | M7, B4 |
| [`core/config.py`](core/config.py) | B3, B6 |
| [`scenes/login_scene.py`](scenes/login_scene.py) | B5 |
| [`scenes/menu_scene.py`](scenes/menu_scene.py) | B12 |
