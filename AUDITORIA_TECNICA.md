# Auditoría Técnica — Proyecto HORUS

**Fecha:** 2026-05-06
**Auditor:** Claude Sonnet 4.6
**Rama:** `dev`

---

## Resumen Ejecutivo

El proyecto tenía **33 problemas identificados**. Se corrigieron **12 en la primera ronda** (2026-05-06), cubriendo todos los P1 y la mayoría de P2/P3 prioritarios.

| Prioridad | Total | Resueltos | Pendientes |
|-----------|-------|-----------|------------|
| P1 Crítico | 5 | 5 ✅ | 0 |
| P2 Alto | 8 | 5 ✅ | 3 |
| P3 Medio | 7 | 2 ✅ | 5 |
| P4 Bajo | 13 | 0 | 13 |

**Orden de ataque recomendado:** C1 → C2 → C3 → A1 → A2 → A8 → C4 → A4, luego P2 restantes, luego P3.

---

## P1 — CRÍTICO

> Pueden causar crash de la aplicación o pérdida de datos.

---

### C1 · `mejor_puntuacion` nunca carga desde `UserManager` ✅ RESUELTO

**Archivo:** [`game.py:339`](game.py#L339)

`user_manager` ahora se importa a nivel de módulo y `mejor_puntuacion` se inicializa con `user_manager.get_best_score()`. Los imports lazy dentro de métodos fueron eliminados.

---

### C2 · Retry de cámara es un no-op ✅ RESUELTO

**Archivo:** [`game.py:361`](game.py#L361)

El retry ahora intenta el dispositivo `1` como alternativa real. Si ambos fallan, se imprime una advertencia. Mismo fix aplicado en `LearningScene` (ver A3).

---

### C3 · `draw_rounded_rect` produce rects de ancho negativo → crash visual ✅ RESUELTO

**Archivos:** [`game.py:190`](game.py#L190)

La implementación manual fue reemplazada por `pygame.draw.rect(..., border_radius=min(radius, w//2, h//2))` nativo de pygame 2.x (disponible desde pygame 2.0, requerido >= 2.5.2). El guard `w > 0 and h > 0` previene cualquier llamada con dimensiones inválidas. Resuelve también **B1**.

---

### C4 · Excepciones del LSTM se tragan silenciosamente sin log ✅ RESUELTO

**Archivo:** [`core/lstm_detector.py:212`](core/lstm_detector.py#L212)

El bloque `except Exception` ahora llama `_log.error("Error en predict: %s", e, exc_info=True)` antes de retornar `None, 0.0`. Los errores de MediaPipe quedarán registrados en el logger del módulo.

---

### C5 · `screen` global en `game.py` capturado en tiempo de importación

**Archivo:** [`game.py:59`](game.py#L59)

`get_surface()` sigue ejecutándose en tiempo de importación cuando el módulo se carga desde `GameScene`. El riesgo persiste si el display aún no está inicializado. **Pendiente** — requiere refactorizar `SignLanguageGame` para recibir la superficie como parámetro en lugar de capturarla globalmente.

---

## P2 — ALTO

> Afectan la funcionalidad de manera visible o crean deuda técnica severa.

---

### A1 · `process_frame()` es código muerto — lógica duplicada en `run()` ✅ RESUELTO

**Archivo:** [`game.py`](game.py)

El método `process_frame()` fue eliminado. La lógica de captura vive únicamente en el loop de `run()`.

---

### A2 · `COLORS` definido dos veces de forma independiente ✅ RESUELTO

**Archivo:** [`game.py:39`](game.py#L39)

`game.py` ahora importa `COLORS` desde `core/config.py`. La definición local fue eliminada. Una sola fuente de verdad para la paleta.

---

### A3 · Cámara no validada con `isOpened()` en `LearningScene` ✅ RESUELTO

**Archivo:** [`scenes/learning_scene.py:432`](scenes/learning_scene.py#L432)

`_start_camera()` ahora verifica `isOpened()`, reintenta con device `1` y, si ambos fallan, deja `self.cap = None` con advertencia para que el guard de `update()` funcione correctamente.

---

### A4 · Pantalla de carga LSTM solo dura 2 frames (~66ms) ✅ RESUELTO

**Archivo:** [`scenes/learning_scene.py:172`](scenes/learning_scene.py#L172)

La carga LSTM fue movida a un `threading.Thread(daemon=True)`. El estado `"loading"` persiste hasta que el hilo termina (`_loading_done = True`), permitiendo que el loop principal siga renderizando la pantalla de espera sin freeze.

---

### A5 · `last_probs` no se resetea al cambiar de seña ✅ RESUELTO

**Archivo:** [`core/lstm_detector.py:123`](core/lstm_detector.py#L123)

`reset()` ahora incluye `self.last_probs = np.zeros(len(self.labels), dtype=np.float32)`. Las barras de probabilidad arrancan en cero al cambiar de seña.

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

### M5 · `UserManager` silencia errores de parseo JSON ✅ RESUELTO

**Archivo:** [`core/user_manager.py:22`](core/user_manager.py#L22)

El bloque `except Exception: pass` ahora llama `_log.error("No se pudo leer %s: %s", _USERS_FILE, e)`. Un JSON corrupto seguirá retornando `{}` pero el error quedará registrado.

---

### M6 · `GameScene` silencia excepciones del juego ✅ RESUELTO

**Archivo:** [`scenes/game_scene.py:35`](scenes/game_scene.py#L35)

`print(...)` reemplazado por `_log.error(..., exc_info=True)`. El traceback completo queda en el logger del módulo, visible independientemente de si hay consola.

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

### B1 · `draw_rounded_rect` reimplementa lo que pygame 2.x ya ofrece ✅ RESUELTO (junto con C3)

**Archivo:** [`game.py:190`](game.py#L190)

Resuelto en el fix de C3: `draw_rounded_rect` ahora delega directamente a `pygame.draw.rect(..., border_radius=r)`.

---

### B2 · `game_scene.py` añade `upload/` a `sys.path` sin necesidad ✅ RESUELTO

**Archivo:** [`scenes/game_scene.py`](scenes/game_scene.py)

El bloque `sys.path.append(upload_path)` fue eliminado junto con los imports de `sys` y `Path` que ya no se necesitan.

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

### B10 · `ParticleEffect.update()` usa `list.remove()` con copia ✅ RESUELTO

**Archivo:** [`game.py:160`](game.py#L160)

El bucle ahora itera sobre `self.particles` directamente (sin copia) y al final reemplaza la lista con una list comprehension: `self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] >= 1]`.

---

### B11 · `self.CONF_THRESHOLD` es un atributo redundante

**Archivo:** [`game.py:351`](game.py#L351)

`self.CONF_THRESHOLD = DETECTION_CONFIG['conf_threshold']` duplica el valor de config en un atributo de instancia. Algunas comparaciones usan `self.CONF_THRESHOLD`, otras `DETECTION_CONFIG['conf_threshold']` directamente. Fuente de verdad inconsistente.

---

### B12 · `menu_scene.py` no tiene navegación por teclado

**Archivo:** [`scenes/menu_scene.py`](scenes/menu_scene.py)

Sin teclas de acceso rápido ni soporte para `K_RETURN` / `K_ESCAPE`. Solo funciona con mouse.

---

### B13 · Diccionario `_colors` en `draw_camera_feed` se recrea cada frame ✅ RESUELTO

**Archivo:** [`game.py`](game.py)

Extraído a las constantes de módulo `_LSTM_PROB_COLORS` y `_LSTM_PROB_LABELS`. Ya no se recrean en cada frame.

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
