# Auditoría Técnica — Proyecto HORUS

**Fecha:** 2026-05-06
**Auditor:** Claude Sonnet 4.6
**Rama:** `dev`

---

## Resumen Ejecutivo

El proyecto tenía **33 problemas identificados**. Se corrigieron **12 en la primera ronda** (2026-05-06) y **14 más en la segunda ronda** (2026-05-06), cerrando todos los P1, P2 y la mayoría de P3/P4.

| Prioridad | Total | Resueltos | Pendientes |
|-----------|-------|-----------|------------|
| P1 Crítico | 5 | 5 ✅ | 0 |
| P2 Alto | 8 | 8 ✅ | 0 |
| P3 Medio | 7 | 6 ✅ | 1 (M3) |
| P4 Bajo | 13 | 10 ✅ | 3 (B7✅implícito, B8✅, resto menor) |

**Pendientes remanentes:** M3 (inconsistencia tamaños de fuente), B9✅ resuelto.

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

### C5 · `screen` global en `game.py` capturado en tiempo de importación ✅ RESUELTO

**Archivo:** [`game.py`](game.py)

`screen` se inicializa como `None` al importar. Se resuelve con `pygame.display.get_surface()` al inicio de `run()` (con `global screen`), garantizando que el display ya esté activo cuando se captura.

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

### A6 · `predict_smoothed()` en `detector.py` es código muerto ✅ RESUELTO

**Archivo:** [`core/detector.py`](core/detector.py)

`predict_smoothed()`, `_SMOOTHING_WINDOW` y `_smooth_buf` eliminados. El import de `deque` también fue removido. El API del detector ahora expone solo `predict()`.

---

### A7 · LSTM entrenado con `activation='relu'` en capas LSTM ✅ RESUELTO (previo)

**Archivo:** [`02_entrenar_modelo.py`](02_entrenar_modelo.py)

Las capas LSTM usan el default `tanh` de Keras. El parámetro explícito `activation="relu"` fue eliminado en una corrección anterior al código de entrenamiento.

---

### A8 · Entrenamiento no guarda `labels.json` automáticamente ✅ RESUELTO (previo)

**Archivo:** [`02_entrenar_modelo.py`](02_entrenar_modelo.py)

`main()` guarda `labels.json` antes de entrenar, garantizando que el orden de clases siempre esté sincronizado con el modelo.

---

## P3 — MEDIO

> Problemas de rendimiento y calidad de código que generan deuda técnica.

---

### M1 · Gradiente de fondo: 800 llamadas `draw.line` por frame ✅ RESUELTO

**Archivo:** [`game.py`](game.py)

`draw_gradient_rect` usa un `_gradient_cache` keyed por `(color1, color2, w, h)`. La primera llamada genera una `pygame.Surface` con el gradiente; las siguientes son un simple `blit` O(1).

---

### M2 · Partículas crean ~600 `pygame.Surface` nuevas por segundo ✅ RESUELTO

**Archivo:** [`game.py`](game.py)

`ParticleEffect` ahora usa una única `_particle_surf` SRCALPHA de clase (se crea una sola vez). Cada partícula hace `fill((0,0,0,0))` + `draw.circle` + `blit`, eliminando todas las allocations.

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

### M4 · Lógica de barras de probabilidad LSTM duplicada ✅ RESUELTO

**Archivos:** [`game.py`](game.py), [`scenes/learning_scene.py`](scenes/learning_scene.py)

Función `draw_lstm_prob_bars(frame, labels, probs, bar_max_w)` extraída a [`core/draw_utils.py`](core/draw_utils.py). Ambos archivos la importan y la invocan con su `bar_max_w` correspondiente (200 vs 180).

---

### M5 · `UserManager` silencia errores de parseo JSON ✅ RESUELTO

**Archivo:** [`core/user_manager.py:22`](core/user_manager.py#L22)

El bloque `except Exception: pass` ahora llama `_log.error("No se pudo leer %s: %s", _USERS_FILE, e)`. Un JSON corrupto seguirá retornando `{}` pero el error quedará registrado.

---

### M6 · `GameScene` silencia excepciones del juego ✅ RESUELTO

**Archivo:** [`scenes/game_scene.py:35`](scenes/game_scene.py#L35)

`print(...)` reemplazado por `_log.error(..., exc_info=True)`. El traceback completo queda en el logger del módulo, visible independientemente de si hay consola.

---

### M7 · `download_models()` se llama una vez por video, no una vez en total ✅ RESUELTO

**Archivo:** [`01_extraer_keypoints.py`](01_extraer_keypoints.py)

`download_models()` movido al inicio de `main()`. `process_video()` ya no lo invoca.

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

### B3 · `SIGN_DISPLAY_NAMES` tiene entradas redundantes para vocales ✅ RESUELTO

**Archivo:** [`core/config.py`](core/config.py)

Las 5 entradas de vocales `'A': 'A'` ... `'U': 'U'` eliminadas. El dict ahora solo contiene las señas dinámicas con nombres de display no triviales.

---

### B4 · `type('Result', (), {...})()` para wrappear resultados de mano ✅ RESUELTO

**Archivo:** [`01_extraer_keypoints.py`](01_extraer_keypoints.py)

Reemplazado por `types.SimpleNamespace(hand_landmarks=...)`. Import de `types` añadido.

---

### B5 · Posición del cursor de texto incorrecta cuando `user_name` está vacío ✅ RESUELTO

**Archivo:** [`scenes/login_scene.py`](scenes/login_scene.py)

`cursor_x` cuando el campo está vacío ahora usa `self.input_rect.left + 12` en lugar de `centerx`, posicionando el cursor en el borde izquierdo del input.

---

### B6 · Umbral `min_votes=2` de LSTM es demasiado permisivo ✅ RESUELTO

**Archivo:** [`core/config.py`](core/config.py)

`LSTM_CONFIG['min_votes']` elevado de 2 a 3 (20% de 15 frames), igualando el umbral de YOLO y reduciendo falsos positivos.

---

### B7 · `02_entrenar_modelo.py` sin semilla aleatoria fija ✅ RESUELTO (previo)

**Archivo:** [`02_entrenar_modelo.py`](02_entrenar_modelo.py)

`SEED = 42` como constante; `np.random.seed(SEED)` y `tf.random.set_seed(SEED)` llamados al inicio de `main()`.

---

### B8 · `_update_yolo` en `LearningScene` ignora la confianza de detección ✅ RESUELTO

**Archivo:** [`scenes/learning_scene.py`](scenes/learning_scene.py)

`_update_yolo` ahora compara `detected_conf >= DETECTION_CONFIG['conf_threshold']`; solo agrega al buffer si pasa el umbral. Detecciones débiles contribuyen como `None` (abstención).

---

### B9 · Título de ventana en `game.py` usa emoji ✅ RESUELTO

**Archivo:** [`game.py`](game.py)

Emoji `🤟` eliminado del caption. Título queda como `"Aprende Lenguaje de Señas - Juego Interactivo"`.

---

### B10 · `ParticleEffect.update()` usa `list.remove()` con copia ✅ RESUELTO

**Archivo:** [`game.py:160`](game.py#L160)

El bucle ahora itera sobre `self.particles` directamente (sin copia) y al final reemplaza la lista con una list comprehension: `self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] >= 1]`.

---

### B11 · `self.CONF_THRESHOLD` es un atributo redundante ✅ RESUELTO

**Archivo:** [`game.py`](game.py)

`self.CONF_THRESHOLD` eliminado. Todos los usos reemplazados por `DETECTION_CONFIG['conf_threshold']` directamente.

---

### B12 · `menu_scene.py` no tiene navegación por teclado ✅ RESUELTO

**Archivo:** [`scenes/menu_scene.py`](scenes/menu_scene.py)

Soporte añadido para `↑`/`W`, `↓`/`S` (navegar), `Enter`/`Space` (confirmar) y `Escape` (salir). El botón seleccionado se resalta igual que el hover de mouse.

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
