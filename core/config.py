from pathlib import Path

# Dimensiones de la ventana
WIDTH  = 1200
HEIGHT = 800
FPS    = 30

# Rutas base
BASE_DIR   = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "assets" / "videos" / "A-E-I-O-U.mp4"
USERS_JSON_PATH = BASE_DIR / "users.json"
IMAGE_DIR  = BASE_DIR / "assets" / "images"
FONT_DIR   = BASE_DIR / "assets" / "fonts"

# Fuentes — Orbitron si está descargada, fallback a pygame default
FONT_REGULAR = str(FONT_DIR / "Orbitron-Regular.ttf") if (FONT_DIR / "Orbitron-Regular.ttf").exists() else None
FONT_BOLD    = str(FONT_DIR / "Orbitron-Bold.ttf")    if (FONT_DIR / "Orbitron-Bold.ttf").exists()    else None
FONT_MEDIUM  = str(FONT_DIR / "Orbitron-Medium.ttf")  if (FONT_DIR / "Orbitron-Medium.ttf").exists()  else FONT_REGULAR

# Diccionario de imágenes de vocales
VOWEL_IMAGES = {
    'A': IMAGE_DIR / "A.jpeg",
    'E': IMAGE_DIR / "E.jpeg",
    'I': IMAGE_DIR / "I.jpeg",
    'O': IMAGE_DIR / "O.jpeg",
    'U': IMAGE_DIR / "U.jpeg",
}

# ── Paleta futurista ───────────────────────────────────────────────────────────
COLORS = {
    # Fondos
    'background': (8,  12, 24),       # Navy muy oscuro
    'card_bg':    (13, 21, 37),       # Azul oscuro

    # Acentos principales
    'primary':    (0,  212, 255),     # Cyan neón
    'secondary':  (139, 92, 246),     # Púrpura

    # Semánticos
    'success':    (0,  230, 118),     # Verde neón
    'error':      (255, 68,  88),     # Rojo
    'warning':    (255, 193,  7),     # Ámbar

    # Texto
    'white':      (200, 216, 255),    # Blanco azulado
    'light_gray': (74,  96,  128),    # Gris-azul (muted)

    # Bordes
    'dark_gray':  (28,  42,  70),     # Borde oscuro

    # Alias
    'accent':     (139, 92, 246),
}

# ── Config UI ──────────────────────────────────────────────────────────────────
UI_CONFIG = {
    'font_regular':      FONT_REGULAR,
    'font_bold':         FONT_BOLD,
    'font_medium_path':  FONT_MEDIUM,

    # Tamaños adaptados para Orbitron (fuente ancha)
    'font_size_title':   48,
    'font_size_large':   36,
    'font_size_medium':  24,
    'font_size_small':   17,
    'font_size_label':   12,

    # Mantener para compatibilidad heredada
    'font_name':         FONT_REGULAR,
    'font_size_large':   36,
    'font_size_medium':  24,
    'font_size_small':   17,

    'button_radius': 0,          # Botones angulares
    'button_padding': 20,
    'border_thickness': 1,
}

# ── Detección YOLO ─────────────────────────────────────────────────────────────
DETECTION_CONFIG = {
    'img_size':          320,
    'skip_frames':       2,
    'conf_threshold':    0.70,
    'min_votes':         3,
    'detections_buffer': 15,
}

# ── Señas dinámicas ────────────────────────────────────────────────────────────
DYNAMIC_SIGNS = ['hola', 'hola_mundo', 'buenos_dias']

SIGN_DISPLAY_NAMES = {
    'hola':        'Hola',
    'hola_mundo':  'Hola Mundo',
    'buenos_dias': 'Buenos Días',
}

# ── Detección LSTM ─────────────────────────────────────────────────────────────
LSTM_CONFIG = {
    'confidence_threshold': 0.80,
    'min_votes':            3,
    'skip_frames':          2,
}
