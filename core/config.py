from pathlib import Path

# Dimensiones de la ventana
WIDTH = 1200
HEIGHT = 800
FPS = 30

# Rutas de archivos
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "assets" / "videos" / "A-E-I-O-U.mp4"
USERS_JSON_PATH = BASE_DIR / "users.json"
IMAGE_DIR = BASE_DIR / "assets" / "images"

# Diccionario de imágenes de vocales
VOWEL_IMAGES = {
    'A': IMAGE_DIR / "A.jpeg",
    'E': IMAGE_DIR / "E.jpeg",
    'I': IMAGE_DIR / "I.jpeg",
    'O': IMAGE_DIR / "O.jpeg",
    'U': IMAGE_DIR / "U.jpeg",
}

# Paleta de colores (Mantenida igual para consistencia visual)
COLORS = {
    'background': (20, 25, 40),
    'card_bg': (45, 55, 80),
    'primary': (100, 200, 255),
    'secondary': (150, 100, 255),
    'success': (50, 200, 100),
    'error': (255, 100, 100),
    'warning': (255, 200, 50),
    'white': (255, 255, 255),
    'light_gray': (200, 200, 200),
    'dark_gray': (100, 100, 100),
    'accent': (255, 150, 50)
}

# Configuración Visual para Niños
UI_CONFIG = {
    'font_name': None,
    'font_size_large': 72,
    'font_size_medium': 48,
    'font_size_small': 32,
    'button_radius': 25,
    'button_padding': 20,
    'border_thickness': 4
}

# Optimización del Detector YOLO
DETECTION_CONFIG = {
    'img_size': 320,
    'skip_frames': 2,
    'conf_threshold': 0.70,
    'min_votes': 3,           # mínimo de detecciones válidas para aceptar una seña vocal
    'detections_buffer': 15,  # capacidad del buffer circular de detecciones en juego
}

# Señas dinámicas (reconocidas por el modelo LSTM)
DYNAMIC_SIGNS = ['hola', 'hola_mundo', 'buenos_dias']

# Nombre visual de cada seña (vocales + dinámicas)
SIGN_DISPLAY_NAMES = {
    'hola': 'Hola',
    'hola_mundo': 'Hola Mundo',
    'buenos_dias': 'Buenos Días',
}

# Configuración del detector LSTM
LSTM_CONFIG = {
    'confidence_threshold': 0.80,  # subido de 0.65 → menos falsos positivos
    'min_votes': 3,       # mínimo de predicciones correctas para aprobar (20% de 15 frames)
    'skip_frames': 2,     # procesar 1 de cada N frames con MediaPipe
}
