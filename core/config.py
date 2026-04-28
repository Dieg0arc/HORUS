import os

# Dimensiones de la ventana
WIDTH = 1200
HEIGHT = 800
FPS = 30

# Rutas de archivos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, "assets", "videos", "A-E-I-O-U.mp4")
USERS_JSON_PATH = os.path.join(BASE_DIR, "users.json")
IMAGE_DIR = os.path.join(BASE_DIR, "assets", "images")

# Diccionario de imágenes de vocales
VOWEL_IMAGES = {
    'A': os.path.join(IMAGE_DIR, "A.jpeg"),
    'E': os.path.join(IMAGE_DIR, "E.jpeg"),
    'I': os.path.join(IMAGE_DIR, "I.jpeg"),
    'O': os.path.join(IMAGE_DIR, "O.jpeg"),
    'U': os.path.join(IMAGE_DIR, "U.jpeg"),
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
    'conf_threshold': 0.70
}

# Señas dinámicas (reconocidas por el modelo LSTM)
DYNAMIC_SIGNS = ['hola', 'hola_mundo', 'buenos_dias']

# Nombre visual de cada seña (vocales + dinámicas)
SIGN_DISPLAY_NAMES = {
    'A': 'A', 'E': 'E', 'I': 'I', 'O': 'O', 'U': 'U',
    'hola': 'Hola',
    'hola_mundo': 'Hola Mundo',
    'buenos_dias': 'Buenos Días',
}

# Configuración del detector LSTM
LSTM_CONFIG = {
    'confidence_threshold': 0.65,
    'min_votes': 2,       # mínimo de predicciones correctas para aprobar
    'skip_frames': 2,     # procesar 1 de cada N frames con MediaPipe
}
