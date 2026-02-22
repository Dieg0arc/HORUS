import pygame
import cv2
import numpy as np
import os
from scenes.base_scene import BaseScene
from core.config import COLORS, WIDTH, HEIGHT, VOWEL_IMAGES, DETECTION_CONFIG
from core.detector import detector

class LearningScene(BaseScene):
    def __init__(self):
        super().__init__()
        self.state = "selection" # selection or practice
        self.selected_vowel = None
        self.vowel_image = None
        
        # Cámara
        self.cap = None
        self.frame_count = 0
        
        # Feedback
        self.feedback_msg = ""
        self.feedback_color = COLORS['white']
        self.detected_class = None
        
        # Botones de vocales
        btn_w, btn_h = 100, 100
        start_x = WIDTH // 2 - (btn_w * 5 + 40 * 4) // 2
        self.vowel_buttons = []
        for i, v in enumerate(['A', 'E', 'I', 'O', 'U']):
            rect = pygame.Rect(start_x + i * (btn_w + 40), HEIGHT // 2, btn_w, btn_h)
            self.vowel_buttons.append({"label": v, "rect": rect, "hover": False})
            
        # Botón Volver
        self.back_button_rect = pygame.Rect(50, HEIGHT - 100, 200, 60)
        self.back_hover = False

    def process_events(self, events):
        mouse_pos = pygame.mouse.get_pos()
        self.back_hover = self.back_button_rect.collidepoint(mouse_pos)
        
        if self.state == "selection":
            for btn in self.vowel_buttons:
                btn["hover"] = btn["rect"].collidepoint(mouse_pos)
        
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.back_hover:
                    if self.state == "practice":
                        self._stop_camera()
                        self.state = "selection"
                    else:
                        from scenes.menu_scene import MenuScene
                        self.switch_to(MenuScene)
                    return # Salir para evitar que procese otros botones en el mismo clic
                
                if self.state == "selection":
                    for btn in self.vowel_buttons:
                        if btn["hover"]:
                            self._start_practice(btn["label"])

    def _start_practice(self, vowel):
        self.selected_vowel = vowel
        self.state = "practice"
        self._load_vowel_image(vowel)
        self._start_camera()

    def _load_vowel_image(self, vowel):
        path = VOWEL_IMAGES.get(vowel)
        if path and os.path.exists(path):
            try:
                self.vowel_image = pygame.image.load(path)
                self.vowel_image = pygame.transform.scale(self.vowel_image, (300, 300))
            except pygame.error:
                print(f"Error: No se puede cargar {path}. Asegúrate de que sea un formato de imagen válido (PNG/JPG).")
                self.vowel_image = None
        else:
            self.vowel_image = None

    def _start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def update(self):
        if self.state == "practice" and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                
                # Inferencia optimizada (saltar frames)
                if self.frame_count % DETECTION_CONFIG['skip_frames'] == 0:
                    self.detected_class, conf, box = detector.predict(frame)
                    
                    if self.detected_class:
                        if self.detected_class.upper() == self.selected_vowel:
                            self.feedback_msg = "Correcto ✅"
                            self.feedback_color = COLORS['success']
                        else:
                            self.feedback_msg = "Intenta de nuevo ❌"
                            self.feedback_color = COLORS['error']
                    else:
                        self.feedback_msg = "Buscando seña..."
                        self.feedback_color = COLORS['warning']
                
                # Convertir para mostrar cámara
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (320, 240))
                frame_surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
                self.cam_surface = pygame.transform.flip(frame_surf, True, False)

    def draw(self, screen):
        screen.fill(COLORS['background'])
        
        if self.state == "selection":
            title = self.font_large.render("Selecciona una vocal para aprender", True, COLORS['primary'])
            screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100)))
            for btn in self.vowel_buttons:
                self.draw_button(screen, btn["rect"], btn["label"], btn["color"] if False else COLORS['secondary'], btn["hover"])
        
        elif self.state == "practice":
            # Título
            title = self.font_large.render(f"Practicando la letra {self.selected_vowel}", True, COLORS['white'])
            screen.blit(title, title.get_rect(center=(WIDTH // 2, 80)))
            
            # Imagen de referencia (Izquierda)
            if self.vowel_image:
                screen.blit(self.vowel_image, (WIDTH // 4 - 150, HEIGHT // 2 - 150))
            else:
                placeholder = self.font_medium.render(f"Imagen {self.selected_vowel}", True, COLORS['light_gray'])
                screen.blit(placeholder, (WIDTH // 4 - 100, HEIGHT // 2))
                
            # Cámara (Derecha)
            if hasattr(self, 'cam_surface'):
                cam_rect = self.cam_surface.get_rect(center=(3 * WIDTH // 4, HEIGHT // 2))
                pygame.draw.rect(screen, COLORS['primary'], cam_rect.inflate(10, 10), 3, border_radius=10)
                screen.blit(self.cam_surface, cam_rect)
                
            # Feedback (Abajo)
            fb_surf = self.font_large.render(self.feedback_msg, True, self.feedback_color)
            screen.blit(fb_surf, fb_surf.get_rect(center=(WIDTH // 2, HEIGHT - 200)))

        # Botón Volver
        self.draw_button(screen, self.back_button_rect, "Volver", COLORS['accent'], self.back_hover)

    def __del__(self):
        self._stop_camera()
