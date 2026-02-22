import pygame
from core.config import COLORS, UI_CONFIG

class BaseScene:
    """Clase base para todas las escenas de la aplicación."""
    
    def __init__(self):
        self.next_scene = None
        self.font_large = pygame.font.Font(UI_CONFIG['font_name'], UI_CONFIG['font_size_large'])
        self.font_medium = pygame.font.Font(UI_CONFIG['font_name'], UI_CONFIG['font_size_medium'])
        self.font_small = pygame.font.Font(UI_CONFIG['font_name'], UI_CONFIG['font_size_small'])
    
    def process_events(self, events):
        """Procesa eventos de Pygame (clics, teclas, etc.)."""
        pass
    
    def update(self):
        """Actualiza la lógica de la escena."""
        pass
    
    def draw(self, screen):
        """Dibuja la escena en la pantalla."""
        pass
    
    def switch_to(self, scene_class):
        """Establece la siguiente escena para cambiar."""
        self.next_scene = scene_class

    def draw_button(self, screen, rect, text, color, hover=False):
        """Dibuja un botón con estilo consistente."""
        radius = UI_CONFIG['button_radius']
        thickness = UI_CONFIG['border_thickness']
        
        # Efecto de sombra
        shadow_rect = rect.copy()
        shadow_rect.y += 4
        pygame.draw.rect(screen, (10, 10, 20), shadow_rect, border_radius=radius)
        
        # Color base (brillo si hay hover)
        base_color = list(color)
        if hover:
            base_color = [min(255, c + 30) for c in base_color]
        
        pygame.draw.rect(screen, base_color, rect, border_radius=radius)
        pygame.draw.rect(screen, COLORS['white'], rect, thickness, border_radius=radius)
        
        # Texto centrado
        text_surf = self.font_medium.render(text, True, COLORS['white'])
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)
