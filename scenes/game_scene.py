import sys
import os
import pygame
from scenes.base_scene import BaseScene
from core.config import COLORS, WIDTH, HEIGHT

class GameScene(BaseScene):
    def __init__(self):
        super().__init__()
        # Importación diferida para no cargar el modelo hasta que sea necesario
        current_dir = os.path.dirname(os.path.abspath(__file__))
        upload_path = os.path.join(os.path.dirname(current_dir), "upload")
        if upload_path not in sys.path:
            sys.path.append(upload_path)
        
        self.loading = True
            
    def update(self):
        # Al entrar a esta escena, ejecutamos el juego original
        if self.loading:
            # Dibujar un mensaje de carga rápido con el estilo del sistema
            screen = pygame.display.get_surface()
            screen.fill(COLORS['background'])
            
            load_surf = self.font_large.render("Iniciando Juego...", True, COLORS['primary'])
            screen.blit(load_surf, load_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50)))
            
            sub_surf = self.font_medium.render("Preparando cámara y detector optimizado", True, COLORS['white'])
            screen.blit(sub_surf, sub_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50)))
            
            pygame.display.flip()
            
            try:
                from game import SignLanguageGame
                game_instance = SignLanguageGame()
                
                # Ejecutar el loop del juego (bloqueante)
                game_instance.run()
                
            except Exception as e:
                print(f"Error al iniciar el juego: {e}")
                
            # Volver al menú
            from scenes.menu_scene import MenuScene
            self.switch_to(MenuScene)
            self.loading = False

    def draw(self, screen):
        # El dibujo se maneja en el update (pantalla de carga) o en game.run()
        pass
