import sys
import pygame
from pathlib import Path
from scenes.base_scene import BaseScene
from core.config import COLORS, WIDTH, HEIGHT

class GameScene(BaseScene):
    def __init__(self):
        super().__init__()
        upload_path = str(Path(__file__).resolve().parent.parent / "upload")
        if upload_path not in sys.path:
            sys.path.append(upload_path)

        # Importar una sola vez en __init__, no en cada llamada a update()
        from game import SignLanguageGame
        self._GameClass = SignLanguageGame
        self.loading = True

    def update(self):
        if self.loading:
            screen = pygame.display.get_surface()
            screen.fill(COLORS['background'])

            load_surf = self.font_large.render("Iniciando Juego...", True, COLORS['primary'])
            screen.blit(load_surf, load_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50)))

            sub_surf = self.font_medium.render("Preparando cámara y detector optimizado", True, COLORS['white'])
            screen.blit(sub_surf, sub_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50)))

            pygame.display.flip()

            try:
                game_instance = self._GameClass()
                game_instance.run()
            except Exception as e:
                print(f"Error al iniciar el juego: {e}")
            finally:
                self.loading = False
                from scenes.menu_scene import MenuScene
                self.switch_to(MenuScene)

    def draw(self, screen):
        # El dibujo se maneja en el update (pantalla de carga) o en game.run()
        pass
