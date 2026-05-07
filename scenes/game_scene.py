import pygame
from scenes.base_scene import BaseScene
from core.config import COLORS, WIDTH, HEIGHT
from core.logger import get_logger

_log = get_logger("game_scene")

class GameScene(BaseScene):
    def __init__(self):
        super().__init__()
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
                _log.error("Error al iniciar el juego: %s", e, exc_info=True)
            finally:
                self.loading = False
                from scenes.menu_scene import MenuScene
                self.switch_to(MenuScene)

    def draw(self, screen):
        # El dibujo se maneja en el update (pantalla de carga) o en game.run()
        pass
