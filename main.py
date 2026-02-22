import pygame
import sys
from core.config import WIDTH, HEIGHT, FPS, COLORS
from scenes.login_scene import LoginScene

class HorusApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("HORUS - Navegación de Escenas")
        self.clock = pygame.time.Clock()
        self.active_scene = LoginScene()
        self.running = True

    def run(self):
        while self.running:
            # Capturar eventos
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.running = False
            
            # Procesar eventos de la escena
            self.active_scene.process_events(events)
            
            # Actualizar lógica de la escena
            self.active_scene.update()
            
            # Cambiar de escena si es necesario
            if self.active_scene.next_scene:
                self.active_scene = self.active_scene.next_scene()
            
            # Dibujar
            self.active_scene.draw(self.screen)
            
            pygame.display.flip()
            self.clock.tick(FPS)
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = HorusApp()
    app.run()
