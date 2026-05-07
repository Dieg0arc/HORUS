import pygame
import sys
from scenes.base_scene import BaseScene
from core.config import COLORS, WIDTH, HEIGHT
from core.user_manager import user_manager

class MenuScene(BaseScene):
    def __init__(self):
        super().__init__()

        # Define buttons
        button_width = 400
        button_height = 100
        spacing = 40
        start_y = HEIGHT // 2 - 100

        self.buttons = [
            {"label": "Aprender", "rect": pygame.Rect(WIDTH // 2 - button_width // 2, start_y, button_width, button_height), "color": COLORS['primary'], "action": "learn", "hover": False},
            {"label": "Jugar", "rect": pygame.Rect(WIDTH // 2 - button_width // 2, start_y + button_height + spacing, button_width, button_height), "color": COLORS['success'], "action": "play", "hover": False},
            {"label": "Salir", "rect": pygame.Rect(WIDTH // 2 - button_width // 2, start_y + (button_height + spacing) * 2, button_width, button_height), "color": COLORS['error'], "action": "exit", "hover": False}
        ]
        self._selected_idx = 0

    def process_events(self, events):
        mouse_pos = pygame.mouse.get_pos()
        for btn in self.buttons:
            btn["hover"] = btn["rect"].collidepoint(mouse_pos)

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                for btn in self.buttons:
                    if btn["hover"]:
                        self._handle_action(btn["action"])
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_DOWN, pygame.K_s):
                    self._selected_idx = (self._selected_idx + 1) % len(self.buttons)
                elif event.key in (pygame.K_UP, pygame.K_w):
                    self._selected_idx = (self._selected_idx - 1) % len(self.buttons)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self._handle_action(self.buttons[self._selected_idx]["action"])
                elif event.key == pygame.K_ESCAPE:
                    self._handle_action("exit")

    def _handle_action(self, action):
        if action == "learn":
            from scenes.learning_scene import LearningScene
            self.switch_to(LearningScene)
        elif action == "play":
            from scenes.game_scene import GameScene
            self.switch_to(GameScene)
        elif action == "exit":
            pygame.quit()
            sys.exit()

    def draw(self, screen):
        screen.fill(COLORS['background'])
        
        # Saludo al niño
        user_name = user_manager.get_user_name()
        welcome_text = self.font_large.render(f"¡Hola, {user_name}!", True, COLORS['white'])
        welcome_rect = welcome_text.get_rect(center=(WIDTH // 2, 100))
        # Sombra
        screen.blit(self.font_large.render(f"¡Hola, {user_name}!", True, (10, 10, 20)), (welcome_rect.x + 4, welcome_rect.y + 4))
        screen.blit(welcome_text, welcome_rect)
        
        sub_text = self.font_small.render("¿Qué quieres hacer hoy?", True, COLORS['light_gray'])
        sub_rect = sub_text.get_rect(center=(WIDTH // 2, 170))
        screen.blit(sub_text, sub_rect)
        
        # Dibujar botones usando helper (hover por mouse o selección por teclado)
        for i, btn in enumerate(self.buttons):
            is_active = btn["hover"] or (i == self._selected_idx)
            self.draw_button(screen, btn["rect"], btn["label"], btn["color"], is_active)
