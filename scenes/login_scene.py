import pygame
from scenes.base_scene import BaseScene
from core.config import COLORS, WIDTH, HEIGHT
from core.user_manager import user_manager

class LoginScene(BaseScene):
    def __init__(self):
        super().__init__()
        self.user_name = ""
        
        # UI Elements
        self.input_rect = pygame.Rect(WIDTH // 2 - 200, HEIGHT // 2 - 32, 400, 64)
        self.button_rect = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 100, 300, 80)
        self.cursor_visible = True
        self.cursor_timer = 0
        self.btn_hover = False

    def process_events(self, events):
        mouse_pos = pygame.mouse.get_pos()
        self.btn_hover = self.button_rect.collidepoint(mouse_pos)
        
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.user_name = self.user_name[:-1]
                elif event.key == pygame.K_RETURN:
                    self._continue()
                else:
                    if len(self.user_name) < 15 and (event.unicode.isalnum() or event.unicode == " "):
                        self.user_name += event.unicode
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.btn_hover:
                    self._continue()

    def _continue(self):
        if self.user_name.strip():
            user_manager.set_user(self.user_name.strip())
            from scenes.menu_scene import MenuScene
            self.switch_to(MenuScene)

    def update(self):
        self.cursor_timer += 1
        if self.cursor_timer >= 30:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0

    def draw(self, screen):
        screen.fill(COLORS['background'])
        
        # Dibujar Título con estilo vivo
        title_surf = self.font_large.render("¡Hola! ¿Cómo te llamas?", True, COLORS['primary'])
        title_rect = title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 150))
        # Sombra suave del título
        screen.blit(self.font_large.render("¡Hola! ¿Cómo te llamas?", True, (10, 10, 20)), (title_rect.x + 4, title_rect.y + 4))
        screen.blit(title_surf, title_rect)
        
        # Dibujar Campo de Texto
        pygame.draw.rect(screen, COLORS['card_bg'], self.input_rect, border_radius=15)
        pygame.draw.rect(screen, COLORS['primary'], self.input_rect, 3, border_radius=15)
        
        name_surf = self.font_medium.render(self.user_name, True, COLORS['white'])
        name_rect = name_surf.get_rect(center=self.input_rect.center)
        screen.blit(name_surf, name_rect)
        
        # Cursor
        if self.cursor_visible:
            cursor_x = name_rect.right + 5 if self.user_name else self.input_rect.left + 12
            pygame.draw.line(screen, COLORS['white'], (cursor_x, self.input_rect.centery - 20), (cursor_x, self.input_rect.centery + 20), 2)
            
        # Botón Continuar (Usando helper)
        btn_color = COLORS['success'] if self.user_name.strip() else COLORS['dark_gray']
        self.draw_button(screen, self.button_rect, "Continuar", btn_color, self.btn_hover)
