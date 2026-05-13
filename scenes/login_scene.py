"""scenes/login_scene.py — Pantalla de identificación estilo HUD futurista."""

import pygame
from scenes.base_scene import BaseScene
from core.config import COLORS, WIDTH, HEIGHT
from core.user_manager import user_manager


class LoginScene(BaseScene):
    def __init__(self):
        super().__init__()
        self.user_name   = ""
        self.cursor_vis  = True
        self.cursor_timer = 0
        self.btn_hover   = False

        card_w, card_h = 420, 320
        cx = WIDTH  // 2 - card_w // 2
        cy = HEIGHT // 2 - card_h // 2

        self.card_rect  = pygame.Rect(cx, cy, card_w, card_h)
        self.input_rect = pygame.Rect(cx + 30, cy + 190, card_w - 60, 48)
        self.button_rect= pygame.Rect(cx + 30, cy + 258, card_w - 60, 44)

    # ── Eventos ───────────────────────────────────────────────────────────────

    def process_events(self, events):
        mouse = pygame.mouse.get_pos()
        self.btn_hover = self.button_rect.collidepoint(mouse)

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.user_name = self.user_name[:-1]
                elif event.key == pygame.K_RETURN:
                    self._continue()
                elif len(self.user_name) < 15 and (event.unicode.isalnum() or event.unicode == " "):
                    self.user_name += event.unicode.upper()
            if event.type == pygame.MOUSEBUTTONDOWN and self.btn_hover:
                self._continue()

    def _continue(self):
        if self.user_name.strip():
            user_manager.set_user(self.user_name.strip())
            from scenes.menu_scene import MenuScene
            self.switch_to(MenuScene)

    def update(self):
        self.cursor_timer += 1
        if self.cursor_timer >= 30:
            self.cursor_vis   = not self.cursor_vis
            self.cursor_timer = 0

    # ── Dibujo ────────────────────────────────────────────────────────────────

    def draw(self, screen):
        screen.fill(COLORS['background'])

        # Scan line de fondo (sutil, muy lenta)
        self.draw_scan_line(screen, (0, 0, WIDTH, HEIGHT), COLORS['primary'], speed=4.0, alpha=18)

        # Etiquetas de sistema (esquinas)
        self._draw_sys_labels(screen)

        # Tarjeta central
        self._draw_card(screen)

    def _draw_sys_labels(self, screen):
        """Etiquetas HUD en las esquinas."""
        ver = self.font_label.render("HORUS · SYS v2.0", True, COLORS['light_gray'])
        screen.blit(ver, (16, 14))

        if self.is_blink(1.2):
            online = self.font_label.render("● ONLINE", True, COLORS['success'])
            screen.blit(online, (WIDTH - online.get_width() - 16, 14))

        cam = self.font_label.render("CAM ● READY", True, COLORS['light_gray'])
        screen.blit(cam, (WIDTH - cam.get_width() - 16, HEIGHT - cam.get_height() - 14))

    def _draw_card(self, screen):
        cx = self.card_rect.x
        cy = self.card_rect.y
        cw = self.card_rect.width
        ch = self.card_rect.height

        # Fondo de tarjeta
        pygame.draw.rect(screen, COLORS['card_bg'], self.card_rect)
        pygame.draw.rect(screen, COLORS['primary'],  self.card_rect, 1)

        # Corner brackets (más grandes para la tarjeta principal)
        self.draw_corner_brackets(screen, (cx, cy, cw, ch), COLORS['primary'], size=20, thickness=2)

        # HORUS title
        title = self.font_title.render("HORUS", True, COLORS['primary'])
        screen.blit(title, title.get_rect(centerx=cx + cw // 2, top=cy + 22))

        # Subtítulo
        sub = self.font_label.render("LENGUA DE SEÑAS · IA", True, COLORS['light_gray'])
        screen.blit(sub, sub.get_rect(centerx=cx + cw // 2, top=cy + 22 + title.get_height() + 6))

        # Línea divisoria
        div_y = cy + 22 + title.get_height() + 30
        pygame.draw.line(screen, COLORS['dark_gray'], (cx + 30, div_y), (cx + cw - 30, div_y), 1)

        # Label del input
        id_lbl = self.font_label.render("IDENTIFICACIÓN DE AGENTE", True, COLORS['light_gray'])
        screen.blit(id_lbl, (cx + 30, div_y + 14))

        # Input field
        pygame.draw.rect(screen, COLORS['background'], self.input_rect)
        border_color = COLORS['primary'] if self.user_name else COLORS['dark_gray']
        pygame.draw.rect(screen, border_color, self.input_rect, 1)

        display = self.user_name
        if self.cursor_vis:
            display += "_"
        name_surf = self.font_medium.render(display, True, COLORS['primary'])
        screen.blit(name_surf, name_surf.get_rect(
            midleft=(self.input_rect.x + 12, self.input_rect.centery)))

        # Botón ACCEDER
        has_name  = bool(self.user_name.strip())
        btn_color = COLORS['primary'] if has_name else COLORS['dark_gray']
        self.draw_button(screen, self.button_rect, "ACCEDER  →",
                         btn_color, self.btn_hover and has_name)
