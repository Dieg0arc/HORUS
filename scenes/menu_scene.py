"""scenes/menu_scene.py — Menú principal estilo HUD futurista."""

import sys
import pygame
from scenes.base_scene import BaseScene
from core.config import COLORS, WIDTH, HEIGHT
from core.user_manager import user_manager


class MenuScene(BaseScene):
    def __init__(self):
        super().__init__()

        btn_w, btn_h = 320, 56
        spacing      = 14
        start_y      = HEIGHT // 2 - 50

        self.buttons = [
            {"label": "01  APRENDER",  "rect": pygame.Rect(WIDTH // 2 - btn_w // 2, start_y,                     btn_w, btn_h), "color": COLORS['primary'],   "action": "learn",  "hover": False, "border_only": True},
            {"label": "02  JUGAR",     "rect": pygame.Rect(WIDTH // 2 - btn_w // 2, start_y + (btn_h + spacing), btn_w, btn_h), "color": COLORS['primary'],   "action": "play",   "hover": False, "border_only": False},
            {"label": "03  SALIR",     "rect": pygame.Rect(WIDTH // 2 - btn_w // 2, start_y + (btn_h + spacing) * 2, btn_w, btn_h), "color": COLORS['dark_gray'], "action": "exit",   "hover": False, "border_only": True},
        ]
        self._selected_idx = 1

    # ── Eventos ───────────────────────────────────────────────────────────────

    def process_events(self, events):
        mouse = pygame.mouse.get_pos()
        for btn in self.buttons:
            btn["hover"] = btn["rect"].collidepoint(mouse)

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

    # ── Dibujo ────────────────────────────────────────────────────────────────

    def draw(self, screen):
        screen.fill(COLORS['background'])

        # Scan line de fondo
        self.draw_scan_line(screen, (0, 0, WIDTH, HEIGHT), COLORS['primary'], speed=3.5, alpha=18)

        # HUD info bar superior
        self._draw_top_bar(screen)

        # Nombre del agente + bienvenida
        self._draw_welcome(screen)

        # Botones
        for i, btn in enumerate(self.buttons):
            is_active = btn["hover"] or (i == self._selected_idx)
            color = COLORS['primary'] if is_active and btn["action"] != "exit" else btn["color"]
            self.draw_button(screen, btn["rect"], btn["label"], color,
                             hover=is_active,
                             border_only=btn["border_only"])

        # Indicador de sistema en la parte inferior
        self._draw_bottom_bar(screen)

    def _draw_top_bar(self, screen):
        name  = user_manager.get_user_name()
        best  = user_manager.get_best_score()
        left  = self.font_label.render("HORUS v2.0", True, COLORS['light_gray'])
        right = self.font_label.render(f"AGENTE: {name}   |   MEJOR: {best}", True, COLORS['light_gray'])
        screen.blit(left,  (16, 14))
        screen.blit(right, (WIDTH - right.get_width() - 16, 14))
        # Línea separadora
        pygame.draw.line(screen, COLORS['dark_gray'], (0, 36), (WIDTH, 36), 1)

    def _draw_welcome(self, screen):
        sub  = self.font_label.render("BIENVENIDO", True, COLORS['light_gray'])
        name = user_manager.get_user_name()
        big  = self.font_title.render(name.upper(), True, COLORS['primary'])
        center_x = WIDTH // 2
        # Posicionar arriba de los botones
        btn_top = self.buttons[0]["rect"].top
        big_y   = btn_top - big.get_height() - 20
        sub_y   = big_y - sub.get_height() - 6
        screen.blit(sub,  sub.get_rect(centerx=center_x, top=sub_y))
        screen.blit(big,  big.get_rect(centerx=center_x, top=big_y))

    def _draw_bottom_bar(self, screen):
        pygame.draw.line(screen, COLORS['dark_gray'], (0, HEIGHT - 36), (WIDTH, HEIGHT - 36), 1)
        if self.is_blink(0.8):
            mod = self.font_label.render("MÓDULO DE DETECCIÓN · ACTIVO", True, COLORS['light_gray'])
            screen.blit(mod, mod.get_rect(centerx=WIDTH // 2, centery=HEIGHT - 18))
