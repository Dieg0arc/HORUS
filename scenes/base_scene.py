"""scenes/base_scene.py — Clase base para todas las escenas de HORUS.

Carga las fuentes Orbitron (con fallback automático si no están descargadas)
y provee helpers de dibujado compartidos: botones angulares HUD,
corner brackets y scan line.
"""

import pygame
from core.config import COLORS, UI_CONFIG
from core.draw_utils import draw_corner_brackets, draw_scan_line, is_blink_visible


def _load_font(path, size):
    """Carga una fuente desde path si existe; de lo contrario usa pygame default."""
    if path:
        try:
            return pygame.font.Font(path, size)
        except Exception:
            pass
    return pygame.font.Font(None, size + 10)


class BaseScene:
    """Clase base para todas las escenas de la aplicación."""

    def __init__(self):
        self.next_scene = None

        bold    = UI_CONFIG['font_bold']
        regular = UI_CONFIG['font_regular']

        self.font_title  = _load_font(bold,    UI_CONFIG['font_size_title'])
        self.font_large  = _load_font(bold,    UI_CONFIG['font_size_large'])
        self.font_medium = _load_font(regular, UI_CONFIG['font_size_medium'])
        self.font_small  = _load_font(regular, UI_CONFIG['font_size_small'])
        self.font_label  = _load_font(regular, UI_CONFIG['font_size_label'])

    # ── Navegación ────────────────────────────────────────────────────────────

    def process_events(self, events):
        pass

    def update(self):
        pass

    def draw(self, screen):
        pass

    def switch_to(self, scene_class):
        self.next_scene = scene_class

    # ── Helpers de dibujado ───────────────────────────────────────────────────

    def draw_button(self, screen, rect, text: str, color, hover: bool = False,
                    text_color=None, border_only: bool = False):
        """Botón estilo HUD: angular, sin esquinas redondeadas."""
        thickness = UI_CONFIG['border_thickness']

        if border_only:
            bg = tuple(min(255, c + 15) for c in COLORS['card_bg']) if hover else COLORS['card_bg']
            pygame.draw.rect(screen, bg, rect)
            pygame.draw.rect(screen, color, rect, thickness)
            tc = text_color or color
        else:
            base = tuple(min(255, c + 20) for c in color) if hover else color
            pygame.draw.rect(screen, base, rect)
            tc = text_color or COLORS['background']

        text_surf = self.font_small.render(text, True, tc)
        screen.blit(text_surf, text_surf.get_rect(center=rect.center))

    def draw_corner_brackets(self, screen, rect, color=None, size=16, thickness=2):
        draw_corner_brackets(screen, rect, color or COLORS['primary'], size, thickness)

    def draw_scan_line(self, screen, area_rect, color=None, speed=2.5, alpha=40):
        draw_scan_line(screen, area_rect, color or COLORS['primary'], speed, alpha)

    @staticmethod
    def is_blink(freq=1.5) -> bool:
        return is_blink_visible(freq)
