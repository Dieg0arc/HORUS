"""scenes/game_scene.py

Escena que envuelve SignLanguageGame e integra el juego con el sistema
de escenas de main.py sin anidar bucles de eventos.

Cambios v2:
- SignLanguageGame se inicializa en un hilo de fondo para que la pantalla
  de carga sea fluida (cámara y YOLO listo antes del primer frame jugable).
- process_events / update / draw delegan directamente al juego.
- Cuando el juego termina (ESC / Q), limpia recursos y vuelve al menú.
"""

from __future__ import annotations

import threading

import pygame

from scenes.base_scene import BaseScene
from core.config import COLORS, WIDTH, HEIGHT
from core.logger import get_logger

_log = get_logger("game_scene")


class GameScene(BaseScene):
    def __init__(self):
        super().__init__()
        self._game = None
        self._loading = True
        self._load_error: str | None = None

        # Cargar SignLanguageGame en un hilo daemon para que la pantalla de
        # carga se muestre mientras se abre la cámara y se preparan modelos.
        threading.Thread(target=self._init_game, daemon=True,
                         name="GameLoader").start()

    # ── Carga en background ───────────────────────────────────────────────────

    def _init_game(self):
        try:
            from game import SignLanguageGame
            self._game = SignLanguageGame()
        except Exception as exc:
            _log.error("Error al iniciar el juego: %s", exc, exc_info=True)
            self._load_error = str(exc)
        finally:
            self._loading = False

    # ── Interfaz BaseScene ────────────────────────────────────────────────────

    def process_events(self, events):
        if self._loading or self._game is None:
            return

        quit_requested = self._game.handle_events(events)
        if quit_requested:
            self._finish()

    def update(self):
        if self._loading:
            return

        # Si la carga falló, volver al menú
        if self._game is None:
            _log.error("Juego no disponible (error en carga): %s", self._load_error)
            from scenes.menu_scene import MenuScene
            self.switch_to(MenuScene)
            return

        self._game.update()

        # El juego puede pedir salida vía should_exit (p.ej. después de cleanup)
        if self._game.should_exit:
            self._finish()

    def draw(self, screen: pygame.Surface):
        if self._loading or self._game is None:
            self._draw_loading(screen)
        else:
            self._game.draw(screen)

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _finish(self):
        """Limpia el juego y regresa al menú."""
        if self._game is not None:
            try:
                self._game.cleanup()
            except Exception as exc:
                _log.warning("Error en cleanup del juego: %s", exc)
        from scenes.menu_scene import MenuScene
        self.switch_to(MenuScene)

    def _draw_loading(self, screen: pygame.Surface):
        screen.fill(COLORS['background'])

        title = self.font_large.render("Iniciando Juego...", True, COLORS['primary'])
        screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50)))

        if self._load_error:
            err = self.font_small.render(
                f"Error: {self._load_error[:80]}", True, COLORS['error'])
            screen.blit(err, err.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 60)))
        else:
            sub = self.font_medium.render(
                "Preparando cámara y detector...", True, COLORS['white'])
            screen.blit(sub, sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50)))
