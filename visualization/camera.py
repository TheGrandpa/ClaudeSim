"""
Camera — pan, zoom, and coordinate transforms for the pygame renderer.
"""

from __future__ import annotations

import pygame
from config import SimConfig, CONFIG


class Camera:
    def __init__(self, cfg: SimConfig = CONFIG) -> None:
        self.cfg = cfg
        self.zoom: float = cfg.camera_initial_zoom
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        self._follow_creature_id: int = -1

    # ── Transform ─────────────────────────────────────────────────────────────

    def world_to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        sx = int((wx - self.offset_x) * self.zoom)
        sy = int((wy - self.offset_y) * self.zoom)
        return sx, sy

    def screen_to_world(self, sx: int, sy: int) -> tuple[float, float]:
        wx = sx / self.zoom + self.offset_x
        wy = sy / self.zoom + self.offset_y
        return wx, wy

    def world_to_screen_radius(self, r: float) -> int:
        return max(1, int(r * self.zoom))

    # ── Control ───────────────────────────────────────────────────────────────

    def pan(self, dx: float, dy: float) -> None:
        self.offset_x += dx / self.zoom
        self.offset_y += dy / self.zoom
        self._follow_creature_id = -1

    def zoom_in(self) -> None:
        self.zoom = min(self.cfg.camera_max_zoom, self.zoom * (1 + self.cfg.camera_zoom_speed))

    def zoom_out(self) -> None:
        self.zoom = max(self.cfg.camera_min_zoom, self.zoom / (1 + self.cfg.camera_zoom_speed))

    def zoom_around(self, sx: int, sy: int, factor: float) -> None:
        """Zoom centered on screen point (sx, sy)."""
        wx, wy = self.screen_to_world(sx, sy)
        self.zoom = max(self.cfg.camera_min_zoom,
                        min(self.cfg.camera_max_zoom, self.zoom * factor))
        # Adjust offset so world point stays under cursor
        self.offset_x = wx - sx / self.zoom
        self.offset_y = wy - sy / self.zoom
        self._follow_creature_id = -1

    def snap_to_world(self, screen_w: int, screen_h: int) -> None:
        """Zoom to fit entire world on screen."""
        self._follow_creature_id = -1
        zoom_x = screen_w / self.cfg.world_width
        zoom_y = screen_h / self.cfg.world_height
        self.zoom = min(zoom_x, zoom_y) * 0.98
        self.offset_x = -(screen_w / self.zoom - self.cfg.world_width) / 2
        self.offset_y = -(screen_h / self.zoom - self.cfg.world_height) / 2

    def follow(self, creature_id: int) -> None:
        self._follow_creature_id = creature_id

    def update_follow(self, population, screen_w: int, screen_h: int) -> None:
        if self._follow_creature_id < 0:
            return
        creature = population.get_by_id(self._follow_creature_id)
        if creature is None:
            self._follow_creature_id = -1
            return
        # Center creature on screen
        self.offset_x = float(creature.pos[0]) - screen_w / (2 * self.zoom)
        self.offset_y = float(creature.pos[1]) - screen_h / (2 * self.zoom)

    @property
    def following(self) -> int:
        return self._follow_creature_id
