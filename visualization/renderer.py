"""
PygameRenderer — main rendering loop.

Draws: world background, resources, creatures (with trails/rays if enabled),
HUD overlay, and delegates family tree rendering to FamilyTreeView.
"""

from __future__ import annotations

import math
from typing import List, Optional, TYPE_CHECKING

import pygame

from config import SimConfig, CONFIG
from visualization.camera import Camera
from visualization.colors import (
    BG, WORLD_BG, WORLD_BORDER, RESOURCE, RESOURCE_DIM,
    HUD_TEXT, RAY_COLOR, TRAIL_COLOR,
)
from visualization.hud import HUD
from visualization.creature_detail import CreatureDetailPanel, PANEL_W
from visualization.event_log import get_log
from visualization.options_menu import OptionsMenu
from visualization.creature_picker import CreaturePickerModal

if TYPE_CHECKING:
    from core.creature import Creature
    from core.population import Population
    from core.lineage import LineageRegistry
    from simulation.loop import SimulationLoop
    from evolution.speciation import Speciator


class Renderer:
    def __init__(self, cfg: SimConfig = CONFIG) -> None:
        self.cfg = cfg
        self.screen = pygame.display.set_mode(
            (cfg.window_width, cfg.window_height), pygame.RESIZABLE
        )
        pygame.display.set_caption("Evolution Simulation")
        self.camera = Camera(cfg)
        self.hud = HUD()
        self.detail = CreatureDetailPanel()
        self.options = OptionsMenu(cfg)
        self.picker = CreaturePickerModal()
        self._clock = pygame.time.Clock()
        self._fps: float = 0.0

        # Fit world on startup
        self.camera.snap_to_world(cfg.window_width, cfg.window_height)

    # ── Main render ───────────────────────────────────────────────────────────

    def render(
        self,
        loop: "SimulationLoop",
        population: "Population",
        lineage: "LineageRegistry",
        speciator: "Speciator",
        paused: bool,
    ) -> None:
        sw, sh = self.screen.get_size()
        self.camera.update_follow(population, sw, sh)

        # Refresh detail panel with live creature data
        if self.detail.visible:
            c = population.get_by_id(self.detail.creature_id)
            if c:
                self.detail.refresh(c, lineage)
            else:
                self.detail.close()

        self.screen.fill(BG)

        # World background rect
        wx0, wy0 = self.camera.world_to_screen(0, 0)
        wx1, wy1 = self.camera.world_to_screen(self.cfg.world_width, self.cfg.world_height)
        world_rect = pygame.Rect(wx0, wy0, wx1 - wx0, wy1 - wy0)
        pygame.draw.rect(self.screen, WORLD_BG, world_rect)
        pygame.draw.rect(self.screen, WORLD_BORDER, world_rect, 2)

        self._draw_resources(loop.world)
        self._draw_creatures(population)
        self.hud.draw(self.screen, loop, population, speciator, self._fps, paused)
        if self.detail.visible:
            self.detail.draw(self.screen)
        else:
            get_log().draw(self.screen)
        self.options.draw(self.screen)
        self.picker.draw(self.screen)

        pygame.display.flip()
        self._fps = self._clock.tick(self.cfg.target_fps)
        self._fps = self._clock.get_fps()

    # ── Resources ─────────────────────────────────────────────────────────────

    def _draw_resources(self, world) -> None:
        min_e = self.cfg.resource_min_energy
        max_e = self.cfg.resource_max_energy
        base_r = self.camera.world_to_screen_radius(self.cfg.resource_radius)
        base_r = max(2, base_r)

        for res in world.active_resources:
            sx, sy = self.camera.world_to_screen(res.x, res.y)
            sw, sh = self.screen.get_size()
            if sx < -base_r or sx > sw + base_r or sy < -base_r or sy > sh + base_r:
                continue
            t = (res.energy - min_e) / max(max_e - min_e, 1.0)
            color = (
                int(RESOURCE_DIM[0] + (RESOURCE[0] - RESOURCE_DIM[0]) * t),
                int(RESOURCE_DIM[1] + (RESOURCE[1] - RESOURCE_DIM[1]) * t),
                int(RESOURCE_DIM[2] + (RESOURCE[2] - RESOURCE_DIM[2]) * t),
            )
            r = max(2, int(base_r * (0.6 + 0.4 * t)))
            pygame.draw.circle(self.screen, color, (sx, sy), r)

    # ── Creatures ─────────────────────────────────────────────────────────────

    def _draw_creatures(self, population: "Population") -> None:
        base_cr = self.camera.world_to_screen_radius(self.cfg.creature_radius)

        for creature in population.creatures:
            # Scale rendered radius by the creature's size gene
            cr = max(3, int(base_cr * creature.genome.behavior.size))

            sx, sy = self.camera.world_to_screen(
                float(creature.pos[0]), float(creature.pos[1])
            )
            sw, sh = self.screen.get_size()
            if sx < -cr * 3 or sx > sw + cr * 3 or sy < -cr * 3 or sy > sh + cr * 3:
                continue

            # Trail
            if self.cfg.show_trails and len(creature._trail) > 1:
                self._draw_trail(creature, cr)

            # Ray visualization
            if self.cfg.show_rays:
                self._draw_rays(creature, cr)

            # Body (triangle pointing in heading direction)
            self._draw_creature_body(creature, sx, sy, cr)

            # Name label (only when zoomed in enough)
            if self.camera.zoom > 0.8:
                self._draw_name(creature, sx, sy, cr)

            # Follow indicator
            if self.camera.following == creature.id:
                pygame.draw.circle(self.screen, (255, 255, 100), (sx, sy), cr + 4, 2)

    def _draw_creature_body(self, creature: "Creature", sx: int, sy: int, cr: int) -> None:
        app = creature.genome.appearance
        angle = creature.angle
        e_ratio = min(creature.energy / creature.max_energy, 1.0)

        primary  = app.primary_rgb(e_ratio)
        belly    = app.belly_rgb(e_ratio)

        # Blend outline toward red for carnivores, toward green for herbivores
        cb = creature.genome.behavior.carnivore_bias
        base_out = app.outline_rgb()
        outline = (
            int(base_out[0] * (1 - cb) + 220 * cb),
            int(base_out[1] * (1 - cb) +  40 * cb),
            int(base_out[2] * (1 - cb) +  40 * cb),
        )

        # ── Local space helpers ───────────────────────────────────────────────
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        def to_screen(lx: float, ly: float):
            wx = cos_a * lx - sin_a * ly
            wy = sin_a * lx + cos_a * ly
            return (sx + int(wx), sy + int(wy))

        nl  = app.body_length  * cr    # nose length
        bw  = app.body_width   * cr    # body half-width
        tl  = app.tail_length  * cr    # tail length
        tf  = app.tail_fork    * cr    # tail fork spread
        fl  = app.fin_length   * cr    # fin projection
        fpos = app.fin_position        # 0=mid, 1=back

        # ── Main body polygon (6 points) ──────────────────────────────────────
        # nose_tip, right_shoulder, right_back, tail, left_back, left_shoulder
        body_pts = [
            to_screen(nl,               0.0),
            to_screen(nl * 0.15,        bw),
            to_screen(-tl * 0.5,        bw * 0.55),
            to_screen(-tl,              0.0),
            to_screen(-tl * 0.5,       -bw * 0.55),
            to_screen(nl * 0.15,       -bw),
        ]

        pygame.draw.polygon(self.screen, primary, body_pts)

        # ── Belly stripe (inner polygon, secondary color) ─────────────────────
        if app.belly_ratio > 0.05:
            belly_shrink = 1.0 - app.belly_ratio
            belly_pts = [
                to_screen(nl * 0.85,                 0.0),
                to_screen(nl * 0.10,                 bw * belly_shrink),
                to_screen(-tl * 0.45,                bw * 0.45 * belly_shrink),
                to_screen(-tl * 0.85,                0.0),
                to_screen(-tl * 0.45,               -bw * 0.45 * belly_shrink),
                to_screen(nl * 0.10,                -bw * belly_shrink),
            ]
            pygame.draw.polygon(self.screen, belly, belly_pts)

        # ── Forked tail ───────────────────────────────────────────────────────
        if tf > cr * 0.08:
            fork_base_x = -tl * 0.7
            tail_pts_top = [
                to_screen(fork_base_x,  bw * 0.25),
                to_screen(-tl,          tf),
                to_screen(-tl + tf * 0.5, 0.0),
            ]
            tail_pts_bot = [
                to_screen(fork_base_x, -bw * 0.25),
                to_screen(-tl,         -tf),
                to_screen(-tl + tf * 0.5, 0.0),
            ]
            pygame.draw.polygon(self.screen, primary, tail_pts_top)
            pygame.draw.polygon(self.screen, primary, tail_pts_bot)

        # ── Fins ──────────────────────────────────────────────────────────────
        if app.fin_count > 0 and fl > cr * 0.1:
            # fin base position along body (local x)
            fin_x = nl * 0.1 - (nl * 0.1 + tl * 0.5) * fpos

            for side in range(app.fin_count):
                sign = 1 if side == 0 else -1
                fin_pts = [
                    to_screen(fin_x + fl * 0.35,  sign * bw * 0.85),
                    to_screen(fin_x - fl * 0.35,  sign * bw * 0.85),
                    to_screen(fin_x,               sign * (bw + fl)),
                ]
                pygame.draw.polygon(self.screen, primary, fin_pts)
                pygame.draw.polygon(self.screen, outline, fin_pts, 1)

        # ── Outline ───────────────────────────────────────────────────────────
        pygame.draw.polygon(self.screen, outline, body_pts, 1)

        # ── Diet & reproduction mode indicators (only when zoomed in) ─────────
        if cr >= 10:
            beh = creature.genome.behavior
            # Small filled circle at the tail: blue = sexual, yellow = asexual
            tail_sx, tail_sy = to_screen(-tl, 0.0)
            repro_col = (80, 140, 255) if beh.sexual_bias > 0.5 else (255, 220, 50)
            pygame.draw.circle(self.screen, repro_col, (tail_sx, tail_sy), max(2, cr // 5))

    def _draw_trail(self, creature: "Creature", cr: int) -> None:
        trail = creature._trail
        if len(trail) < 2:
            return
        from config import CONFIG
        # Threshold: if two consecutive world points are more than half the world
        # size apart, the creature wrapped — skip that segment to avoid a line
        # shooting all the way across the screen.
        half_w = CONFIG.world_width  * 0.5
        half_h = CONFIG.world_height * 0.5
        dim = tuple(int(c * 0.3) for c in creature.color[:3])
        n = len(trail)
        for i in range(1, n):
            x0, y0 = trail[i - 1]
            x1, y1 = trail[i]
            if abs(x1 - x0) > half_w or abs(y1 - y0) > half_h:
                continue   # skip wrap-crossing segment
            sx0, sy0 = self.camera.world_to_screen(x0, y0)
            sx1, sy1 = self.camera.world_to_screen(x1, y1)
            brightness = int(80 * i / n)
            seg_col = tuple(min(255, int(c * brightness / 80)) for c in creature.color[:3])
            pygame.draw.line(self.screen, seg_col, (sx0, sy0), (sx1, sy1), 1)

    def _draw_rays(self, creature: "Creature", cr: int) -> None:
        from config import CONFIG
        ray_length = creature.genome.behavior.ray_length
        for i in range(CONFIG.ray_count):
            angle = creature.angle + i * (2 * math.pi / CONFIG.ray_count)
            ex = float(creature.pos[0]) + math.cos(angle) * ray_length
            ey = float(creature.pos[1]) + math.sin(angle) * ray_length
            sx0, sy0 = self.camera.world_to_screen(float(creature.pos[0]), float(creature.pos[1]))
            sx1, sy1 = self.camera.world_to_screen(ex, ey)
            pygame.draw.line(self.screen, (60, 80, 120), (sx0, sy0), (sx1, sy1), 1)

    def _draw_name(self, creature: "Creature", sx: int, sy: int, cr: int) -> None:
        if not hasattr(self, "_name_font"):
            self._name_font = pygame.font.SysFont("monospace", 11)
        name_surf = self._name_font.render(creature.name.short(), True, (180, 190, 200))
        self.screen.blit(name_surf, (sx - name_surf.get_width() // 2, sy - cr - 14))

    # ── Pick ──────────────────────────────────────────────────────────────────

    def panel_x(self) -> int:
        sw, _ = self.screen.get_size()
        return sw - PANEL_W

    def pick_creature(self, screen_x: int, screen_y: int, population: "Population") -> Optional[int]:
        """Return id of creature closest to screen click, or None."""
        wx, wy = self.camera.screen_to_world(screen_x, screen_y)
        best_id = None
        # Scale pick radius inversely with zoom so it stays ~15px on screen
        pick_radius = max(self.cfg.creature_radius * 3, 15.0 / max(self.camera.zoom, 0.01))
        best_dist = pick_radius
        for creature in population.creatures:
            dx = float(creature.pos[0]) - wx
            dy = float(creature.pos[1]) - wy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_id = creature.id
        return best_id
