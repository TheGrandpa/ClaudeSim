"""
HUD — heads-up display drawn over the simulation viewport.

Left column (top):   tick / year / FPS / population / species / births / deaths / resources / speed
Left column (mid):   active species list with color swatches and member counts
Left column (bot):   oldest alive creatures
Bottom-left:         energy histogram
Bottom-right:        hint text

Clickable zones:
  Species rows  → opens detail for oldest member of that species
  Oldest rows   → opens detail for that creature
"""

from __future__ import annotations

import pygame
from typing import List, Optional, Tuple, TYPE_CHECKING

from visualization.colors import HUD_TEXT, HUD_ACCENT, RESOURCE

if TYPE_CHECKING:
    from core.population import Population
    from simulation.loop import SimulationLoop
    from evolution.speciation import Speciator

# Speed presets cycled by [ and ] keys
SPEED_PRESETS = [1, 2, 4, 8, 16]

# Max species rows to show before footer
MAX_SPECIES_ROWS = 15

# Click-zone hover color
_HOVER_TINT = (255, 255, 255, 30)


class HUD:
    def __init__(self) -> None:
        self._font_small = None
        self._font_med   = None
        self._initialized = False
        # List of (pygame.Rect, creature_id) populated each frame
        self._click_zones: List[Tuple[pygame.Rect, int]] = []

    def _init_fonts(self) -> None:
        if self._initialized:
            return
        self._font_small = pygame.font.SysFont("monospace", 13)
        self._font_med   = pygame.font.SysFont("monospace", 15, bold=True)
        self._initialized = True

    def get_creature_at(self, mx: int, my: int) -> Optional[int]:
        """Return creature_id if (mx, my) falls within a clickable zone, else None."""
        for rect, cid in self._click_zones:
            if rect.collidepoint(mx, my):
                return cid
        return None

    def draw(
        self,
        surface: pygame.Surface,
        loop: "SimulationLoop",
        population: "Population",
        speciator: "Speciator",
        fps: float,
        paused: bool,
    ) -> None:
        self._init_fonts()
        self._click_zones.clear()

        sw, sh = surface.get_size()
        mx, my = pygame.mouse.get_pos()

        from config import CONFIG
        year      = loop.tick_count // CONFIG.ticks_per_year
        year_tick = loop.tick_count % CONFIG.ticks_per_year

        # ── Top-left stats panel ──────────────────────────────────────────────
        lines = [
            ("YEAR",       f"{year}  (tick {year_tick:,})"),
            ("TOTAL TICK",  f"{loop.tick_count:,}"),
            ("FPS",        f"{fps:.0f}"),
            ("SPEED",      f"{CONFIG.sim_speed}x"),
            ("POPULATION", f"{population.count}"),
            ("SPECIES",    f"{speciator.species_count()}"),
            ("BIRTHS",     f"{loop.total_births:,}"),
            ("DEATHS",     f"{loop.total_deaths:,}"),
            ("RESOURCES",  f"{loop.world.resource_count}"),
        ]

        x, y = 10, 10
        for label, value in lines:
            lbl_surf = self._font_small.render(f"{label}: ", True, HUD_ACCENT)
            val_surf = self._font_small.render(value, True, HUD_TEXT)
            surface.blit(lbl_surf, (x, y))
            surface.blit(val_surf, (x + lbl_surf.get_width(), y))
            y += 18

        if paused:
            pause_surf = self._font_med.render("[ PAUSED ]", True, (255, 200, 60))
            surface.blit(pause_surf, (sw // 2 - pause_surf.get_width() // 2, 10))

        # ── Species list ──────────────────────────────────────────────────────
        y += 6
        species_bottom = self._draw_species_list(surface, speciator, population, x=10, y=y, mx=mx, my=my)
        oldest_bottom  = self._draw_oldest_creatures(surface, population, x=10, y=species_bottom + 8, mx=mx, my=my)

        # ── Energy histogram (bottom-left) ────────────────────────────────────
        self._draw_energy_histogram(surface, population, x=10, y=sh - 90)

        # Hint (bottom-right)
        hint = self._font_small.render(
            "O — options   I — saved creatures   [ / ] — speed",
            True, (65, 75, 95)
        )
        surface.blit(hint, (sw - hint.get_width() - 10, sh - 18))

    # ── Species list ──────────────────────────────────────────────────────────

    def _draw_species_list(
        self, surface: pygame.Surface, speciator: "Speciator",
        population: "Population",
        x: int, y: int, mx: int, my: int,
    ) -> int:
        """Returns the y position just below the last drawn row."""
        species_list = sorted(
            speciator.species,
            key=lambda sp: len(sp.members),
            reverse=True,
        )
        if not species_list:
            return y

        hdr = self._font_small.render("── ACTIVE SPECIES ──", True, (55, 70, 95))
        surface.blit(hdr, (x, y))
        y += 16

        row_h  = 17
        swatch = 10

        for sp in species_list[:MAX_SPECIES_ROWS]:
            color = sp.representative.appearance.primary_rgb()

            # Find oldest member for click action
            oldest_member_id: Optional[int] = None
            if sp.members:
                oldest = max(sp.members, key=lambda c: c.age)
                oldest_member_id = oldest.id

            # Build the full row rect for hover/click
            row_rect = pygame.Rect(x, y, 200, row_h)

            # Hover highlight
            hovering = row_rect.collidepoint(mx, my) and oldest_member_id is not None
            if hovering:
                hl = pygame.Surface((row_rect.width, row_rect.height), pygame.SRCALPHA)
                hl.fill((255, 255, 255, 25))
                surface.blit(hl, row_rect.topleft)

            # Register click zone
            if oldest_member_id is not None:
                self._click_zones.append((row_rect, oldest_member_id))

            # Color swatch
            pygame.draw.rect(surface, color, (x, y + (row_h - swatch) // 2, swatch, swatch))
            pygame.draw.rect(surface, (60, 70, 90), (x, y + (row_h - swatch) // 2, swatch, swatch), 1)

            # Species ID and count
            count = len(sp.members)
            stag  = sp.stagnation

            # Rep name from first member (if any)
            rep_name = sp.members[0].name.short() if sp.members else "?"

            id_col    = (180, 210, 255) if hovering else (130, 150, 180)
            name_col  = (140, 160, 185) if hovering else (100, 115, 135)

            id_txt    = self._font_small.render(f"#{sp.species_id}", True, id_col)
            count_txt = self._font_small.render(f"{count:>3}", True, HUD_TEXT)
            name_txt  = self._font_small.render(rep_name, True, name_col)

            # Stagnation warning tint
            if stag > 10:
                stag_col = (180, 80, 60)
                stag_s = self._font_small.render(f"s{stag}", True, stag_col)
            else:
                stag_s = None

            surface.blit(id_txt,    (x + swatch + 4, y))
            surface.blit(count_txt, (x + swatch + 38, y))
            surface.blit(name_txt,  (x + swatch + 66, y))
            if stag_s:
                surface.blit(stag_s, (x + swatch + 148, y))

            # Underline hint on hover
            if hovering:
                pygame.draw.line(surface, (100, 130, 180),
                                 (x + swatch + 4, y + row_h - 2),
                                 (x + swatch + 4 + id_txt.get_width() + 60, y + row_h - 2))

            y += row_h

        if len(species_list) > MAX_SPECIES_ROWS:
            hidden = len(species_list) - MAX_SPECIES_ROWS
            more = self._font_small.render(
                f"  +{hidden} more species…", True, (55, 70, 95)
            )
            surface.blit(more, (x, y))
            y += 16

        return y

    # ── Oldest creatures ──────────────────────────────────────────────────────

    def _draw_oldest_creatures(
        self, surface: pygame.Surface, population: "Population",
        x: int, y: int, mx: int, my: int,
    ) -> int:
        creatures = population.creatures
        if not creatures:
            return y

        from config import CONFIG
        oldest = sorted(creatures, key=lambda c: c.age, reverse=True)[:5]

        hdr = self._font_small.render("── OLDEST ALIVE ──", True, (55, 70, 95))
        surface.blit(hdr, (x, y))
        y += 16

        row_h  = 17
        swatch = 10

        for rank, c in enumerate(oldest, 1):
            color = c.genome.appearance.primary_rgb()

            # ── Click zone covers the name line ───────────────────────────────
            name_rect = pygame.Rect(x, y, 200, row_h)
            hovering  = name_rect.collidepoint(mx, my)
            if hovering:
                hl = pygame.Surface((name_rect.width, name_rect.height), pygame.SRCALPHA)
                hl.fill((255, 255, 255, 25))
                surface.blit(hl, name_rect.topleft)

            self._click_zones.append((name_rect, c.id))

            # ── Line 1: rank  [swatch]  name ──────────────────────────────────
            pygame.draw.rect(surface, color,
                             (x, y + (row_h - swatch) // 2, swatch, swatch))
            pygame.draw.rect(surface, (60, 70, 90),
                             (x, y + (row_h - swatch) // 2, swatch, swatch), 1)

            rank_s  = self._font_small.render(f"{rank}.", True, (80, 90, 110))
            name_col = (230, 245, 255) if hovering else (200, 215, 235)
            name_s  = self._font_small.render(c.name.short(), True, name_col)
            surface.blit(rank_s, (x + swatch + 2,  y))
            surface.blit(name_s, (x + swatch + 18, y))

            if hovering:
                nx = x + swatch + 18
                pygame.draw.line(surface, (140, 170, 210),
                                 (nx, y + row_h - 2),
                                 (nx + name_s.get_width(), y + row_h - 2))

            y += row_h

            # ── Line 2: age  (years + ticks) ──────────────────────────────────
            yrs = c.age // CONFIG.ticks_per_year
            tks = c.age %  CONFIG.ticks_per_year
            if yrs > 0:
                age_str = f"Age: {yrs}yr {tks:,}t  ({c.age:,} ticks)"
            else:
                age_str = f"Age: {tks:,} ticks"

            age_s = self._font_small.render(age_str, True, (110, 140, 90))
            surface.blit(age_s, (x + swatch + 18, y))
            y += row_h + 2   # small gap between entries

        return y

    # ── Energy histogram ──────────────────────────────────────────────────────

    def _draw_energy_histogram(
        self, surface: pygame.Surface, population: "Population",
        x: int, y: int
    ) -> None:
        creatures = population.creatures
        if not creatures:
            return

        from config import CONFIG
        buckets   = 20
        max_e     = CONFIG.max_energy
        counts    = [0] * buckets
        for c in creatures:
            b = min(buckets - 1, int(c.energy / max_e * buckets))
            counts[b] += 1

        bar_w     = 6
        bar_gap   = 1
        max_count = max(counts) or 1
        bar_max_h = 55
        total_bar_w = buckets * (bar_w + bar_gap) - bar_gap

        # Padding inside the frame
        pad = 5
        frame_x = x - pad
        frame_y = y - pad
        frame_w = total_bar_w + pad * 2
        frame_h = bar_max_h + pad * 2 + 14  # +14 for x-axis labels

        # Title above frame
        label = self._font_small.render("ENERGY DISTRIBUTION", True, (100, 115, 140))
        surface.blit(label, (frame_x, frame_y - 15))

        # Frame background + border
        pygame.draw.rect(surface, (14, 18, 28), (frame_x, frame_y, frame_w, frame_h))
        pygame.draw.rect(surface, (45, 58, 80), (frame_x, frame_y, frame_w, frame_h), 1)

        # Bars
        for i, count in enumerate(counts):
            bh = max(1, int(count / max_count * bar_max_h)) if count else 0
            hue_frac = i / (buckets - 1)
            r = int(min(255, hue_frac * 2 * 255))
            g = int(min(255, (1 - hue_frac) * 2 * 255))
            color = (r, g, 40)
            bx = x + i * (bar_w + bar_gap)
            rect = pygame.Rect(bx, y + bar_max_h - bh, bar_w, bh)
            pygame.draw.rect(surface, color, rect)

        # X-axis baseline
        pygame.draw.line(surface, (55, 65, 85),
                         (x, y + bar_max_h), (x + total_bar_w, y + bar_max_h), 1)

        # X-axis labels: 0, 250, 500
        for val, frac in ((0, 0.0), (int(max_e // 2), 0.5), (int(max_e), 1.0)):
            lx = x + int(frac * total_bar_w)
            tick_s = self._font_small.render(str(val), True, (70, 85, 105))
            surface.blit(tick_s, (lx - tick_s.get_width() // 2, y + bar_max_h + 3))
            pygame.draw.line(surface, (55, 65, 85), (lx, y + bar_max_h), (lx, y + bar_max_h + 2))

        # Key: colored gradient strip + "Low" / "High" labels
        key_y = frame_y - 30
        key_x = frame_x
        key_w = total_bar_w + pad * 2
        key_h = 7
        for i in range(key_w):
            frac = i / max(key_w - 1, 1)
            r = int(min(255, frac * 2 * 255))
            g = int(min(255, (1 - frac) * 2 * 255))
            pygame.draw.line(surface, (r, g, 40),
                             (key_x + i, key_y), (key_x + i, key_y + key_h))
        pygame.draw.rect(surface, (45, 58, 80), (key_x, key_y, key_w, key_h), 1)

        lo_s = self._font_small.render("Low", True, (160, 70, 40))
        hi_s = self._font_small.render("High", True, (60, 180, 60))
        surface.blit(lo_s, (key_x, key_y - 13))
        surface.blit(hi_s, (key_x + key_w - hi_s.get_width(), key_y - 13))
