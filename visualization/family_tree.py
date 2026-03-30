"""
Family tree viewer — pannable/zoomable overlay showing the genealogy of a
clicked creature.

Layout: nodes are positioned by generation (Y axis) and spread horizontally.
Dead creatures are drawn dimmed; the focus creature is highlighted.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import pygame

from core.lineage import LineageRecord, LineageRegistry
from visualization.colors import (
    TREE_BG, TREE_NODE_ALIVE, TREE_NODE_DEAD, TREE_NODE_FOCUS,
    TREE_EDGE, TREE_TEXT,
)

if TYPE_CHECKING:
    pass

NODE_RADIUS = 14
GENERATION_SPACING = 80    # pixels between generations
SIBLING_SPACING    = 50    # pixels between siblings


class FamilyTreeView:
    def __init__(self) -> None:
        self._font = None
        self._font_small = None
        self.visible: bool = False
        self.focus_id: int = -1

        # Pan state
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._dragging: bool = False
        self._drag_start: Tuple[int, int] = (0, 0)
        self._drag_offset_start: Tuple[float, float] = (0.0, 0.0)

        # Cached layout
        self._node_positions: Dict[int, Tuple[float, float]] = {}
        self._records: Dict[int, LineageRecord] = {}

    def _init_fonts(self) -> None:
        if self._font is None:
            self._font       = pygame.font.SysFont("monospace", 12, bold=True)
            self._font_small = pygame.font.SysFont("monospace", 11)

    def open(self, creature_id: int, lineage: LineageRegistry) -> None:
        self.focus_id = creature_id
        self.visible = True
        self._records = lineage.full_tree(creature_id)
        self._layout()
        # Center on focus node
        if creature_id in self._node_positions:
            fx, fy = self._node_positions[creature_id]
            self._offset_x = -fx
            self._offset_y = -fy

    def close(self) -> None:
        self.visible = False
        self.focus_id = -1

    def toggle(self, creature_id: int, lineage: LineageRegistry) -> None:
        if self.visible and self.focus_id == creature_id:
            self.close()
        else:
            self.open(creature_id, lineage)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _layout(self) -> None:
        """
        Assign (x, y) positions to each node.
        Y = generation * GENERATION_SPACING (younger generations higher on screen)
        X = spread evenly within each generation layer.
        """
        if not self._records:
            return

        # Group by generation
        gen_groups: Dict[int, List[int]] = {}
        for cid, rec in self._records.items():
            gen_groups.setdefault(rec.generation, []).append(cid)

        positions: Dict[int, Tuple[float, float]] = {}
        min_gen = min(gen_groups.keys())
        max_gen = max(gen_groups.keys())

        for gen, members in gen_groups.items():
            count = len(members)
            y = (max_gen - gen) * GENERATION_SPACING
            x_start = -(count - 1) * SIBLING_SPACING / 2.0
            for i, cid in enumerate(sorted(members)):
                positions[cid] = (x_start + i * SIBLING_SPACING, y)

        self._node_positions = positions

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return
        self._init_fonts()

        sw, sh = surface.get_size()
        panel_w = min(sw, 900)
        panel_h = min(sh, 700)
        panel_x = (sw - panel_w) // 2
        panel_y = (sh - panel_h) // 2

        # Background panel
        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surf.fill((*TREE_BG, 235))
        pygame.draw.rect(panel_surf, (60, 70, 90), (0, 0, panel_w, panel_h), 2)

        cx = panel_w // 2 + self._offset_x
        cy = panel_h // 2 + self._offset_y

        # ── Edges ─────────────────────────────────────────────────────────────
        for cid, rec in self._records.items():
            if rec.parent_ids:
                for pid in rec.parent_ids:
                    if pid in self._node_positions and cid in self._node_positions:
                        px, py = self._node_positions[pid]
                        nx, ny = self._node_positions[cid]
                        pygame.draw.line(
                            panel_surf, TREE_EDGE,
                            (int(cx + px), int(cy + py)),
                            (int(cx + nx), int(cy + ny)),
                            1,
                        )

        # ── Nodes ─────────────────────────────────────────────────────────────
        for cid, (nx, ny) in self._node_positions.items():
            rec = self._records.get(cid)
            if rec is None:
                continue
            sx = int(cx + nx)
            sy = int(cy + ny)

            if cid == self.focus_id:
                color = TREE_NODE_FOCUS
                radius = NODE_RADIUS + 3
            elif rec.is_alive:
                color = rec.color if rec.color != (200, 200, 200) else TREE_NODE_ALIVE
                radius = NODE_RADIUS
            else:
                color = TREE_NODE_DEAD
                radius = NODE_RADIUS - 2

            pygame.draw.circle(panel_surf, color, (sx, sy), radius)
            pygame.draw.circle(panel_surf, (200, 210, 220), (sx, sy), radius, 1)

            # Name label
            name_surf = self._font_small.render(rec.name_short, True, TREE_TEXT)
            panel_surf.blit(name_surf, (sx - name_surf.get_width() // 2, sy + radius + 2))

            # Lifespan for dead creatures
            if not rec.is_alive and rec.lifespan is not None:
                life_surf = self._font_small.render(
                    f"†{rec.lifespan}", True, (100, 100, 120)
                )
                panel_surf.blit(life_surf, (sx - life_surf.get_width() // 2, sy + radius + 14))

        # ── Title ─────────────────────────────────────────────────────────────
        focus_rec = self._records.get(self.focus_id)
        if focus_rec:
            title = self._font.render(
                f"Family Tree — {focus_rec.name_full}", True, (220, 225, 235)
            )
            panel_surf.blit(title, (10, 10))

        close_hint = self._font_small.render("Click creature again or press T to close", True, (80, 90, 110))
        panel_surf.blit(close_hint, (10, panel_h - 20))

        surface.blit(panel_surf, (panel_x, panel_y))

    # ── Input handling ────────────────────────────────────────────────────────

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Returns True if event was consumed."""
        if not self.visible:
            return False

        if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
            self.close()
            return True

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self._dragging = True
                self._drag_start = event.pos
                self._drag_offset_start = (self._offset_x, self._offset_y)
                return True

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self._dragging = False
                return True

        if event.type == pygame.MOUSEMOTION and self._dragging:
            dx = event.pos[0] - self._drag_start[0]
            dy = event.pos[1] - self._drag_start[1]
            self._offset_x = self._drag_offset_start[0] + dx
            self._offset_y = self._drag_offset_start[1] + dy
            return True

        return False
