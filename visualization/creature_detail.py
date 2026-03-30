"""
CreatureDetailPanel — shown when a creature is clicked.

Layout (right-anchored, full height, ~900px wide):
  ┌──────────────┬──────────────────┬──────────────────┐
  │  Family Tree │  Portrait + Name │  Stats + NN Viz  │
  │   (300px)    │     (280px)      │     (320px)      │
  └──────────────┴──────────────────┴──────────────────┘

Family tree:  scrollable lineage graph
Portrait:     large creature rendering, name, generation
Stats:        energy, age, species, brain complexity, etc.
NN Viz:       nodes and connections of the NEAT network
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import pygame

from core.appearance import AppearanceGene
from core.lineage import LineageRecord, LineageRegistry
from visualization.colors import (
    TREE_BG, TREE_NODE_ALIVE, TREE_NODE_DEAD, TREE_NODE_FOCUS, TREE_EDGE, TREE_TEXT,
)

if TYPE_CHECKING:
    from core.creature import Creature
    from core.genome import Genome
    from nn.network import NeuralNetwork

# ── Layout constants ──────────────────────────────────────────────────────────

PANEL_W       = 900
TREE_W        = 290
PORTRAIT_W    = 280
STATS_W       = PANEL_W - TREE_W - PORTRAIT_W   # 330

BG_COLOR      = (13, 15, 22, 230)
DIVIDER       = (45, 55, 75)
HEADER_BG     = (20, 25, 38)
TEXT_PRIMARY  = (220, 228, 240)
TEXT_DIM      = (110, 120, 140)
TEXT_ACCENT   = (100, 180, 255)
ENERGY_HI     = (80, 210, 120)
ENERGY_LO     = (210, 80,  80)

# NN node colors
NN_INPUT_COL  = ( 60, 130, 220)
NN_HIDDEN_COL = (140, 100, 200)
NN_OUTPUT_COL = ( 60, 200, 130)
NN_POS_EDGE   = ( 80, 200, 100)
NN_NEG_EDGE   = (210,  70,  70)
NN_DIS_EDGE   = ( 50,  55,  65)


class CreatureDetailPanel:
    def __init__(self) -> None:
        self.visible: bool = False
        self.creature_id: int = -1

        # Set to True for one frame when user clicks Save / Spawn
        self.save_requested:  bool = False
        self.spawn_requested: bool = False

        # Cached data
        self._creature: Optional["Creature"] = None
        self._tree_records: Dict[int, LineageRecord] = {}
        self._tree_positions: Dict[int, Tuple[float, float]] = {}

        # Tree scroll
        self._tree_scroll: float = 0.0

        # Fonts (lazy init)
        self._f_small  = None
        self._f_med    = None
        self._f_large  = None
        self._f_title  = None

        # Portrait surface cache
        self._portrait_surf: Optional[pygame.Surface] = None
        self._portrait_for_id: int = -1

        # Drag state for tree pan
        self._drag_active: bool = False
        self._drag_start_y: int = 0
        self._drag_scroll_start: float = 0.0

        # Button rects (screen coords, set during draw)
        self._save_btn_rect:  Optional[pygame.Rect] = None
        self._spawn_btn_rect: Optional[pygame.Rect] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def open(self, creature: "Creature", lineage: LineageRegistry) -> None:
        self.creature_id = creature.id
        self._creature = creature
        self.visible = True
        self._tree_records = lineage.full_tree(creature.id)
        self._layout_tree()
        self._tree_scroll = self._default_scroll()
        self._portrait_surf = None  # invalidate cache

    def close(self) -> None:
        self.visible = False
        self.creature_id = -1
        self._creature = None

    def refresh(self, creature: "Creature", lineage: LineageRegistry) -> None:
        """Call each frame while open to keep data current."""
        if creature and creature.id == self.creature_id:
            self._creature = creature

    def handle_event(self, event: pygame.event.Event, panel_x: int) -> bool:
        """Returns True if event consumed."""
        if not self.visible:
            return False

        if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
            self.close()
            return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            # Save button
            if self._save_btn_rect and self._save_btn_rect.collidepoint(mx, my):
                self.save_requested = True
                return True
            # Spawn button
            if self._spawn_btn_rect and self._spawn_btn_rect.collidepoint(mx, my):
                self.spawn_requested = True
                return True
            if mx >= panel_x:
                # In tree area?
                if mx < panel_x + TREE_W:
                    self._drag_active = True
                    self._drag_start_y = event.pos[1]
                    self._drag_scroll_start = self._tree_scroll
                return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._drag_active = False

        if event.type == pygame.MOUSEMOTION and self._drag_active:
            dy = event.pos[1] - self._drag_start_y
            self._tree_scroll = self._drag_scroll_start - dy
            return True

        if event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if mx >= panel_x and mx < panel_x + TREE_W:
                self._tree_scroll -= event.y * 30
                return True

        return False

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible or self._creature is None:
            return

        self._init_fonts()
        sw, sh = surface.get_size()
        panel_x = sw - PANEL_W

        # Dim world behind panel
        dim = pygame.Surface((PANEL_W, sh), pygame.SRCALPHA)
        dim.fill((8, 10, 16, 220))
        surface.blit(dim, (panel_x, 0))

        # Left border
        pygame.draw.line(surface, DIVIDER, (panel_x, 0), (panel_x, sh), 2)

        # Draw the three columns
        self._draw_tree(surface,    panel_x,                       sh)
        self._draw_portrait(surface, panel_x + TREE_W,             sh)
        self._draw_stats(surface,   panel_x + TREE_W + PORTRAIT_W, sh)

        # Vertical dividers
        pygame.draw.line(surface, DIVIDER,
            (panel_x + TREE_W, 0), (panel_x + TREE_W, sh), 1)
        pygame.draw.line(surface, DIVIDER,
            (panel_x + TREE_W + PORTRAIT_W, 0),
            (panel_x + TREE_W + PORTRAIT_W, sh), 1)

    # ── Family tree column ────────────────────────────────────────────────────

    def _layout_tree(self) -> None:
        if not self._tree_records:
            return
        gen_groups: Dict[int, List[int]] = {}
        for cid, rec in self._tree_records.items():
            gen_groups.setdefault(rec.generation, []).append(cid)

        positions: Dict[int, Tuple[float, float]] = {}
        max_gen = max(gen_groups.keys(), default=0)
        GEN_H = 70
        SIB_W = 46

        for gen, members in gen_groups.items():
            count = len(members)
            y = (max_gen - gen) * GEN_H + GEN_H
            x_start = -(count - 1) * SIB_W / 2.0
            for i, cid in enumerate(sorted(members)):
                positions[cid] = (x_start + i * SIB_W, y)

        self._tree_positions = positions
        self._tree_max_y = max((p[1] for p in positions.values()), default=0) + 80

    def _default_scroll(self) -> float:
        if self.creature_id in self._tree_positions:
            return self._tree_positions[self.creature_id][1] - 200
        return 0.0

    def _draw_tree(self, surface: pygame.Surface, ox: int, sh: int) -> None:
        # Header
        pygame.draw.rect(surface, HEADER_BG, (ox, 0, TREE_W, 26))
        self._text(surface, "FAMILY TREE", ox + 8, 6, self._f_med, TEXT_ACCENT)
        drag_hint = self._f_small.render("drag to scroll", True, TEXT_DIM)
        surface.blit(drag_hint, (ox + TREE_W - drag_hint.get_width() - 6, 7))

        # Clip to column
        clip = surface.subsurface(pygame.Rect(ox, 26, TREE_W, sh - 26))
        clip.fill((13, 15, 22))

        cx = TREE_W // 2
        scroll = int(self._tree_scroll)

        # Edges
        for cid, rec in self._tree_records.items():
            if cid not in self._tree_positions:
                continue
            nx, ny = self._tree_positions[cid]
            if rec.parent_ids:
                for pid in rec.parent_ids:
                    if pid not in self._tree_positions:
                        continue
                    px, py = self._tree_positions[pid]
                    pygame.draw.line(clip, DIVIDER,
                        (int(cx + nx), int(ny - scroll)),
                        (int(cx + px), int(py - scroll)), 1)

        # Nodes
        for cid, (nx, ny) in self._tree_positions.items():
            sx = int(cx + nx)
            sy = int(ny - scroll)
            if sy < -20 or sy > sh:
                continue
            rec = self._tree_records.get(cid)
            if rec is None:
                continue

            if cid == self.creature_id:
                col = TREE_NODE_FOCUS
                r = 12
            elif rec.is_alive:
                col = rec.color if rec.color != (200, 200, 200) else TREE_NODE_ALIVE
                r = 9
            else:
                col = TREE_NODE_DEAD
                r = 7

            pygame.draw.circle(clip, col, (sx, sy), r)
            pygame.draw.circle(clip, TEXT_DIM, (sx, sy), r, 1)

            name_s = self._f_small.render(rec.name_short, True, TEXT_PRIMARY)
            clip.blit(name_s, (sx - name_s.get_width() // 2, sy + r + 1))

            if not rec.is_alive and rec.lifespan:
                life_s = self._f_small.render(f"†{rec.lifespan}", True, TEXT_DIM)
                clip.blit(life_s, (sx - life_s.get_width() // 2, sy + r + 12))

    # ── Portrait column ────────────────────────────────────────────────────────

    def _draw_portrait(self, surface: pygame.Surface, ox: int, sh: int) -> None:
        c = self._creature
        if c is None:
            return

        # Header
        pygame.draw.rect(surface, HEADER_BG, (ox, 0, PORTRAIT_W, 26))
        self._text(surface, "CREATURE", ox + 8, 6, self._f_med, TEXT_ACCENT)

        # Name (large)
        name_s = self._f_large.render(c.name.short(), True, TEXT_PRIMARY)
        ny = 36
        surface.blit(name_s, (ox + PORTRAIT_W // 2 - name_s.get_width() // 2, ny))

        gen_s = self._f_med.render(f"Generation {c.name.generation}", True, TEXT_DIM)
        surface.blit(gen_s, (ox + PORTRAIT_W // 2 - gen_s.get_width() // 2, ny + 26))

        species_s = self._f_small.render(f"Species #{c.species_id}", True, TEXT_ACCENT)
        surface.blit(species_s, (ox + PORTRAIT_W // 2 - species_s.get_width() // 2, ny + 44))

        # Portrait render
        portrait_size = min(PORTRAIT_W - 40, 200)
        portrait_y = ny + 70
        portrait_x = ox + PORTRAIT_W // 2 - portrait_size // 2

        if self._portrait_for_id != c.id:
            self._portrait_surf = self._render_portrait(c, portrait_size)
            self._portrait_for_id = c.id

        if self._portrait_surf:
            surface.blit(self._portrait_surf, (portrait_x, portrait_y))

        # Energy bar below portrait
        bar_y = portrait_y + portrait_size + 12
        bar_w = PORTRAIT_W - 40
        bar_x = ox + 20
        bar_h = 14
        e_ratio = min(c.energy / c.max_energy, 1.0)
        ec = (int(ENERGY_LO[0] + (ENERGY_HI[0]-ENERGY_LO[0]) * e_ratio),
              int(ENERGY_LO[1] + (ENERGY_HI[1]-ENERGY_LO[1]) * e_ratio),
              int(ENERGY_LO[2] + (ENERGY_HI[2]-ENERGY_LO[2]) * e_ratio))
        pygame.draw.rect(surface, (30, 35, 45), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(surface, ec,           (bar_x, bar_y, int(bar_w * e_ratio), bar_h), border_radius=4)
        self._text(surface, f"Energy {c.energy:.0f} / {c.max_energy:.0f}",
                   bar_x, bar_y + bar_h + 4, self._f_small, TEXT_DIM)

        # Parents
        if c.parent_ids:
            par_y = bar_y + bar_h + 22
            pa = self._tree_records.get(c.parent_ids[0])
            pb = self._tree_records.get(c.parent_ids[1])
            pa_name = pa.name_short if pa else f"#{c.parent_ids[0]}"
            pb_name = pb.name_short if pb else f"#{c.parent_ids[1]}"
            self._text(surface, f"Parents:", ox + 8, par_y, self._f_small, TEXT_DIM)
            self._text(surface, pa_name, ox + 8, par_y + 13, self._f_small, TEXT_PRIMARY)
            self._text(surface, pb_name, ox + 8, par_y + 26, self._f_small, TEXT_PRIMARY)

        # ── Save / Spawn buttons (side-by-side at panel bottom) ──────────────
        total_btn_w = PORTRAIT_W - 40
        half_w      = (total_btn_w - 6) // 2   # 6px gap between buttons
        btn_h       = 28
        btn_y       = sh - 44
        mx_now, my_now = pygame.mouse.get_pos()

        # Save (left)
        save_rect = pygame.Rect(ox + 20, btn_y, half_w, btn_h)
        self._save_btn_rect = save_rect
        hov_save = save_rect.collidepoint(mx_now, my_now)
        pygame.draw.rect(surface, (60, 180, 100) if hov_save else (40, 130, 70),
                         save_rect, border_radius=5)
        lbl_save = self._f_med.render("SAVE", True, (230, 255, 235))
        surface.blit(lbl_save, (save_rect.centerx - lbl_save.get_width() // 2,
                                save_rect.centery - lbl_save.get_height() // 2))

        # Spawn (right)
        spawn_rect = pygame.Rect(ox + 20 + half_w + 6, btn_y, half_w, btn_h)
        self._spawn_btn_rect = spawn_rect
        hov_spawn = spawn_rect.collidepoint(mx_now, my_now)
        pygame.draw.rect(surface, (80, 130, 220) if hov_spawn else (50, 90, 170),
                         spawn_rect, border_radius=5)
        lbl_spawn = self._f_med.render("SPAWN COPY", True, (220, 235, 255))
        surface.blit(lbl_spawn, (spawn_rect.centerx - lbl_spawn.get_width() // 2,
                                 spawn_rect.centery - lbl_spawn.get_height() // 2))

    def _render_portrait(self, creature: "Creature", size: int) -> pygame.Surface:
        """Render a large portrait of the creature facing right."""
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))

        app = creature.genome.appearance
        cx = size // 2
        cy = size // 2
        cr = size * 0.28   # scale to fit
        angle = 0.0        # face right
        e_ratio = min(creature.energy / creature.max_energy, 1.0)

        primary = app.primary_rgb(e_ratio)
        belly   = app.belly_rgb(e_ratio)
        outline = app.outline_rgb()

        cos_a, sin_a = math.cos(angle), math.sin(angle)

        def to_pt(lx, ly):
            wx = cos_a * lx - sin_a * ly
            wy = sin_a * lx + cos_a * ly
            return (cx + int(wx), cy + int(wy))

        nl = app.body_length  * cr
        bw = app.body_width   * cr
        tl = app.tail_length  * cr
        tf = app.tail_fork    * cr
        fl = app.fin_length   * cr
        fpos = app.fin_position

        body_pts = [
            to_pt(nl,           0.0),
            to_pt(nl * 0.15,    bw),
            to_pt(-tl * 0.5,    bw * 0.55),
            to_pt(-tl,          0.0),
            to_pt(-tl * 0.5,   -bw * 0.55),
            to_pt(nl * 0.15,   -bw),
        ]
        pygame.draw.polygon(surf, primary, body_pts)

        if app.belly_ratio > 0.05:
            bs = 1.0 - app.belly_ratio
            belly_pts = [
                to_pt(nl * 0.85,          0.0),
                to_pt(nl * 0.10,          bw * bs),
                to_pt(-tl * 0.45,         bw * 0.45 * bs),
                to_pt(-tl * 0.85,         0.0),
                to_pt(-tl * 0.45,        -bw * 0.45 * bs),
                to_pt(nl * 0.10,         -bw * bs),
            ]
            pygame.draw.polygon(surf, belly, belly_pts)

        if tf > cr * 0.06:
            fork_base = -tl * 0.7
            pygame.draw.polygon(surf, primary, [
                to_pt(fork_base,      bw * 0.25),
                to_pt(-tl,            tf),
                to_pt(-tl + tf * 0.5, 0.0),
            ])
            pygame.draw.polygon(surf, primary, [
                to_pt(fork_base,      -bw * 0.25),
                to_pt(-tl,            -tf),
                to_pt(-tl + tf * 0.5,  0.0),
            ])

        if app.fin_count > 0 and fl > cr * 0.08:
            fin_x = nl * 0.1 - (nl * 0.1 + tl * 0.5) * fpos
            for side in range(app.fin_count):
                sign = 1 if side == 0 else -1
                fin_pts = [
                    to_pt(fin_x + fl * 0.35,  sign * bw * 0.85),
                    to_pt(fin_x - fl * 0.35,  sign * bw * 0.85),
                    to_pt(fin_x,               sign * (bw + fl)),
                ]
                pygame.draw.polygon(surf, primary, fin_pts)
                pygame.draw.polygon(surf, outline, fin_pts, 2)

        pygame.draw.polygon(surf, outline, body_pts, 2)

        # Eye
        eye_x, eye_y = to_pt(nl * 0.55, bw * 0.35)
        pygame.draw.circle(surf, (230, 235, 245), (eye_x, eye_y), max(2, int(cr * 0.12)))
        pygame.draw.circle(surf, (20, 20, 30),    (eye_x, eye_y), max(1, int(cr * 0.07)))

        return surf

    # ── Stats + NN column ─────────────────────────────────────────────────────

    def _draw_stats(self, surface: pygame.Surface, ox: int, sh: int) -> None:
        c = self._creature
        if c is None:
            return

        pygame.draw.rect(surface, HEADER_BG, (ox, 0, STATS_W, 26))
        self._text(surface, "STATS & BRAIN", ox + 8, 6, self._f_med, TEXT_ACCENT)

        y = 34
        row_h = 17

        # ── Stats ─────────────────────────────────────────────────────────────
        rec = self._tree_records.get(c.id)

        age_str = f"{c.years}y {c.age % 5000} ticks" if c.years > 0 else f"{c.age} ticks"
        beh = c.genome.behavior
        stamina_frac = c.stamina / max(beh.stamina_capacity, 1.0)
        stats = [
            ("Age",        age_str),
            ("Energy",     f"{c.energy:.1f} / {c.max_energy:.0f}"),
            ("Stamina",    f"{c.stamina:.0f} / {beh.stamina_capacity:.0f}  ({stamina_frac:.0%})"),
            ("Signal",     f"{c.signal:.2f}"),
            ("Speed",      f"{math.sqrt(float(c.vel[0]**2+c.vel[1]**2)):.2f}"),
            ("Species",    f"#{c.species_id}"),
            ("Generation", f"{c.name.generation}"),
            ("Nodes",      f"{len(c.genome.nodes)} ({len(c.genome.hidden_ids())} hidden)"),
            ("Weights",    f"{c.genome.weight_count()} active"),
            ("Food eaten", f"{c._food_eaten:,}"),
            ("Children",   f"{rec.children_count if rec else 0}"),
        ]

        for label, value in stats:
            self._text(surface, label + ":", ox + 8, y, self._f_small, TEXT_DIM)
            self._text(surface, value, ox + 110, y, self._f_small, TEXT_PRIMARY)
            y += row_h

        # ── Appearance traits ─────────────────────────────────────────────────
        y += 6
        self._text(surface, "── APPEARANCE ──", ox + 8, y, self._f_small, DIVIDER)
        y += row_h
        app = c.genome.appearance
        app_stats = [
            ("Body",   f"L={app.body_length:.2f}  W={app.body_width:.2f}"),
            ("Tail",   f"L={app.tail_length:.2f}  Fork={app.tail_fork:.2f}"),
            ("Fins",   f"{app.fin_count} fin(s)  L={app.fin_length:.2f}"),
        ]
        for label, value in app_stats:
            self._text(surface, label + ":", ox + 8, y, self._f_small, TEXT_DIM)
            self._text(surface, value, ox + 60, y, self._f_small, TEXT_PRIMARY)
            y += row_h

        # ── Behavior gene ─────────────────────────────────────────────────────
        y += 4
        self._text(surface, "── BEHAVIOR ──", ox + 8, y, self._f_small, DIVIDER)
        y += row_h
        # beh already set above in stats block
        # Friendly diet/repro labels
        cb = beh.carnivore_bias
        if cb < 0.25:
            diet_label = "Herbivore"
        elif cb > 0.75:
            diet_label = "Carnivore"
        else:
            diet_label = f"Omnivore"

        sb = beh.sexual_bias
        if sb < 0.2:
            repro_label = "Asexual"
        elif sb > 0.8:
            repro_label = "Sexual"
        else:
            repro_label = f"Mixed ({sb:.0%})"

        beh_stats = [
            ("Diet",     f"{diet_label} ({cb:.2f})"),
            ("Repro",    repro_label),
            ("Size",     f"{beh.size:.2f}  (max E {c.max_energy:.0f})"),
            ("Stamina",  f"cap {beh.stamina_capacity:.0f}  rec {beh.stamina_recovery:.2f}/t"),
            ("Ray len",  f"{beh.ray_length:.0f}"),
            ("Eat thr",  f"{beh.eat_threshold:.2f}"),
            ("Atk thr",  f"{beh.attack_threshold:.2f}"),
            ("Rep thr",  f"{beh.reproduce_threshold:.2f}"),
            ("Flee thr", f"{beh.flee_threshold:.2f}"),
            ("Food pri", f"{beh.food_priority:.2f}"),
        ]
        for label, value in beh_stats:
            self._text(surface, label + ":", ox + 8, y, self._f_small, TEXT_DIM)
            self._text(surface, value, ox + 80, y, self._f_small, TEXT_PRIMARY)
            y += row_h

        # ── Neural network visualization ──────────────────────────────────────
        y += 10
        nn_h = sh - y - 12
        if nn_h > 80:
            self._draw_nn(surface, ox, y, STATS_W, nn_h, c)

    def _draw_nn(self, surface: pygame.Surface, ox: int, oy: int,
                 w: int, h: int, creature: "Creature") -> None:
        self._text(surface, "── NEURAL NETWORK ──", ox + 8, oy, self._f_small, DIVIDER)
        oy += 16
        h -= 16

        genome  = creature.genome
        network = creature.brain

        if network._dirty:
            network._build_order()

        order = network._order
        nodes  = genome.nodes

        from core.genome import NodeType

        input_ids  = [n for n in order if nodes[n].node_type == NodeType.INPUT]
        output_ids = sorted([n for n in order if nodes[n].node_type == NodeType.OUTPUT])
        hidden_ids = [n for n in order if nodes[n].node_type == NodeType.HIDDEN]

        # Group inputs into 6 labelled bands (matches 59-input layout)
        input_groups = [
            ("Rays",      input_ids[:24]),
            ("Food",      input_ids[24:33]),
            ("Creatures", input_ids[33:51]),
            ("Hunger",    input_ids[51:53]),
            ("Self",      input_ids[53:58]),
            ("Stamina",   input_ids[58:59]),
        ]

        # Positions: inputs left column, hidden middle, outputs right column
        pad_x = 28
        col_in  = ox + pad_x
        col_out = ox + w - pad_x
        col_hid = (col_in + col_out) // 2

        def ys(items, total_h, margin=20):
            n = len(items)
            if n == 0:
                return {}
            step = (total_h - 2 * margin) / max(n - 1, 1)
            return {item: oy + margin + i * step for i, item in enumerate(items)}

        # Group input positions (one dot per group)
        group_ids = [g[0] for g in input_groups]
        group_y   = ys(group_ids, h)

        # Hidden and output positions
        hidden_y = ys(hidden_ids, h)
        output_y = ys(output_ids, h)

        output_labels = ["Thrust", "Turn", "Eat", "Reproduce", "Attack", "Flee", "Signal"]

        # Build a lookup: node_id -> (x, y) for connection drawing
        node_xy: Dict[int, Tuple[float, float]] = {}

        # Map individual input nodes to their group's y
        for label, group_members in input_groups:
            gy = group_y.get(label, oy + h // 2)
            for nid in group_members:
                node_xy[nid] = (col_in, gy)

        for nid in hidden_ids:
            node_xy[nid] = (col_hid, hidden_y[nid])
        for nid in output_ids:
            node_xy[nid] = (col_out, output_y[nid])

        # Draw connections first (behind nodes)
        for conn in genome.connections.values():
            if conn.in_node not in node_xy or conn.out_node not in node_xy:
                continue
            x1, y1 = node_xy[conn.in_node]
            x2, y2 = node_xy[conn.out_node]
            if not conn.enabled:
                pygame.draw.line(surface, NN_DIS_EDGE, (int(x1), int(y1)), (int(x2), int(y2)), 1)
            else:
                t = min(abs(conn.weight) / 3.0, 1.0)
                col = NN_POS_EDGE if conn.weight >= 0 else NN_NEG_EDGE
                dim_col = tuple(int(c * (0.3 + 0.7 * t)) for c in col)
                thickness = max(1, int(t * 2.5))
                pygame.draw.line(surface, dim_col, (int(x1), int(y1)), (int(x2), int(y2)), thickness)

        # Draw group input nodes
        for label, group_members in input_groups:
            gy = group_y.get(label, oy + h // 2)
            pygame.draw.circle(surface, NN_INPUT_COL, (col_in, int(gy)), 7)
            pygame.draw.circle(surface, TEXT_DIM,     (col_in, int(gy)), 7, 1)
            ls = self._f_small.render(label, True, TEXT_DIM)
            surface.blit(ls, (col_in - ls.get_width() - 6, int(gy) - 6))

        # Draw hidden nodes
        for nid in hidden_ids:
            hx, hy = node_xy[nid]
            pygame.draw.circle(surface, NN_HIDDEN_COL, (int(hx), int(hy)), 5)

        # Draw output nodes
        for i, nid in enumerate(output_ids):
            ox2, oy2 = node_xy[nid]
            # Activation level from last forward pass
            act = float(creature.last_action[i]) if i < len(creature.last_action) else 0.0
            t = (act + 1.0) / 2.0
            col = (int(NN_NEG_EDGE[0] + (NN_OUTPUT_COL[0]-NN_NEG_EDGE[0])*t),
                   int(NN_NEG_EDGE[1] + (NN_OUTPUT_COL[1]-NN_NEG_EDGE[1])*t),
                   int(NN_NEG_EDGE[2] + (NN_OUTPUT_COL[2]-NN_NEG_EDGE[2])*t))
            pygame.draw.circle(surface, col,      (int(ox2), int(oy2)), 8)
            pygame.draw.circle(surface, TEXT_DIM, (int(ox2), int(oy2)), 8, 1)
            lbl = output_labels[i] if i < len(output_labels) else str(i)
            ls = self._f_small.render(lbl, True, TEXT_DIM)
            surface.blit(ls, (int(ox2) + 11, int(oy2) - 6))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _init_fonts(self) -> None:
        if self._f_small is None:
            self._f_small = pygame.font.SysFont("monospace", 12)
            self._f_med   = pygame.font.SysFont("monospace", 13, bold=True)
            self._f_large = pygame.font.SysFont("monospace", 18, bold=True)
            self._f_title = pygame.font.SysFont("monospace", 22, bold=True)

    def _text(self, surface, text, x, y, font, color):
        s = font.render(str(text), True, color)
        surface.blit(s, (x, y))
