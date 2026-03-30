"""
OptionsMenu — centered modal overlay with three tabs.

  CONTROLS    — keybindings reference table
  SIMULATION  — live config tweaks (+/- buttons and toggles)
  EVENT LOG   — per-type suppression toggles
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Callable

import pygame

from config import SimConfig, CONFIG
from visualization.event_log import get_log, ALL_TYPES, _TYPE_COLORS

# ── Palette ───────────────────────────────────────────────────────────────────

BG          = (14, 16, 24, 245)
PANEL_BG    = (18, 22, 32)
HEADER_BG   = (22, 28, 42)
BORDER      = (50, 60, 85)
TAB_ACTIVE  = (40, 55, 90)
TAB_IDLE    = (22, 28, 42)
TEXT        = (220, 228, 240)
TEXT_DIM    = (100, 115, 140)
TEXT_ACCENT = (100, 180, 255)
BTN_BG      = (35, 45, 68)
BTN_HOV     = (55, 70, 105)
BTN_ACT     = (70, 130, 200)
TOGGLE_ON   = (60, 190, 110)
TOGGLE_OFF  = (80,  50,  50)

PANEL_W = 720
PANEL_H = 520
TAB_H   = 36

TABS = ["CONTROLS", "SIMULATION", "EVENT LOG", "CHARTS"]


# ── Simple widget helpers ─────────────────────────────────────────────────────

class _Button:
    def __init__(self, rect: pygame.Rect, label: str,
                 color=BTN_BG, hover_color=BTN_HOV):
        self.rect = rect
        self.label = label
        self.color = color
        self.hover_color = hover_color
        self.hovered = False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font,
             override_color=None) -> None:
        col = override_color or (self.hover_color if self.hovered else self.color)
        pygame.draw.rect(surface, col, self.rect, border_radius=4)
        pygame.draw.rect(surface, BORDER, self.rect, 1, border_radius=4)
        ts = font.render(self.label, True, TEXT)
        surface.blit(ts, (self.rect.centerx - ts.get_width() // 2,
                          self.rect.centery - ts.get_height() // 2))

    def update_hover(self, mx: int, my: int) -> None:
        self.hovered = self.rect.collidepoint(mx, my)

    def clicked(self, mx: int, my: int) -> bool:
        return self.rect.collidepoint(mx, my)


# ── Options Menu ──────────────────────────────────────────────────────────────

class OptionsMenu:
    def __init__(self, cfg: SimConfig = CONFIG) -> None:
        self.cfg = cfg
        self.visible = False
        self._tab = 0        # 0=Controls 1=Simulation 2=EventLog
        self._fonts_init = False
        self._f_sm  = None
        self._f_med = None
        self._f_lg  = None

        # Scroll offset for simulation tab
        self._sim_scroll = 0

        # Built lazily
        self._tab_btns: List[_Button] = []
        self._sim_items: List[dict] = []
        self._log_btns: dict = {}
        self._close_btn: Optional[_Button] = None

    # ── Public ────────────────────────────────────────────────────────────────

    def toggle(self) -> None:
        self.visible = not self.visible

    def open(self) -> None:
        self.visible = True

    def close(self) -> None:
        self.visible = False

    def handle_event(self, event: pygame.event.Event,
                     sw: int, sh: int) -> bool:
        """Returns True if event consumed."""
        if not self.visible:
            return False

        ox, oy = sw // 2 - PANEL_W // 2, sh // 2 - PANEL_H // 2
        mx, my = pygame.mouse.get_pos()

        # Update hover states
        for btn in self._tab_btns:
            btn.update_hover(mx, my)
        if self._close_btn:
            self._close_btn.update_hover(mx, my)
        for btn in self._log_btns.values():
            btn.update_hover(mx, my)
        for item in self._sim_items:
            item["btn_minus"].update_hover(mx, my)
            item["btn_plus"].update_hover(mx, my)
            if "btn_toggle" in item:
                item["btn_toggle"].update_hover(mx, my)

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_o, pygame.K_ESCAPE):
                self.close()
                return True

        if event.type == pygame.MOUSEWHEEL and self._tab == 1:
            self._sim_scroll = max(0, self._sim_scroll - event.y * 20)
            return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Must be inside panel to consume
            panel_rect = pygame.Rect(ox, oy, PANEL_W, PANEL_H)
            if not panel_rect.collidepoint(mx, my):
                self.close()
                return True   # still consume — click outside closes

            # Close button
            if self._close_btn and self._close_btn.clicked(mx, my):
                self.close()
                return True

            # Tab buttons
            for i, btn in enumerate(self._tab_btns):
                if btn.clicked(mx, my):
                    self._tab = i
                    self._sim_scroll = 0
                    return True

            # Event log toggles
            if self._tab == 2:
                for etype, btn in self._log_btns.items():
                    if btn.clicked(mx, my):
                        log = get_log()
                        log.set_suppressed(etype, not log.is_suppressed(etype))
                        return True

            # Simulation +/-  toggles
            if self._tab == 1:
                for item in self._sim_items:
                    if item["btn_minus"].clicked(mx, my):
                        item["dec"]()
                        return True
                    if item["btn_plus"].clicked(mx, my):
                        item["inc"]()
                        return True
                    if "btn_toggle" in item and item["btn_toggle"].clicked(mx, my):
                        item["toggle"]()
                        return True

            return True   # consume all clicks inside panel

        return False

    # ── Draw ─────────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return
        self._init_fonts()
        sw, sh = surface.get_size()
        ox = sw // 2 - PANEL_W // 2
        oy = sh // 2 - PANEL_H // 2

        # Dim background
        dim = pygame.Surface((sw, sh), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 160))
        surface.blit(dim, (0, 0))

        # Panel background
        panel = pygame.Surface((PANEL_W, PANEL_H), pygame.SRCALPHA)
        panel.fill(BG)
        pygame.draw.rect(panel, BORDER, (0, 0, PANEL_W, PANEL_H), 2, border_radius=6)
        surface.blit(panel, (ox, oy))

        # Title bar
        pygame.draw.rect(surface, HEADER_BG, (ox, oy, PANEL_W, TAB_H), border_radius=6)
        title_s = self._f_lg.render("OPTIONS", True, TEXT_ACCENT)
        surface.blit(title_s, (ox + 14, oy + (TAB_H - title_s.get_height()) // 2))

        # Close button
        self._close_btn = _Button(
            pygame.Rect(ox + PANEL_W - 36, oy + 4, 28, 28),
            "✕", color=(60, 30, 30), hover_color=(140, 50, 50)
        )
        self._close_btn.draw(surface, self._f_med)

        # Tab bar
        tab_y = oy + TAB_H
        tab_w = PANEL_W // len(TABS)
        self._tab_btns = []
        for i, label in enumerate(TABS):
            btn = _Button(
                pygame.Rect(ox + i * tab_w, tab_y, tab_w, TAB_H),
                label
            )
            btn.update_hover(*pygame.mouse.get_pos())
            col = TAB_ACTIVE if i == self._tab else TAB_IDLE
            btn.draw(surface, self._f_med, override_color=col)
            self._tab_btns.append(btn)

        # Tab underline
        pygame.draw.line(surface, BORDER,
            (ox, tab_y + TAB_H), (ox + PANEL_W, tab_y + TAB_H), 1)

        content_y = tab_y + TAB_H + 8
        content_h = PANEL_H - TAB_H * 2 - 12

        if self._tab == 0:
            self._draw_controls(surface, ox + 14, content_y, PANEL_W - 28, content_h)
        elif self._tab == 1:
            self._draw_simulation(surface, ox + 14, content_y, PANEL_W - 28, content_h, ox, oy)
        elif self._tab == 2:
            self._draw_event_log(surface, ox + 14, content_y, PANEL_W - 28, content_h)
        elif self._tab == 3:
            self._draw_charts_help(surface, ox + 14, content_y, PANEL_W - 28, content_h)

    # ── Controls tab ─────────────────────────────────────────────────────────

    def _draw_controls(self, surface, ox, oy, w, h) -> None:
        bindings = [
            ("WASD / Arrow Keys",  "Pan camera"),
            ("Scroll Wheel",       "Zoom in / out"),
            ("F",                  "Fit entire world on screen"),
            ("Click creature",     "Open detail panel + follow"),
            ("Click world (empty)","Close detail panel"),
            ("T  /  Escape",       "Close detail panel"),
            ("P  /  Space",        "Pause / Unpause"),
            ("S",                  "Save simulation"),
            ("L",                  "Load last save"),
            ("I",                  "Open saved creature picker (load & spawn)"),
            ("R",                  "Toggle ray visualization"),
            ("G",                  "Toggle signal aura visualization (broadcast signal output)"),
            ("] / [",              "Increase / decrease simulation speed (1x 2x 4x 8x 16x)"),
            ("O",                  "Open / close this options menu"),
            ("Q  /  Escape",       "Quit (when no panel open)"),
        ]

        key_col = ox
        act_col = ox + 220
        row_h   = 24
        y = oy + 4

        # Headers
        surface.blit(self._f_med.render("Key / Action", True, TEXT_ACCENT), (key_col, y))
        surface.blit(self._f_med.render("Description", True, TEXT_ACCENT), (act_col, y))
        y += row_h
        pygame.draw.line(surface, BORDER, (ox, y - 4), (ox + w, y - 4), 1)

        for key, desc in bindings:
            if y > oy + h:
                break
            ks = self._f_sm.render(key, True, TEXT)
            ds = self._f_sm.render(desc, True, TEXT_DIM)
            surface.blit(ks, (key_col, y))
            surface.blit(ds, (act_col, y))
            y += row_h

    # ── Simulation tab ────────────────────────────────────────────────────────

    def _build_sim_items(self, base_x: int, base_y: int) -> None:
        """Build the list of simulation setting widgets."""
        cfg = self.cfg

        def toggle_bool(attr):
            setattr(cfg, attr, not getattr(cfg, attr))

        def adj(attr, delta, lo, hi):
            v = round(getattr(cfg, attr) + delta, 4)
            setattr(cfg, attr, max(lo, min(hi, v)))

        ROW_H = 50
        BTN_W = 28

        items = [
            # label, get_val, inc, dec, is_bool, description
            ("Show Trails",         lambda: cfg.show_trails,
             lambda: toggle_bool("show_trails"), None, True,
             "Draw a movement trail behind each creature showing recent path"),
            ("Show Rays",           lambda: cfg.show_rays,
             lambda: toggle_bool("show_rays"), None, True,
             "Visualise the 6 sensor rays each creature casts to detect surroundings"),
            ("Show Signals",        lambda: cfg.show_signals,
             lambda: toggle_bool("show_signals"), None, True,
             "Draw a gold aura ring around creatures currently broadcasting a signal"),
            ("Population Floor",    lambda: cfg.population_floor_enabled,
             lambda: toggle_bool("population_floor_enabled"), None, True,
             "Automatically inject new random creatures when population drops too low"),
            ("Floor Size",          lambda: cfg.population_floor,
             lambda: adj("population_floor", 5, 5, 100),
             lambda: adj("population_floor", -5, 5, 100), False,
             "Minimum population count before random injection triggers  (step 5)"),
            ("Resource Spawn Rate", lambda: f"{cfg.resource_spawn_rate:.1f}/tick",
             lambda: adj("resource_spawn_rate", 0.5, 0.1, 20),
             lambda: adj("resource_spawn_rate", -0.5, 0.1, 20), False,
             "New food items added to the world each tick  (step 0.5)"),
            ("Max Speed",           lambda: f"{cfg.max_speed:.2f}",
             lambda: adj("max_speed", 0.25, 0.5, 10),
             lambda: adj("max_speed", -0.25, 0.5, 10), False,
             "Fastest a creature can move in world units per tick  (step 0.25)"),
            ("Metabolic Cost",      lambda: f"{cfg.metabolic_cost_per_tick:.3f}",
             lambda: adj("metabolic_cost_per_tick", 0.01, 0.001, 1.0),
             lambda: adj("metabolic_cost_per_tick", -0.01, 0.001, 1.0), False,
             "Base energy drained each tick just for being alive  (step 0.01)"),
            ("NN Cost/Weight",      lambda: f"{cfg.nn_cost_per_weight:.4f}",
             lambda: adj("nn_cost_per_weight", 0.0001, 0, 0.01),
             lambda: adj("nn_cost_per_weight", -0.0001, 0, 0.01), False,
             "Energy cost per active connection per tick — larger brains cost more  (step 0.0001)"),
            ("Reproduce Threshold", lambda: f"{cfg.reproduce_energy_threshold:.0f}",
             lambda: adj("reproduce_energy_threshold", 10, 50, 400),
             lambda: adj("reproduce_energy_threshold", -10, 50, 400), False,
             "Minimum energy both parents must have before mating is attempted  (step 10)"),
            ("Reproduce Cost",      lambda: f"{cfg.reproduce_cost:.0f}",
             lambda: adj("reproduce_cost", 5, 10, 200),
             lambda: adj("reproduce_cost", -5, 10, 200), False,
             "Energy each parent spends when producing offspring  (step 5)"),
            ("Weight Perturb Rate", lambda: f"{cfg.weight_perturb_rate:.2f}",
             lambda: adj("weight_perturb_rate", 0.05, 0, 1),
             lambda: adj("weight_perturb_rate", -0.05, 0, 1), False,
             "Probability each connection weight is nudged during mutation  (step 0.05)"),
            ("Add Node Rate",       lambda: f"{cfg.add_node_rate:.3f}",
             lambda: adj("add_node_rate", 0.005, 0, 0.2),
             lambda: adj("add_node_rate", -0.005, 0, 0.2), False,
             "Chance per birth that a new hidden neuron is inserted into the brain  (step 0.005)"),
            ("Add Connection Rate", lambda: f"{cfg.add_connection_rate:.3f}",
             lambda: adj("add_connection_rate", 0.005, 0, 0.3),
             lambda: adj("add_connection_rate", -0.005, 0, 0.3), False,
             "Chance per birth that a new synaptic connection is added to the brain  (step 0.005)"),
            ("Mate Search Radius",  lambda: f"{cfg.mate_search_radius:.0f}",
             lambda: adj("mate_search_radius", 10, 20, 400),
             lambda: adj("mate_search_radius", -10, 20, 400), False,
             "How close two creatures must be in world units to attempt mating  (step 10)"),
        ]

        self._sim_items = []
        for i, (label, get_val, inc_fn, dec_fn, is_bool, desc) in enumerate(items):
            y = base_y + i * ROW_H
            val_x = base_x + 250

            if is_bool:
                toggle_btn = _Button(
                    pygame.Rect(val_x, y + 2, 70, 24), "", BTN_BG, BTN_HOV
                )
                self._sim_items.append({
                    "label": label, "desc": desc, "get_val": get_val, "is_bool": True,
                    "btn_minus": _Button(pygame.Rect(0, 0, 0, 0), ""),  # dummy
                    "btn_plus":  _Button(pygame.Rect(0, 0, 0, 0), ""),  # dummy
                    "btn_toggle": toggle_btn,
                    "toggle": inc_fn,
                    "y": y,
                })
            else:
                minus_btn = _Button(pygame.Rect(val_x, y + 2, BTN_W, 24), "−")
                plus_btn  = _Button(pygame.Rect(val_x + BTN_W + 50, y + 2, BTN_W, 24), "+")
                self._sim_items.append({
                    "label": label, "desc": desc, "get_val": get_val, "is_bool": False,
                    "btn_minus": minus_btn, "btn_plus": plus_btn,
                    "inc": inc_fn, "dec": dec_fn,
                    "y": y,
                })

    def _wrap_text(self, text: str, font: pygame.font.Font, max_w: int) -> List[str]:
        words = text.split(" ")
        lines, current = [], ""
        for word in words:
            test = (current + " " + word).strip()
            if font.size(test)[0] <= max_w:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [text]

    def _draw_simulation(self, surface, ox, oy, w, h, panel_ox, panel_oy) -> None:
        ROW_H  = 50
        BTN_W  = 28
        CTRL_W = 115   # width reserved for the control block (toggle or +/- value +)
        val_x  = ox + 155
        desc_x = val_x + CTRL_W + 10
        desc_w = (panel_ox + PANEL_W - 14) - desc_x

        self._build_sim_items(ox, oy - self._sim_scroll)

        clip_rect = pygame.Rect(panel_ox, panel_oy + TAB_H * 2 + 8, PANEL_W, h)
        old_clip = surface.get_clip()
        surface.set_clip(clip_rect)

        for item in self._sim_items:
            y = item["y"]
            if y + ROW_H < clip_rect.top or y > clip_rect.bottom:
                continue

            mx, my = pygame.mouse.get_pos()
            if (self._sim_items.index(item) % 2) == 0:
                pygame.draw.rect(surface, (22, 27, 40),
                    pygame.Rect(ox - 4, y, w + 8, ROW_H - 2))

            # Label (left column)
            label_s = self._f_med.render(item["label"], True, TEXT)
            surface.blit(label_s, (ox, y + (ROW_H - label_s.get_height()) // 2))

            # Control (middle column), vertically centred
            ctrl_y = y + (ROW_H - 26) // 2
            if item["is_bool"]:
                val = item["get_val"]()
                col = TOGGLE_ON if val else TOGGLE_OFF
                btn = item["btn_toggle"]
                btn.rect = pygame.Rect(val_x, ctrl_y, 70, 26)
                btn.update_hover(mx, my)
                btn.draw(surface, self._f_sm, override_color=col)
                state_s = self._f_sm.render("ON" if val else "OFF", True, TEXT)
                surface.blit(state_s, (btn.rect.centerx - state_s.get_width() // 2,
                                       btn.rect.centery - state_s.get_height() // 2))
            else:
                val_str = str(item["get_val"]())
                item["btn_minus"].rect = pygame.Rect(val_x,               ctrl_y, BTN_W, 26)
                item["btn_plus"].rect  = pygame.Rect(val_x + BTN_W + 50, ctrl_y, BTN_W, 26)
                item["btn_minus"].update_hover(mx, my)
                item["btn_plus"].update_hover(mx, my)
                item["btn_minus"].draw(surface, self._f_med)
                item["btn_plus"].draw(surface, self._f_med)
                vs = self._f_sm.render(val_str, True, TEXT_ACCENT)
                surface.blit(vs, (val_x + BTN_W + 4, ctrl_y + (26 - vs.get_height()) // 2))

            # Description (right column, word-wrapped)
            lines = self._wrap_text(item["desc"], self._f_sm, desc_w)
            line_h = self._f_sm.get_height() + 1
            total_text_h = len(lines) * line_h
            text_y = y + (ROW_H - total_text_h) // 2
            for line in lines:
                ds = self._f_sm.render(line, True, TEXT_DIM)
                surface.blit(ds, (desc_x, text_y))
                text_y += line_h

        surface.set_clip(old_clip)

        total_h = len(self._sim_items) * ROW_H
        if total_h > h:
            hint = self._f_sm.render("↑↓  scroll", True, TEXT_DIM)
            surface.blit(hint, (ox + w - hint.get_width(), panel_oy + PANEL_H - 22))

    # ── Event log tab ─────────────────────────────────────────────────────────

    def _draw_event_log(self, surface, ox, oy, w, h) -> None:
        log = get_log()
        self._log_btns = {}

        surface.blit(
            self._f_med.render("Toggle which events appear in the Event Log:", True, TEXT_DIM),
            (ox, oy)
        )
        oy += 28

        COL_W   = 200
        ROW_H   = 42
        COLS    = 3
        BTN_W_L = 160
        BTN_H   = 28

        for i, etype in enumerate(ALL_TYPES):
            col = i % COLS
            row = i // COLS
            ex = ox + col * COL_W
            ey = oy + row * ROW_H

            suppressed = log.is_suppressed(etype)
            type_color = _TYPE_COLORS.get(etype, (160, 160, 160))

            btn = _Button(
                pygame.Rect(ex, ey, BTN_W_L, BTN_H),
                etype,
                color=(30, 35, 48) if suppressed else (25, 45, 35),
                hover_color=BTN_HOV,
            )
            btn.update_hover(*pygame.mouse.get_pos())
            btn.draw(surface, self._f_med,
                     override_color=(30, 35, 48) if suppressed else None)

            # Color swatch
            swatch_x = ex + BTN_W_L + 6
            pygame.draw.rect(surface, type_color if not suppressed else (50, 50, 60),
                             pygame.Rect(swatch_x, ey + 6, 14, 14), border_radius=3)

            # ON/OFF label
            state_col = (80, 90, 110) if suppressed else type_color
            state_s = self._f_sm.render(
                "HIDDEN" if suppressed else "SHOWN", True, state_col
            )
            surface.blit(state_s, (ex, ey + BTN_H + 2))

            self._log_btns[etype] = btn

        # Description footer
        fy = oy + (len(ALL_TYPES) // COLS + 1) * ROW_H + 16
        surface.blit(
            self._f_sm.render("Click any button to toggle that event type on or off.", True, TEXT_DIM),
            (ox, fy)
        )

    # ── Charts help tab ───────────────────────────────────────────────────────

    def _draw_charts_help(self, surface, ox, oy, w, h) -> None:
        charts = [
            (
                "ENERGY DISTRIBUTION  (bottom-left HUD)",
                [
                    "Shows how creature energy is spread across the population right now.",
                    "",
                    "  X-axis:  Energy level from 0 (left) to max energy (right).",
                    "  Y-axis:  Number of creatures at each energy level (relative).",
                    "  Color:   Red = low energy (starving), Green = high energy (well-fed).",
                    "  Key:     The gradient strip above the chart shows the color scale.",
                    "",
                    "How to read it:",
                    "  Bars bunched LEFT  →  population is starving; food may be scarce.",
                    "  Bars bunched RIGHT →  creatures are thriving; resources are plentiful.",
                    "  Tall bar at far left →  many creatures near death this tick.",
                    "  Bimodal split      →  two groups: one thriving, one struggling.",
                    "  Even spread        →  healthy mix of energy levels.",
                    "",
                    "The chart updates every frame in real time.",
                ],
            ),
        ]

        y = oy + 4
        row_h = 16

        for title, lines in charts:
            # Section title
            ts = self._f_med.render(title, True, TEXT_ACCENT)
            surface.blit(ts, (ox, y))
            y += row_h + 4

            pygame.draw.line(surface, BORDER, (ox, y - 2), (ox + w, y - 2), 1)

            for line in lines:
                if y > oy + h - row_h:
                    break
                col = TEXT_DIM if line.startswith(" ") or line == "" else TEXT
                ls = self._f_sm.render(line, True, col)
                surface.blit(ls, (ox + 4, y))
                y += row_h

            y += 12  # gap between sections

    # ── Fonts ─────────────────────────────────────────────────────────────────

    def _init_fonts(self) -> None:
        if not self._fonts_init:
            self._f_sm  = pygame.font.SysFont("monospace", 12)
            self._f_med = pygame.font.SysFont("monospace", 13, bold=True)
            self._f_lg  = pygame.font.SysFont("monospace", 16, bold=True)
            self._fonts_init = True
