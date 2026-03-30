"""
CreaturePickerModal — browse and spawn saved creatures.

Opens as a centered modal.  Each row shows the creature's name, generation,
age when saved, tick saved, and species id.  Clicking "Load" spawns it.
"""

from __future__ import annotations

from typing import List, Optional

import pygame

BG          = (13, 17, 28, 245)
BORDER      = (50, 65, 90)
HEADER_BG   = (20, 28, 45)
ROW_BG      = (18, 23, 36)
ROW_ALT     = (22, 28, 42)
ROW_HOVER   = (30, 38, 58)
TEXT_PRI    = (220, 228, 240)
TEXT_DIM    = (110, 120, 140)
TEXT_ACCENT = (100, 180, 255)
BTN_LOAD    = ( 50, 160,  90)
BTN_LOAD_HO = ( 70, 200, 110)
BTN_CLOSE   = ( 80,  50,  50)
BTN_CLOSE_HO= (120,  70,  70)

MODAL_W = 640
MODAL_H = 480
ROW_H   = 44
HEADER_H = 38


class CreaturePickerModal:
    def __init__(self) -> None:
        self.visible: bool = False
        self.load_requested: Optional[str] = None   # filename to spawn

        self._entries: List[dict] = []
        self._scroll: int = 0
        self._hovered_row: int = -1
        self._hovered_close: bool = False

        self._f_small = None
        self._f_med   = None
        self._f_large = None

        # Cached rects (set during draw, used in handle_event)
        self._close_rect: Optional[pygame.Rect] = None
        self._load_rects: List[tuple] = []   # list of (rect, filename)
        self._list_rect:  Optional[pygame.Rect] = None
        self._ox: int = 0
        self._oy: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def open(self, entries: List[dict]) -> None:
        self.visible = True
        self.load_requested = None
        self._entries = entries
        self._scroll = 0

    def close(self) -> None:
        self.visible = False

    def toggle(self, entries: List[dict]) -> None:
        if self.visible:
            self.close()
        else:
            self.open(entries)

    # ── Events ────────────────────────────────────────────────────────────────

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.close()
            return True

        if event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            self._hovered_close = bool(self._close_rect and self._close_rect.collidepoint(mx, my))
            if self._list_rect and self._list_rect.collidepoint(mx, my):
                rel_y = my - self._list_rect.top + self._scroll
                self._hovered_row = rel_y // ROW_H
            else:
                self._hovered_row = -1
            return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            # Close button
            if self._close_rect and self._close_rect.collidepoint(mx, my):
                self.close()
                return True
            # Load buttons
            for rect, filename in self._load_rects:
                if rect.collidepoint(mx, my):
                    self.load_requested = filename
                    self.close()
                    return True
            # Click inside modal but not a button — consume
            ox, oy = self._ox, self._oy
            if pygame.Rect(ox, oy, MODAL_W, MODAL_H).collidepoint(mx, my):
                return True
            # Click outside modal — close
            self.close()
            return True

        if event.type == pygame.MOUSEWHEEL:
            if self._list_rect:
                mx, my = pygame.mouse.get_pos()
                if self._list_rect.collidepoint(mx, my):
                    self._scroll = max(0, self._scroll - event.y * ROW_H)
                    return True

        return False

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return
        self._init_fonts()

        sw, sh = surface.get_size()
        ox = (sw - MODAL_W) // 2
        oy = (sh - MODAL_H) // 2
        self._ox, self._oy = ox, oy

        # Background overlay
        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        surface.blit(overlay, (0, 0))

        # Modal background
        modal_surf = pygame.Surface((MODAL_W, MODAL_H), pygame.SRCALPHA)
        modal_surf.fill(BG)
        surface.blit(modal_surf, (ox, oy))
        pygame.draw.rect(surface, BORDER, (ox, oy, MODAL_W, MODAL_H), 2, border_radius=4)

        # Header
        pygame.draw.rect(surface, HEADER_BG, (ox, oy, MODAL_W, HEADER_H))
        title = self._f_large.render("SAVED CREATURES", True, TEXT_ACCENT)
        surface.blit(title, (ox + 12, oy + (HEADER_H - title.get_height()) // 2))

        # Close button
        close_rect = pygame.Rect(ox + MODAL_W - 36, oy + 6, 28, 26)
        self._close_rect = close_rect
        close_col = BTN_CLOSE_HO if self._hovered_close else BTN_CLOSE
        pygame.draw.rect(surface, close_col, close_rect, border_radius=4)
        xs = self._f_med.render("✕", True, TEXT_PRI)
        surface.blit(xs, (close_rect.centerx - xs.get_width() // 2,
                          close_rect.centery - xs.get_height() // 2))

        # Column headers
        hdr_y = oy + HEADER_H + 4
        self._text(surface, "NAME",       ox + 10,  hdr_y, self._f_med, TEXT_DIM)
        self._text(surface, "GEN",        ox + 190, hdr_y, self._f_med, TEXT_DIM)
        self._text(surface, "AGE",        ox + 250, hdr_y, self._f_med, TEXT_DIM)
        self._text(surface, "TICK SAVED", ox + 330, hdr_y, self._f_med, TEXT_DIM)
        self._text(surface, "SPECIES",    ox + 450, hdr_y, self._f_med, TEXT_DIM)
        hdr_line_y = hdr_y + 18
        pygame.draw.line(surface, BORDER, (ox + 8, hdr_line_y), (ox + MODAL_W - 8, hdr_line_y))

        # List area
        list_top = hdr_line_y + 4
        list_h   = MODAL_H - (list_top - oy) - 8
        list_rect = pygame.Rect(ox, list_top, MODAL_W, list_h)
        self._list_rect = list_rect

        # Clip to list area
        clip = surface.subsurface(list_rect)

        self._load_rects = []
        max_scroll = max(0, len(self._entries) * ROW_H - list_h)
        self._scroll = min(self._scroll, max_scroll)

        if not self._entries:
            no_saves = self._f_med.render("No saved creatures found.", True, TEXT_DIM)
            clip.blit(no_saves, (MODAL_W // 2 - no_saves.get_width() // 2,
                                 list_h // 2 - no_saves.get_height() // 2))
        else:
            for i, entry in enumerate(self._entries):
                row_y = i * ROW_H - self._scroll
                if row_y + ROW_H < 0 or row_y > list_h:
                    continue

                bg_col = ROW_HOVER if i == self._hovered_row else (ROW_ALT if i % 2 else ROW_BG)
                pygame.draw.rect(clip, bg_col, (0, row_y, MODAL_W, ROW_H - 2), border_radius=2)

                name_str = entry.get("name_str", "?")
                gen      = entry.get("generation", 0)
                age      = entry.get("age", 0)
                tick     = entry.get("tick", 0)
                spc      = entry.get("species_id", 0)
                fname    = entry.get("filename", "")

                ty = row_y + (ROW_H - 14) // 2
                self._text(clip, name_str,     10,  ty, self._f_med, TEXT_PRI)
                self._text(clip, str(gen),     190, ty, self._f_small, TEXT_DIM)
                self._text(clip, f"{age:,}",   250, ty, self._f_small, TEXT_DIM)
                self._text(clip, f"{tick:,}",  330, ty, self._f_small, TEXT_DIM)
                self._text(clip, f"#{spc}",    450, ty, self._f_small, TEXT_DIM)

                # Load button
                btn_rect_local = pygame.Rect(MODAL_W - 80, row_y + 8, 68, ROW_H - 18)
                btn_rect_screen = pygame.Rect(list_rect.left + btn_rect_local.x,
                                              list_rect.top  + btn_rect_local.y,
                                              btn_rect_local.w, btn_rect_local.h)
                self._load_rects.append((btn_rect_screen, fname))

                hov = btn_rect_screen.collidepoint(pygame.mouse.get_pos())
                btn_col = BTN_LOAD_HO if hov else BTN_LOAD
                pygame.draw.rect(clip, btn_col, btn_rect_local, border_radius=4)
                ls = self._f_med.render("Load", True, (240, 255, 240))
                clip.blit(ls, (btn_rect_local.centerx - ls.get_width() // 2,
                               btn_rect_local.centery - ls.get_height() // 2))

        # Count hint
        count_s = self._f_small.render(
            f"{len(self._entries)} creature(s) saved  ·  I to open/close",
            True, TEXT_DIM
        )
        surface.blit(count_s, (ox + MODAL_W // 2 - count_s.get_width() // 2,
                                oy + MODAL_H - 20))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _init_fonts(self) -> None:
        if self._f_small is None:
            self._f_small = pygame.font.SysFont("monospace", 12)
            self._f_med   = pygame.font.SysFont("monospace", 13, bold=True)
            self._f_large = pygame.font.SysFont("monospace", 16, bold=True)

    @staticmethod
    def _text(surface, text, x, y, font, color):
        s = font.render(str(text), True, color)
        surface.blit(s, (x, y))
