"""
EventLog — collects simulation events and renders them as a right-side panel.

Any module can push events via the module-level `push()` function.
The renderer calls `draw()` each frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import pygame

# ── Event types ───────────────────────────────────────────────────────────────

BIRTH       = "BIRTH"
DEATH       = "DEATH"
FORK        = "FORK"        # new species created
EXTINCTION  = "EXTINCT"     # species wiped out
ADD_NODE    = "ADD NODE"
ADD_CONN    = "ADD CONN"
INJECT      = "INJECT"      # population floor injection
MILESTONE   = "MILESTONE"

# Colors per type
_TYPE_COLORS = {
    BIRTH:      (100, 220, 130),
    DEATH:      (180,  70,  70),
    FORK:       ( 80, 180, 255),
    EXTINCTION: (255, 140,  50),
    ADD_NODE:   (200, 120, 255),
    ADD_CONN:   (150, 100, 220),
    INJECT:     (255, 220,  60),
    MILESTONE:  (255, 200,  80),
}

_TYPE_PREFIX = {
    BIRTH:      "+ ",
    DEATH:      "x ",
    FORK:       "~ ",
    EXTINCTION: "! ",
    ADD_NODE:   "o ",
    ADD_CONN:   "- ",
    INJECT:     "* ",
    MILESTONE:  "★ ",
}

PANEL_WIDTH  = 270
MAX_ENTRIES  = 60    # lines kept in memory
VISIBLE_ROWS = 28    # lines shown at once


@dataclass
class LogEntry:
    tick: int
    event_type: str
    message: str


ALL_TYPES = [BIRTH, DEATH, FORK, EXTINCTION, ADD_NODE, ADD_CONN, INJECT, MILESTONE]


class EventLog:
    def __init__(self) -> None:
        self._entries: Deque[LogEntry] = deque(maxlen=MAX_ENTRIES)
        self._font = None
        self._font_bold = None
        self._scroll: int = 0   # 0 = show newest at bottom
        self._suppressed: set = set()   # event types that are filtered out

    def push(self, event_type: str, message: str, tick: int) -> None:
        if event_type in self._suppressed:
            return
        self._entries.append(LogEntry(tick, event_type, message))
        self._scroll = 0   # snap to newest on new event

    def is_suppressed(self, event_type: str) -> bool:
        return event_type in self._suppressed

    def set_suppressed(self, event_type: str, suppressed: bool) -> None:
        if suppressed:
            self._suppressed.add(event_type)
        else:
            self._suppressed.discard(event_type)

    def _init_fonts(self) -> None:
        if self._font is None:
            self._font      = pygame.font.SysFont("monospace", 12)
            self._font_bold = pygame.font.SysFont("monospace", 12, bold=True)

    def draw(self, surface: pygame.Surface) -> None:
        self._init_fonts()
        sw, sh = surface.get_size()

        panel_x = sw - PANEL_WIDTH
        panel_h = sh

        # Semi-transparent background
        panel_surf = pygame.Surface((PANEL_WIDTH, panel_h), pygame.SRCALPHA)
        panel_surf.fill((12, 14, 20, 210))
        pygame.draw.line(panel_surf, (50, 60, 80), (0, 0), (0, panel_h), 2)

        # Title bar
        pygame.draw.rect(panel_surf, (20, 25, 35), (0, 0, PANEL_WIDTH, 22))
        title = self._font_bold.render("EVENT LOG", True, (120, 140, 180))
        panel_surf.blit(title, (8, 4))

        entries = list(self._entries)
        total = len(entries)
        # Show newest at bottom — take last VISIBLE_ROWS
        start = max(0, total - VISIBLE_ROWS - self._scroll)
        visible = entries[start : start + VISIBLE_ROWS]

        row_h = 17
        y = 28
        for entry in visible:
            color = _TYPE_COLORS.get(entry.event_type, (180, 180, 180))
            prefix = _TYPE_PREFIX.get(entry.event_type, "  ")

            # Tick number (dim)
            tick_str = f"{entry.tick:>6} "
            tick_surf = self._font.render(tick_str, True, (60, 70, 90))
            panel_surf.blit(tick_surf, (4, y))

            # Type tag
            tag = f"{prefix}{entry.event_type:<7} "
            tag_surf = self._font_bold.render(tag, True, color)
            panel_surf.blit(tag_surf, (4 + tick_surf.get_width(), y))

            # Message — wrap if too long
            max_msg_w = PANEL_WIDTH - 8
            msg_x = 4
            msg_y = y + row_h
            msg_surf = self._font.render(entry.message, True, (170, 180, 195))
            # Truncate to fit
            msg_text = entry.message
            while msg_surf.get_width() > PANEL_WIDTH - 10 and len(msg_text) > 4:
                msg_text = msg_text[:-4] + "..."
                msg_surf = self._font.render(msg_text, True, (170, 180, 195))
            panel_surf.blit(msg_surf, (8, msg_y))

            y += row_h * 2

        # Scroll hint
        if self._scroll > 0:
            hint = self._font.render(f"^ {self._scroll} newer", True, (80, 90, 110))
            panel_surf.blit(hint, (8, panel_h - 18))

        surface.blit(panel_surf, (panel_x, 0))

    def handle_scroll(self, direction: int) -> None:
        """direction: +1 scroll up (older), -1 scroll down (newer)."""
        total = len(self._entries)
        self._scroll = max(0, min(total - VISIBLE_ROWS, self._scroll + direction * 3))


# ── Module-level singleton ─────────────────────────────────────────────────────

_log = EventLog()


def push(event_type: str, message: str, tick: int) -> None:
    _log.push(event_type, message, tick)


def get_log() -> EventLog:
    return _log
