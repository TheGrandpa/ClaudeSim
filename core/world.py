"""
World — the simulation environment.

Manages:
  - Resource spawning, placement, and consumption
  - SpatialHash for fast proximity queries on both resources and creatures
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import SimConfig, CONFIG


# ── Spatial hash ──────────────────────────────────────────────────────────────

class SpatialHash:
    """
    Uniform grid spatial hash.  Stores arbitrary integer ids (creature ids or
    resource indices) in cells.  Cell size should be >= the largest query radius
    so a single-ring cell search always covers the full radius.
    """

    def __init__(self, cell_size: float) -> None:
        self.cell_size = cell_size
        self._cells: Dict[Tuple[int, int], Set[int]] = {}

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x // self.cell_size), int(y // self.cell_size))

    def insert(self, entity_id: int, x: float, y: float) -> None:
        c = self._cell(x, y)
        self._cells.setdefault(c, set()).add(entity_id)

    def remove(self, entity_id: int, x: float, y: float) -> None:
        c = self._cell(x, y)
        if c in self._cells:
            self._cells[c].discard(entity_id)

    def move(self, entity_id: int, old_x: float, old_y: float,
             new_x: float, new_y: float) -> None:
        old_c = self._cell(old_x, old_y)
        new_c = self._cell(new_x, new_y)
        if old_c != new_c:
            if old_c in self._cells:
                self._cells[old_c].discard(entity_id)
            self._cells.setdefault(new_c, set()).add(entity_id)

    def query_radius(self, x: float, y: float, radius: float) -> Set[int]:
        """Return all ids within radius of (x, y).  May include false positives."""
        cx = int(x // self.cell_size)
        cy = int(y // self.cell_size)
        cells_needed = int(radius // self.cell_size) + 1
        result: Set[int] = set()
        for dx in range(-cells_needed, cells_needed + 1):
            for dy in range(-cells_needed, cells_needed + 1):
                cell = (cx + dx, cy + dy)
                if cell in self._cells:
                    result |= self._cells[cell]
        return result

    def clear(self) -> None:
        self._cells.clear()


# ── Resource ──────────────────────────────────────────────────────────────────

class Resource:
    __slots__ = ("idx", "x", "y", "energy", "active")

    def __init__(self, idx: int, x: float, y: float, energy: float) -> None:
        self.idx = idx
        self.x = x
        self.y = y
        self.energy = energy
        self.active = True


# ── World ─────────────────────────────────────────────────────────────────────

class World:
    def __init__(self, cfg: SimConfig = CONFIG) -> None:
        self.cfg = cfg
        self.width = cfg.world_width
        self.height = cfg.world_height

        # Pre-allocate resource pool
        self._resources: List[Resource] = [
            Resource(i, 0.0, 0.0, 0.0) for i in range(cfg.max_resources)
        ]
        self._active_resource_ids: Set[int] = set()
        self._free_resource_ids: List[int] = list(range(cfg.max_resources))

        # Spatial indices
        self.resource_hash = SpatialHash(cfg.spatial_cell_size)
        self.creature_hash = SpatialHash(cfg.spatial_cell_size)

        # Fractional spawn accumulator
        self._spawn_accumulator: float = 0.0

        # Seed initial resources
        self.spawn_resources(cfg.initial_resources)

    # ── Resource management ───────────────────────────────────────────────────

    def spawn_resources(self, count: int) -> None:
        for _ in range(count):
            self._spawn_one_resource()

    def _spawn_one_resource(self) -> Optional[Resource]:
        if not self._free_resource_ids:
            return None
        idx = self._free_resource_ids.pop()
        r = self._resources[idx]
        r.x = random.uniform(0, self.width)
        r.y = random.uniform(0, self.height)
        r.energy = random.uniform(self.cfg.resource_min_energy, self.cfg.resource_max_energy)
        r.active = True
        self._active_resource_ids.add(idx)
        self.resource_hash.insert(idx, r.x, r.y)
        return r

    def tick_resource_spawn(self) -> None:
        """Called once per tick to maintain resource spawn rate."""
        self._spawn_accumulator += self.cfg.resource_spawn_rate
        while self._spawn_accumulator >= 1.0:
            self._spawn_one_resource()
            self._spawn_accumulator -= 1.0

    def consume_resource(self, resource_idx: int) -> float:
        """Remove resource and return its energy value."""
        r = self._resources[resource_idx]
        if not r.active:
            return 0.0
        energy = r.energy
        r.active = False
        self._active_resource_ids.discard(resource_idx)
        self._free_resource_ids.append(resource_idx)
        self.resource_hash.remove(resource_idx, r.x, r.y)
        return energy

    def get_nearby_resources(self, x: float, y: float, radius: float) -> List[Resource]:
        """Return active resources within radius, sorted by wrapped distance."""
        candidates: Set[int] = set()
        for qx, qy in self._wrap_query_origins(x, y, radius):
            candidates |= self.resource_hash.query_radius(qx, qy, radius)
        result = []
        r2 = radius * radius
        for idx in candidates:
            res = self._resources[idx]
            if not res.active:
                continue
            dx, dy = self._wrap_delta(res.x - x, res.y - y)
            if dx * dx + dy * dy <= r2:
                result.append(res)
        result.sort(key=lambda r: sum(d*d for d in self._wrap_delta(r.x - x, r.y - y)))
        return result

    @property
    def active_resources(self) -> List[Resource]:
        return [self._resources[i] for i in self._active_resource_ids]

    @property
    def resource_count(self) -> int:
        return len(self._active_resource_ids)

    # ── Creature spatial index ────────────────────────────────────────────────

    def add_creature_to_index(self, creature_id: int, x: float, y: float) -> None:
        self.creature_hash.insert(creature_id, x, y)

    def remove_creature_from_index(self, creature_id: int, x: float, y: float) -> None:
        self.creature_hash.remove(creature_id, x, y)

    def update_creature_position(
        self, creature_id: int, old_x: float, old_y: float, new_x: float, new_y: float
    ) -> None:
        self.creature_hash.move(creature_id, old_x, old_y, new_x, new_y)

    def get_nearby_creature_ids(self, x: float, y: float, radius: float) -> Set[int]:
        result: Set[int] = set()
        for qx, qy in self._wrap_query_origins(x, y, radius):
            result |= self.creature_hash.query_radius(qx, qy, radius)
        return result

    # ── Boundary (toroidal) ───────────────────────────────────────────────────

    def wrap_position(self, pos: np.ndarray) -> np.ndarray:
        """Wrap a position to stay within world bounds (toroidal)."""
        return np.array([pos[0] % self.width, pos[1] % self.height], dtype=np.float32)

    # Keep old name as alias so save/load code doesn't break
    def clamp_position(self, pos: np.ndarray) -> np.ndarray:
        return self.wrap_position(pos)

    def _wrap_delta(self, dx: float, dy: float) -> Tuple[float, float]:
        """Shortest-path delta across a toroidal world."""
        hw = self.width  * 0.5
        hh = self.height * 0.5
        if dx >  hw: dx -= self.width
        elif dx < -hw: dx += self.width
        if dy >  hh: dy -= self.height
        elif dy < -hh: dy += self.height
        return dx, dy

    def _wrap_query_origins(self, x: float, y: float, radius: float) -> List[Tuple[float, float]]:
        """
        Return a list of query origins to cover wrap-around edges.
        Always includes (x, y); adds mirrored origins when near a boundary.
        """
        origins = [(x, y)]
        if x - radius < 0:
            origins.append((x + self.width, y))
        if x + radius > self.width:
            origins.append((x - self.width, y))
        if y - radius < 0:
            origins.append((x, y + self.height))
        if y + radius > self.height:
            origins.append((x, y - self.height))
        # Corners
        if x - radius < 0 and y - radius < 0:
            origins.append((x + self.width, y + self.height))
        if x - radius < 0 and y + radius > self.height:
            origins.append((x + self.width, y - self.height))
        if x + radius > self.width and y - radius < 0:
            origins.append((x - self.width, y + self.height))
        if x + radius > self.width and y + radius > self.height:
            origins.append((x - self.width, y - self.height))
        return origins

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "resources": [
                {"x": r.x, "y": r.y, "energy": r.energy}
                for r in self._resources if r.active
            ]
        }

    def load_resources_from_dict(self, d: dict) -> None:
        """Restore resource state from a saved dict."""
        self.resource_hash.clear()
        self._active_resource_ids.clear()
        self._free_resource_ids = list(range(self.cfg.max_resources))
        for entry in d.get("resources", []):
            r = self._spawn_one_resource()
            if r:
                # Override position and energy
                self.resource_hash.remove(r.idx, r.x, r.y)
                r.x, r.y, r.energy = entry["x"], entry["y"], entry["energy"]
                self.resource_hash.insert(r.idx, r.x, r.y)
