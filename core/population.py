"""
Population — manages the living creature list and spatial index bookkeeping.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from config import SimConfig, CONFIG
from core.creature import Creature
from core.world import World

if TYPE_CHECKING:
    pass


class Population:
    def __init__(self, world: World, cfg: SimConfig = CONFIG) -> None:
        self.world = world
        self.cfg = cfg
        self._creatures: Dict[int, Creature] = {}   # id -> Creature

    # ── Add / remove ──────────────────────────────────────────────────────────

    def add(self, creature: Creature) -> None:
        self._creatures[creature.id] = creature
        self.world.add_creature_to_index(
            creature.id, float(creature.pos[0]), float(creature.pos[1])
        )

    def remove(self, creature_id: int) -> Optional[Creature]:
        creature = self._creatures.pop(creature_id, None)
        if creature:
            self.world.remove_creature_from_index(
                creature_id, float(creature.pos[0]), float(creature.pos[1])
            )
        return creature

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_by_id(self, creature_id: int) -> Optional[Creature]:
        return self._creatures.get(creature_id)

    @property
    def creatures(self) -> List[Creature]:
        return list(self._creatures.values())

    @property
    def count(self) -> int:
        return len(self._creatures)

    # ── Spatial sync ──────────────────────────────────────────────────────────

    def sync_position(self, creature: Creature, old_x: float, old_y: float) -> None:
        """Update spatial hash after creature moves."""
        self.world.update_creature_position(
            creature.id, old_x, old_y,
            float(creature.pos[0]), float(creature.pos[1])
        )

    # ── Reaping ───────────────────────────────────────────────────────────────

    def reap_dead(self) -> List[Creature]:
        """Remove and return all dead creatures."""
        dead = [c for c in self._creatures.values() if not c.is_alive()]
        for c in dead:
            self.remove(c.id)
        return dead

    # ── Random spawn ──────────────────────────────────────────────────────────

    def random_spawn_position(self) -> np.ndarray:
        return np.array([
            random.uniform(50, self.cfg.world_width - 50),
            random.uniform(50, self.cfg.world_height - 50),
        ], dtype=np.float32)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"creatures": [c.to_dict() for c in self._creatures.values()]}

    def load_from_dict(self, d: dict) -> None:
        from core.creature import Creature
        self._creatures.clear()
        for cd in d.get("creatures", []):
            c = Creature.from_dict(cd)
            self.add(c)
