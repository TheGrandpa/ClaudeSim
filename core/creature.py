"""
Creature — runtime state for a single living entity in the simulation.

The Creature holds all mutable per-tick state: position, velocity, energy,
age.  The genome and brain are owned here.  The LineageRecord (birth/death
metadata) lives in core/lineage.py and references the creature by id.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import numpy as np

from core.genome import Genome
from core.naming import CreatureName
from nn.network import NeuralNetwork


# Global id counter — simple monotonic integer
_next_id: int = 0

def _new_id() -> int:
    global _next_id
    _next_id += 1
    return _next_id


# ── Action indices (matches NN output order) ──────────────────────────────────
ACT_THRUST      = 0   # forward thrust [-1, 1]
ACT_TURN        = 1   # turn rate      [-1, 1]
ACT_EAT         = 2   # eat trigger    > eat_threshold → attempt
ACT_REPRODUCE   = 3   # reproduce      > reproduce_threshold → attempt if energy allows
ACT_ATTACK      = 4   # attack         > attack_threshold → attempt nearest creature
ACT_FLEE        = 5   # flee           > flee_threshold → turn 180 + max thrust


class Creature:
    """
    Runtime state for a single creature.

    Attributes
    ----------
    id              unique monotonic integer
    name            heritable syllable name
    pos             (2,) float32 position in world space
    vel             (2,) float32 velocity
    angle           heading in radians
    energy          current energy (dies at ≤ 0)
    age             ticks alive
    genome          NEAT genome
    brain           NeuralNetwork wrapping the genome
    last_action     most recent NN output vector (for HUD/debug)
    color           RGB tuple derived from genome lineage_id
    parent_ids      (id_a, id_b) or None for primordial creatures
    species_id      assigned by speciation each generation
    """

    __slots__ = (
        "id", "name", "pos", "vel", "angle",
        "energy", "age", "years", "genome", "brain",
        "last_action", "signal", "color", "parent_ids", "species_id",
        "_trail", "_food_eaten", "_species_interactions",
    )

    def __init__(
        self,
        genome: Genome,
        name: CreatureName,
        pos: np.ndarray,
        energy: float,
        parent_ids: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.id: int = _new_id()
        self.name = name
        self.pos = pos.astype(np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.angle: float = random.uniform(0, 2 * math.pi)
        self.energy: float = energy
        self.age: int = 0
        self.years: int = 0      # increments every 5000 ticks
        self.genome = genome
        self.brain = NeuralNetwork(genome)
        self.last_action = np.zeros(7, dtype=np.float32)
        self.signal: float = 0.0   # current broadcast signal [0, 1], output node 6
        self.color: Tuple[int, int, int] = genome.appearance.primary_rgb()
        self.parent_ids = parent_ids
        self.species_id: int = 0
        self._trail: list = []   # list of (x, y) for visualization
        self._food_eaten: int = 0
        self._species_interactions: dict = {}  # {species_id: float} in [-1,1]; + friendly, - hostile

    # ── State predicates ──────────────────────────────────────────────────────

    def is_alive(self) -> bool:
        return self.energy > 0.0

    def can_reproduce(self, threshold: float) -> bool:
        return self.energy >= threshold

    # ── Size-derived stats ────────────────────────────────────────────────────

    @property
    def max_energy(self) -> float:
        """Energy cap scales with size^1.5 — large creatures are energy tanks."""
        from config import CONFIG
        return CONFIG.max_energy * (self.genome.behavior.size ** 1.5)

    @property
    def effective_max_speed(self) -> float:
        """Larger creatures move more slowly (size^-0.5)."""
        from config import CONFIG
        return CONFIG.max_speed * (self.genome.behavior.size ** -0.5)

    # ── Energy ────────────────────────────────────────────────────────────────

    def apply_energy_cost(self, delta: float) -> None:
        self.energy = max(0.0, self.energy - delta)

    def add_energy(self, amount: float, cap: float) -> None:
        self.energy = min(cap, self.energy + amount)

    # ── Trail ─────────────────────────────────────────────────────────────────

    def record_trail(self, max_len: int) -> None:
        self._trail.append((float(self.pos[0]), float(self.pos[1])))
        if len(self._trail) > max_len:
            self._trail.pop(0)

    # ── Think ─────────────────────────────────────────────────────────────────

    def think(self, sensor_input: np.ndarray) -> np.ndarray:
        """Run the brain and cache the result."""
        self.last_action = self.brain.forward(sensor_input)
        return self.last_action

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name.to_dict(),
            "pos": self.pos.tolist(),
            "vel": self.vel.tolist(),
            "angle": self.angle,
            "energy": self.energy,
            "age": self.age,
            "years": self.years,
            "genome": self.genome.to_dict(),
            "parent_ids": list(self.parent_ids) if self.parent_ids else None,
            "species_id": self.species_id,
            "species_interactions": {str(k): v for k, v in self._species_interactions.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Creature":
        from core.genome import Genome
        from core.naming import CreatureName
        genome = Genome.from_dict(d["genome"])
        name = CreatureName.from_dict(d["name"])
        pos = np.array(d["pos"], dtype=np.float32)
        c = cls(genome, name, pos, d["energy"],
                tuple(d["parent_ids"]) if d["parent_ids"] else None)
        c.id = d["id"]
        c.vel = np.array(d["vel"], dtype=np.float32)
        c.angle = d["angle"]
        c.age = d["age"]
        c.years = d.get("years", d["age"] // 5000)
        c.species_id = d["species_id"]
        c.color = genome.appearance.primary_rgb()
        raw_si = d.get("species_interactions", {})
        c._species_interactions = {int(k): v for k, v in raw_si.items()}
        return c


# ── Helpers ───────────────────────────────────────────────────────────────────

def _color_from_lineage(lineage_id: int) -> Tuple[int, int, int]:
    """
    Deterministic color from lineage_id using a hash spread across hue space.
    Returns a vivid RGB tuple.
    """
    import colorsys
    # Use lineage_id to pick a hue, keep saturation and value high
    hue = (lineage_id * 0.618033988749895) % 1.0  # golden ratio spread
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def get_next_id() -> int:
    """Return the current value of the creature ID counter (for serialization)."""
    return _next_id


def reset_id_counter(start: int = 0) -> None:
    """Used when loading a saved simulation."""
    global _next_id
    _next_id = start
