"""
Speciation — group creatures into species using genome compatibility distance.

Each generation:
  1. Clear species membership lists (keep species representatives)
  2. For each creature, find the first species whose representative is within
     the compatibility threshold.  If none found, create a new species.
  3. Remove stagnant or empty species.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from config import SimConfig, CONFIG
from core.genome import Genome

if TYPE_CHECKING:
    from core.creature import Creature


@dataclass
class Species:
    species_id: int
    representative: Genome          # genome used for distance comparisons
    members: List["Creature"] = field(default_factory=list)
    best_fitness: float = 0.0
    stagnation: int = 0             # generations without improvement
    age: int = 0


class Speciator:
    def __init__(self, cfg: SimConfig = CONFIG) -> None:
        self.cfg = cfg
        self._species: Dict[int, Species] = {}
        self._next_species_id: int = 1

    @property
    def species(self) -> List[Species]:
        return list(self._species.values())

    def speciate(self, creatures: List["Creature"]) -> None:
        """
        Assign each creature to a species.  Updates creature.species_id in-place.
        """
        # Clear member lists, keep representatives
        for sp in self._species.values():
            sp.members.clear()
            sp.age += 1

        for creature in creatures:
            assigned = False
            for sp in self._species.values():
                dist = creature.genome.compatibility_distance(
                    sp.representative,
                    self.cfg.c1, self.cfg.c2, self.cfg.c3
                )
                if dist < self.cfg.species_compatibility_threshold:
                    sp.members.append(creature)
                    creature.species_id = sp.species_id
                    assigned = True
                    break

            if not assigned:
                new_sp = Species(
                    species_id=self._next_species_id,
                    representative=creature.genome.copy(),
                )
                new_sp.members.append(creature)
                creature.species_id = self._next_species_id
                self._species[self._next_species_id] = new_sp
                self._next_species_id += 1

        # Update representatives to a random member of each species
        for sp in self._species.values():
            if sp.members:
                sp.representative = random.choice(sp.members).genome.copy()

        # Track stagnation
        for sp in self._species.values():
            if sp.members:
                current_best = max(c.energy for c in sp.members)
                if current_best > sp.best_fitness:
                    sp.best_fitness = current_best
                    sp.stagnation = 0
                else:
                    sp.stagnation += 1

        # Cull empty and stagnant species
        to_remove = [
            sid for sid, sp in self._species.items()
            if not sp.members or sp.stagnation > self.cfg.species_stagnation_limit
        ]
        for sid in to_remove:
            del self._species[sid]

    def species_count(self) -> int:
        return len(self._species)

    def to_dict(self) -> dict:
        return {
            "next_species_id": self._next_species_id,
            "species": [
                {
                    "species_id": sp.species_id,
                    "representative": sp.representative.to_dict(),
                    "best_fitness": sp.best_fitness,
                    "stagnation": sp.stagnation,
                    "age": sp.age,
                }
                for sp in self._species.values()
            ],
        }

    @classmethod
    def from_dict(cls, d: dict, cfg: SimConfig = CONFIG) -> "Speciator":
        s = cls(cfg)
        s._next_species_id = d["next_species_id"]
        for sd in d["species"]:
            sp = Species(
                species_id=sd["species_id"],
                representative=Genome.from_dict(sd["representative"]),
                best_fitness=sd["best_fitness"],
                stagnation=sd["stagnation"],
                age=sd["age"],
            )
            s._species[sp.species_id] = sp
        return s
