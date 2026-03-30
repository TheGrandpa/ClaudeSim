"""
Lineage tracking — family tree data and history retention.

LineageRecord   — immutable snapshot of a creature at death (or birth)
LineageRegistry — stores records for all creatures ever born, pruned to
                  max_generations deep per lineage branch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class LineageRecord:
    """
    Lightweight snapshot stored when a creature is born and updated when it dies.
    Does NOT hold the full genome (too expensive) — stores key stats only.
    The genome can be exported separately via the serializer.
    """
    creature_id: int
    name_short: str          # e.g. "Rath-vel-kin"
    name_full: str           # e.g. "Rath-vel-kin  (Gen 7)"
    generation: int          # name generation depth
    parent_ids: Optional[Tuple[int, int]]
    species_id: int
    birth_tick: int
    death_tick: Optional[int] = None
    max_energy: float = 0.0
    total_food_eaten: int = 0
    children_count: int = 0
    color: Tuple[int, int, int] = (200, 200, 200)

    @property
    def is_alive(self) -> bool:
        return self.death_tick is None

    @property
    def lifespan(self) -> Optional[int]:
        if self.death_tick is None:
            return None
        return self.death_tick - self.birth_tick

    def to_dict(self) -> dict:
        return {
            "creature_id": self.creature_id,
            "name_short": self.name_short,
            "name_full": self.name_full,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids) if self.parent_ids else None,
            "species_id": self.species_id,
            "birth_tick": self.birth_tick,
            "death_tick": self.death_tick,
            "max_energy": self.max_energy,
            "total_food_eaten": self.total_food_eaten,
            "children_count": self.children_count,
            "color": list(self.color),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LineageRecord":
        return cls(
            creature_id=d["creature_id"],
            name_short=d["name_short"],
            name_full=d["name_full"],
            generation=d["generation"],
            parent_ids=tuple(d["parent_ids"]) if d["parent_ids"] else None,
            species_id=d["species_id"],
            birth_tick=d["birth_tick"],
            death_tick=d["death_tick"],
            max_energy=d["max_energy"],
            total_food_eaten=d["total_food_eaten"],
            children_count=d["children_count"],
            color=tuple(d["color"]),
        )


class LineageRegistry:
    """
    Stores LineageRecords for all creatures ever born.

    Pruning: once a lineage branch exceeds max_generations, records older
    than max_generations from the youngest living descendant are removed.
    Pruning runs lazily every prune_interval births to avoid per-tick overhead.
    """

    def __init__(self, max_generations: int = 100, prune_interval: int = 500) -> None:
        self.max_generations = max_generations
        self.prune_interval = prune_interval
        self._records: Dict[int, LineageRecord] = {}   # creature_id -> record
        self._birth_count: int = 0

    # ── Write ─────────────────────────────────────────────────────────────────

    def register_birth(self, creature, tick: int) -> None:
        """Call immediately after a creature is created."""
        record = LineageRecord(
            creature_id=creature.id,
            name_short=creature.name.short(),
            name_full=creature.name.full(),
            generation=creature.name.generation,
            parent_ids=creature.parent_ids,
            species_id=creature.species_id,
            birth_tick=tick,
            color=creature.genome.appearance.primary_rgb(),
        )
        self._records[creature.id] = record
        self._birth_count += 1

        # Increment parent children counts
        if creature.parent_ids:
            for pid in creature.parent_ids:
                if pid in self._records:
                    self._records[pid].children_count += 1

        if self._birth_count % self.prune_interval == 0:
            self._prune()

    def register_death(self, creature_id: int, tick: int) -> None:
        if creature_id in self._records:
            self._records[creature_id].death_tick = tick

    def update_stats(self, creature_id: int, energy: float, food_eaten: int) -> None:
        if creature_id in self._records:
            rec = self._records[creature_id]
            if energy > rec.max_energy:
                rec.max_energy = energy
            rec.total_food_eaten = food_eaten

    def update_species(self, creature_id: int, species_id: int) -> None:
        if creature_id in self._records:
            self._records[creature_id].species_id = species_id

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, creature_id: int) -> Optional[LineageRecord]:
        return self._records.get(creature_id)

    def ancestors(self, creature_id: int, max_depth: int = 100) -> List[LineageRecord]:
        """
        BFS upward through parent links.  Returns records oldest-first.
        Depth-limited to max_depth to handle very long chains efficiently.
        """
        visited: set = set()
        result: List[LineageRecord] = []
        queue = [creature_id]
        depth = 0

        while queue and depth < max_depth:
            next_queue = []
            for cid in queue:
                if cid in visited:
                    continue
                visited.add(cid)
                rec = self._records.get(cid)
                if rec is None:
                    continue
                result.append(rec)
                if rec.parent_ids:
                    next_queue.extend(rec.parent_ids)
            queue = next_queue
            depth += 1

        result.reverse()
        return result

    def descendants(self, creature_id: int, max_depth: int = 100) -> List[LineageRecord]:
        """
        BFS downward through children.  Efficient because we index by parent.
        Builds a reverse index on demand (cached until next birth/death).
        """
        children_index = self._build_children_index()
        visited: set = set()
        result: List[LineageRecord] = []
        queue = [creature_id]
        depth = 0

        while queue and depth < max_depth:
            next_queue = []
            for cid in queue:
                if cid in visited:
                    continue
                visited.add(cid)
                rec = self._records.get(cid)
                if rec and cid != creature_id:
                    result.append(rec)
                for child_id in children_index.get(cid, []):
                    if child_id not in visited:
                        next_queue.append(child_id)
            queue = next_queue
            depth += 1

        return result

    def full_tree(self, creature_id: int) -> Dict[int, LineageRecord]:
        """
        Returns a dict of all records in the family tree (ancestors + descendants)
        centered on creature_id.  Used by the family tree visualizer.
        """
        tree: Dict[int, LineageRecord] = {}
        for rec in self.ancestors(creature_id, self.max_generations):
            tree[rec.creature_id] = rec
        for rec in self.descendants(creature_id, self.max_generations):
            tree[rec.creature_id] = rec
        if creature_id in self._records:
            tree[creature_id] = self._records[creature_id]
        return tree

    def all_records(self) -> Dict[int, LineageRecord]:
        return dict(self._records)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_children_index(self) -> Dict[int, List[int]]:
        index: Dict[int, List[int]] = {}
        for cid, rec in self._records.items():
            if rec.parent_ids:
                for pid in rec.parent_ids:
                    index.setdefault(pid, []).append(cid)
        return index

    def _prune(self) -> None:
        """
        Remove records for creatures that are:
          - dead
          - have no living descendants
          - are more than max_generations above the youngest living generation
        """
        if not self._records:
            return

        living_generations = [
            r.generation for r in self._records.values() if r.is_alive
        ]
        if not living_generations:
            return

        youngest_gen = max(living_generations)
        cutoff_gen = youngest_gen - self.max_generations

        to_remove = [
            cid for cid, r in self._records.items()
            if not r.is_alive and r.generation < cutoff_gen
        ]
        for cid in to_remove:
            del self._records[cid]

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "max_generations": self.max_generations,
            "records": [r.to_dict() for r in self._records.values()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LineageRegistry":
        reg = cls(max_generations=d["max_generations"])
        for rd in d["records"]:
            rec = LineageRecord.from_dict(rd)
            reg._records[rec.creature_id] = rec
        return reg
