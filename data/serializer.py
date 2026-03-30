"""
Serializer — save and load full simulation state, plus individual genome export/import.

Save format:
  saves/<name>/world.json      — world resources
  saves/<name>/population.json — all living creatures (including genomes)
  saves/<name>/lineage.json    — lineage registry
  saves/<name>/meta.json       — tick count, config snapshot, innovation registry

Genome export:
  saves/genomes/<creature_name>_<id>.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from config import SimConfig, CONFIG
from core.genome import INNOVATION_REGISTRY, Genome
from core.creature import Creature, reset_id_counter, get_next_id

if TYPE_CHECKING:
    from core.world import World
    from core.population import Population
    from core.lineage import LineageRegistry
    from simulation.loop import SimulationLoop
    from evolution.speciation import Speciator


class Serializer:
    def __init__(self, cfg: SimConfig = CONFIG) -> None:
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "genomes"), exist_ok=True)

    # ── Full world save ───────────────────────────────────────────────────────

    def save(
        self,
        loop: "SimulationLoop",
        population: "Population",
        lineage: "LineageRegistry",
        speciator: "Speciator",
        name: Optional[str] = None,
    ) -> str:
        if name is None:
            name = datetime.now().strftime("save_%Y%m%d_%H%M%S")

        save_path = os.path.join(self.cfg.save_dir, name)
        os.makedirs(save_path, exist_ok=True)

        # Meta
        meta = {
            "tick": loop.tick_count,
            "total_births": loop.total_births,
            "total_deaths": loop.total_deaths,
            "innovation_registry": INNOVATION_REGISTRY.to_dict(),
            "speciator": speciator.to_dict(),
            "next_creature_id": get_next_id(),
        }
        self._write_json(os.path.join(save_path, "meta.json"), meta)

        # World
        self._write_json(os.path.join(save_path, "world.json"), loop.world.to_dict())

        # Population
        self._write_json(os.path.join(save_path, "population.json"), population.to_dict())

        # Lineage
        self._write_json(os.path.join(save_path, "lineage.json"), lineage.to_dict())

        print(f"[Save] Saved to {save_path}  (tick {loop.tick_count})")
        return save_path

    # ── Full world load ───────────────────────────────────────────────────────

    def load(
        self,
        loop: "SimulationLoop",
        population: "Population",
        lineage: "LineageRegistry",
        speciator: "Speciator",
        name: str,
    ) -> bool:
        save_path = os.path.join(self.cfg.save_dir, name)
        if not os.path.isdir(save_path):
            print(f"[Load] Save not found: {save_path}")
            return False

        try:
            meta       = self._read_json(os.path.join(save_path, "meta.json"))
            world_data = self._read_json(os.path.join(save_path, "world.json"))
            pop_data   = self._read_json(os.path.join(save_path, "population.json"))
            lin_data   = self._read_json(os.path.join(save_path, "lineage.json"))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[Load] Error reading save: {e}")
            return False

        # Restore innovation registry
        reg = INNOVATION_REGISTRY
        reg_data = meta.get("innovation_registry", {})
        if reg_data:
            reg.set_counters(reg_data.get("global_counter", 0), reg_data.get("node_counter", 0))

        # Restore creature id counter
        reset_id_counter(meta.get("next_creature_id", 0))

        # Restore world resources
        loop.world.load_resources_from_dict(world_data)

        # Restore population
        population.load_from_dict(pop_data)

        # Restore lineage
        from core.lineage import LineageRegistry as LR
        new_lineage = LR.from_dict(lin_data)
        lineage._records = new_lineage._records
        lineage.max_generations = new_lineage.max_generations

        # Restore speciator
        from evolution.speciation import Speciator as SP
        new_spec = SP.from_dict(meta.get("speciator", {}), self.cfg)
        speciator._species = new_spec._species
        speciator._next_species_id = new_spec._next_species_id

        # Restore loop counters
        loop.tick_count = meta.get("tick", 0)
        loop._total_births = meta.get("total_births", 0)
        loop._total_deaths = meta.get("total_deaths", 0)

        print(f"[Load] Loaded from {save_path}  (tick {loop.tick_count})")
        return True

    def list_saves(self) -> list:
        entries = []
        for name in os.listdir(self.cfg.save_dir):
            path = os.path.join(self.cfg.save_dir, name)
            if os.path.isdir(path) and name != "genomes":
                meta_path = os.path.join(path, "meta.json")
                tick = 0
                if os.path.exists(meta_path):
                    try:
                        tick = self._read_json(meta_path).get("tick", 0)
                    except Exception:
                        pass
                entries.append({"name": name, "tick": tick})
        return sorted(entries, key=lambda e: e["name"], reverse=True)

    # ── Creature save / load ──────────────────────────────────────────────────

    def save_creature(self, creature: "Creature", tick: int) -> str:
        """Save a single creature (genome, appearance, name, stats) to disk."""
        safe_name = creature.name.short().replace("-", "_").lower()
        filename = f"{safe_name}_{creature.id}.json"
        path = os.path.join(self.cfg.save_dir, "genomes", filename)
        data = {
            "format": "creature_v1",
            "saved_at_tick": tick,
            "creature_id": creature.id,
            "name": creature.name.to_dict(),
            "genome": creature.genome.to_dict(),   # includes appearance
            "age": creature.age,
            "energy": creature.energy,
            "species_id": creature.species_id,
            "parent_ids": list(creature.parent_ids) if creature.parent_ids else None,
        }
        self._write_json(path, data)
        print(f"[Creature] Saved {creature.name.short()} → {filename}")
        return filename

    def load_creature(self, filename: str) -> Optional["Creature"]:
        """
        Load a creature from a saved file.  Returns a new Creature ready to
        be spawned into the simulation (position set by caller).
        """
        path = os.path.join(self.cfg.save_dir, "genomes", filename)
        if not os.path.exists(path):
            print(f"[Creature] File not found: {path}")
            return None
        try:
            data = self._read_json(path)
            from core.genome import Genome
            from core.naming import CreatureName
            import numpy as np

            genome = Genome.from_dict(data["genome"])
            name   = CreatureName.from_dict(data["name"])
            pos    = np.array([0.0, 0.0], dtype=np.float32)   # caller sets real pos
            parent_ids = tuple(data["parent_ids"]) if data.get("parent_ids") else None

            creature = Creature(genome, name, pos, self.cfg.initial_energy, parent_ids)
            creature.species_id = data.get("species_id", 0)
            print(f"[Creature] Loaded {name.short()} from {filename}")
            return creature
        except Exception as e:
            print(f"[Creature] Error loading {filename}: {e}")
            return None

    def list_saved_creatures(self) -> list:
        """Return list of dicts describing each saved creature file."""
        folder = os.path.join(self.cfg.save_dir, "genomes")
        entries = []
        for fname in os.listdir(folder):
            if not fname.endswith(".json"):
                continue
            try:
                data = self._read_json(os.path.join(folder, fname))
                if data.get("format") != "creature_v1":
                    continue
                entries.append({
                    "filename": fname,
                    "name":     data.get("name", {}).get("syllables", ["?"]),
                    "name_str": "-".join(s.capitalize() for s in data.get("name", {}).get("syllables", ["?"])),
                    "generation": data.get("name", {}).get("generation", 0),
                    "age":        data.get("age", 0),
                    "tick":       data.get("saved_at_tick", 0),
                    "species_id": data.get("species_id", 0),
                })
            except Exception:
                continue
        return sorted(entries, key=lambda e: e["filename"], reverse=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _write_json(path: str, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))

    @staticmethod
    def _read_json(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
