"""
ActionResolver — translates NN output vectors into world mutations.

NN output layout (6 values, all in [-1, 1] via tanh):
  0  ACT_THRUST      forward thrust
  1  ACT_TURN        turn rate
  2  ACT_EAT         > eat_threshold       → attempt eat nearest food
                       energy gain scaled by (1 - carnivore_bias)
  3  ACT_REPRODUCE   > reproduce_threshold → attempt reproduce
                       mode determined by sexual_bias gene
  4  ACT_ATTACK      > attack_threshold    → attack nearest creature
                       energy gain scaled by carnivore_bias (primary source for predators)
  5  ACT_FLEE        > flee_threshold      → override movement: turn 180 + max thrust

Reproduction modes:
  Sexual (roll < sexual_bias):   two consenting same-species partners, each pays reproduce_cost.
  Asexual (roll >= sexual_bias): single parent, parent pays reproduce_cost, offspring is
                                  a mutated clone (no crossover).

Diet:
  carnivore_bias = 0.0  →  herbivore: full energy from food, no energy from attacks
  carnivore_bias = 1.0  →  carnivore: full energy from attacks, no energy from food
  Values in between     →  omnivore with proportional gains
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from config import SimConfig, CONFIG
from core.world import World

if TYPE_CHECKING:
    from core.creature import Creature
    from core.population import Population

_INTERACTION_DELTA = 0.25


class ActionResolver:
    def __init__(self, cfg: SimConfig = CONFIG) -> None:
        self.cfg = cfg
        # Sexual: (parent_a, parent_b)
        self._reproduce_queue:  List[Tuple["Creature", "Creature"]] = []
        # Asexual: single parent
        self._asexual_queue:    List["Creature"] = []

    def clear_queues(self) -> None:
        self._reproduce_queue.clear()
        self._asexual_queue.clear()

    # ── Per-creature resolution ───────────────────────────────────────────────

    def resolve(
        self,
        creature: "Creature",
        action: np.ndarray,
        world: World,
        population: "Population",
    ) -> float:
        beh = creature.genome.behavior
        extra_cost = 0.0

        if action[2] > beh.eat_threshold:
            extra_cost += self._try_eat(creature, world)

        if action[4] > beh.attack_threshold:
            extra_cost += self._try_attack(creature, world, population)

        if action[3] > beh.reproduce_threshold:
            self._queue_reproduce(creature, world, population)

        return extra_cost

    # ── Eat ───────────────────────────────────────────────────────────────────

    def _try_eat(self, creature: "Creature", world: World) -> float:
        """Consume nearest food resource.  Energy gain scales with herbivore fraction.
        Larger creatures reach slightly farther to eat (sqrt(size) scale)."""
        cx, cy = float(creature.pos[0]), float(creature.pos[1])
        size_reach = self.cfg.resource_radius * 2.5 * math.sqrt(creature.genome.behavior.size)
        nearby = world.get_nearby_resources(cx, cy, size_reach)
        if not nearby:
            return self.cfg.eat_cost

        resource = nearby[0]
        dx, dy = world._wrap_delta(resource.x - cx, resource.y - cy)
        if math.sqrt(dx * dx + dy * dy) <= size_reach:
            raw_energy  = world.consume_resource(resource.idx)
            herb_frac   = 1.0 - creature.genome.behavior.carnivore_bias
            creature.add_energy(raw_energy * herb_frac, creature.max_energy)
            creature._food_eaten += 1
        return self.cfg.eat_cost

    # ── Attack ────────────────────────────────────────────────────────────────

    def _try_attack(
        self, creature: "Creature", world: World, population: "Population"
    ) -> float:
        """
        Deal damage to nearest creature.  Energy returned to attacker scales with
        carnivore_bias — the primary energy source for carnivores.
        """
        cx, cy = float(creature.pos[0]), float(creature.pos[1])
        nearby_ids = world.get_nearby_creature_ids(cx, cy, self.cfg.creature_radius * 3)
        closest:      Optional["Creature"] = None
        closest_dist: float = float("inf")

        for cid in nearby_ids:
            if cid == creature.id:
                continue
            other = population.get_by_id(cid)
            if other is None:
                continue
            dx, dy = world._wrap_delta(float(other.pos[0]) - cx, float(other.pos[1]) - cy)
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < closest_dist:
                closest_dist = dist
                closest = other

        if closest and closest_dist <= self.cfg.creature_radius * 3:
            # Larger attackers deal more damage; larger targets take damage to their larger HP pool
            damage       = self.cfg.attack_damage * creature.genome.behavior.size
            actual_dmg   = min(damage, closest.energy)
            closest.apply_energy_cost(actual_dmg)

            # Carnivores convert damage to energy; herbivores gain almost nothing.
            # carnivore_efficiency cap keeps pure carnivory slightly less efficient
            # than herbivory (prevents trivial dominance).
            carni_frac = creature.genome.behavior.carnivore_bias * self.cfg.carnivore_efficiency
            creature.add_energy(actual_dmg * carni_frac, creature.max_energy)

            # Record inter-species interaction
            if closest.species_id != creature.species_id:
                _update_interaction(creature, closest.species_id, +_INTERACTION_DELTA)
                _update_interaction(closest,  creature.species_id, -_INTERACTION_DELTA)

        return self.cfg.attack_cost

    # ── Reproduce ─────────────────────────────────────────────────────────────

    def _queue_reproduce(
        self, creature: "Creature", world: World, population: "Population"
    ) -> None:
        if not creature.can_reproduce(self.cfg.reproduce_energy_threshold):
            return

        already_sexual  = {c.id for pair in self._reproduce_queue  for c in pair}
        already_asexual = {c.id for c in self._asexual_queue}
        if creature.id in already_sexual or creature.id in already_asexual:
            return

        if random.random() < creature.genome.behavior.sexual_bias:
            # ── Sexual ────────────────────────────────────────────────────────
            self._queue_sexual(creature, world, population, already_sexual | already_asexual)
        else:
            # ── Asexual ───────────────────────────────────────────────────────
            self._asexual_queue.append(creature)

    def _queue_sexual(
        self, creature: "Creature", world: World, population: "Population",
        already_queued: set,
    ) -> None:
        cx, cy = float(creature.pos[0]), float(creature.pos[1])
        nearby_ids = world.get_nearby_creature_ids(cx, cy, self.cfg.mate_search_radius)

        best_mate: Optional["Creature"] = None
        best_dist: float = float("inf")

        for cid in nearby_ids:
            if cid == creature.id or cid in already_queued:
                continue
            other = population.get_by_id(cid)
            if other is None:
                continue
            if not other.can_reproduce(self.cfg.reproduce_energy_threshold):
                continue
            if other.species_id != creature.species_id and creature.species_id != 0:
                continue
            dx, dy = world._wrap_delta(float(other.pos[0]) - cx, float(other.pos[1]) - cy)
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_mate = other

        if best_mate:
            self._reproduce_queue.append((creature, best_mate))

    def get_reproduce_queue(self) -> List[Tuple["Creature", "Creature"]]:
        return list(self._reproduce_queue)

    def get_asexual_queue(self) -> List["Creature"]:
        return list(self._asexual_queue)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _update_interaction(creature: "Creature", species_id: int, delta: float) -> None:
    current = creature._species_interactions.get(species_id, 0.0)
    creature._species_interactions[species_id] = max(-1.0, min(1.0, current + delta))
