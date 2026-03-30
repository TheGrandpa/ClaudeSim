"""
SimulationLoop — orchestrates one tick of the simulation.

tick() order:
  1. Spawn resources
  2. Sense all creatures
  3. Think (NN forward pass)
  4. Physics (movement)
  5. Resolve non-movement actions (eat, attack, queue reproduction)
  6. Apply energy costs
  7. Process reproduction queue
  8. Reap dead creatures
  9. Population floor injection
  10. Speciate (every N ticks)
  11. Stats update
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from config import SimConfig, CONFIG
from core.creature import Creature
from core.genome import make_minimal_genome, INNOVATION_REGISTRY
from core.lineage import LineageRegistry
from core.naming import random_name, inherit_name
from core.population import Population
from core.world import World
from evolution.crossover import crossover
from evolution.mutation import mutate
from evolution.speciation import Speciator
from simulation.actions import ActionResolver
from simulation.physics import integrate, movement_energy_cost
from simulation.sensors import build_sensor_vector
from visualization.event_log import push as log_event
from visualization.event_log import (
    BIRTH, DEATH, FORK, EXTINCTION, ADD_NODE, ADD_CONN, INJECT, MILESTONE
)

SPECIATE_INTERVAL = 50   # ticks between speciation passes


class SimulationLoop:
    def __init__(
        self,
        world: World,
        population: Population,
        lineage: LineageRegistry,
        cfg: SimConfig = CONFIG,
    ) -> None:
        self.world = world
        self.population = population
        self.lineage = lineage
        self.cfg = cfg
        self.tick_count: int = 0
        self.speciator = Speciator(cfg)
        self.resolver = ActionResolver(cfg)

        # Stats (rolling)
        self._total_births: int = 0
        self._total_deaths: int = 0
        self._known_species: set = set()
        self._milestone_births = {100, 500, 1000, 5000, 10000}

    # ── Main tick ─────────────────────────────────────────────────────────────

    def tick(self) -> None:
        self.tick_count += 1
        self.resolver.clear_queues()

        # 1. Spawn resources
        self.world.tick_resource_spawn()

        creatures = self.population.creatures

        for creature in creatures:
            if not creature.is_alive():
                continue

            # 2 & 3. Sense + Think
            sensor = build_sensor_vector(creature, self.world, self.population, self.cfg)
            action = creature.think(sensor)

            # Extract broadcast signal (output 6); map tanh [-1,1] → [0,1]
            # Graceful fallback for genomes with only 6 outputs (old saves).
            raw_sig = float(action[6]) if len(action) > 6 else 0.0
            creature.signal = (raw_sig + 1.0) * 0.5   # [-1,1] → [0,1]

            # 4. Physics — check flee override before integrating
            beh = creature.genome.behavior
            if action[5] > beh.flee_threshold:
                # Turn 180 and run at max speed away from threat
                creature.angle = (creature.angle + math.pi) % (2 * math.pi)
                thrust = 1.0
                turn   = 0.0
            else:
                thrust = float(action[0])
                turn   = float(action[1])

            old_x, old_y = integrate(creature, thrust, turn, self.cfg)
            self.population.sync_position(creature, old_x, old_y)

            # Trail
            if self.cfg.show_trails:
                creature.record_trail(self.cfg.trail_length)

            # 5. Non-movement actions
            extra_cost = self.resolver.resolve(creature, action, self.world, self.population)

            # 6. Energy costs
            # Metabolic base cost scales linearly with size — big bodies cost more to run
            nn_cost    = creature.brain.weight_count() * self.cfg.nn_cost_per_weight
            move_cost  = movement_energy_cost(creature, self.cfg)
            total_cost = (
                self.cfg.metabolic_cost_per_tick * creature.genome.behavior.size
                + nn_cost
                + move_cost
                + extra_cost
            )
            creature.apply_energy_cost(total_cost)
            creature.age += 1

            # Increment year counter when a year boundary is crossed
            creature.years = creature.age // self.cfg.ticks_per_year

            # Update lineage stats
            food_eaten = getattr(creature, "_food_eaten", 0)
            self.lineage.update_stats(creature.id, creature.energy, food_eaten)

        # 7. Process reproduction
        self._process_reproduction()

        # 8. Reap dead
        dead = self.population.reap_dead()
        for c in dead:
            self.lineage.register_death(c.id, self.tick_count)
            self._total_deaths += 1
            log_event(DEATH,
                f"{c.name.short()} starved (age {c.age})",
                self.tick_count)

        # 9. Population floor
        if (self.cfg.population_floor_enabled
                and self.population.count < self.cfg.population_floor):
            n = self.cfg.population_floor - self.population.count
            self._inject_random_creatures(n)
            log_event(INJECT,
                f"{n} new creature(s) injected (pop floor)",
                self.tick_count)

        # 10. Speciate periodically
        if self.tick_count % SPECIATE_INTERVAL == 0:
            before_ids = set(sp.species_id for sp in self.speciator.species)
            self.speciator.speciate(self.population.creatures)
            after_ids  = set(sp.species_id for sp in self.speciator.species)

            for new_sid in after_ids - before_ids:
                sp = next((s for s in self.speciator.species if s.species_id == new_sid), None)
                rep_name = sp.members[0].name.short() if sp and sp.members else "?"
                log_event(FORK,
                    f"Species {new_sid} emerged from {rep_name}",
                    self.tick_count)

            for dead_sid in before_ids - after_ids:
                log_event(EXTINCTION,
                    f"Species {dead_sid} went extinct",
                    self.tick_count)

            for c in self.population.creatures:
                self.lineage.update_species(c.id, c.species_id)

        # 11. Innovation registry — new generation boundary
        if self.tick_count % SPECIATE_INTERVAL == 0:
            INNOVATION_REGISTRY.new_generation()

        # 12. Decay species interaction scores (runs every N ticks)
        if self.tick_count % self.cfg.interaction_decay_interval == 0:
            decay = self.cfg.interaction_decay_rate
            for c in self.population.creatures:
                dead_keys = []
                for sid, val in c._species_interactions.items():
                    new_val = val * decay
                    if abs(new_val) < 0.01:
                        dead_keys.append(sid)
                    else:
                        c._species_interactions[sid] = new_val
                for k in dead_keys:
                    del c._species_interactions[k]

    # ── Reproduction ──────────────────────────────────────────────────────────

    def _process_reproduction(self) -> None:
        # ── Sexual reproduction ───────────────────────────────────────────────
        pairs = self.resolver.get_reproduce_queue()
        for parent_a, parent_b in pairs:
            if self.population.count >= self.cfg.max_population:
                break
            if not parent_a.is_alive() or not parent_b.is_alive():
                continue
            if (not parent_a.can_reproduce(self.cfg.reproduce_energy_threshold)
                    or not parent_b.can_reproduce(self.cfg.reproduce_energy_threshold)):
                continue

            parent_a.apply_energy_cost(self.cfg.reproduce_cost)
            parent_b.apply_energy_cost(self.cfg.reproduce_cost)

            child_genome = crossover(parent_a, parent_b)
            mut = mutate(child_genome, self.cfg, INNOVATION_REGISTRY)

            child_name = inherit_name(
                parent_a.name, parent_b.name,
                mutation_rate=self.cfg.syllable_mutation_rate,
                add_syllable=mut["structural"],
                min_syllables=self.cfg.name_min_syllables,
                max_syllables=self.cfg.name_max_syllables,
            )

            mid_pos = (parent_a.pos + parent_b.pos) / 2.0
            mid_pos += np.array([random.uniform(-20, 20), random.uniform(-20, 20)],
                                dtype=np.float32)
            mid_pos = self.world.wrap_position(mid_pos)

            child = Creature(
                genome=child_genome, name=child_name, pos=mid_pos,
                energy=self.cfg.initial_energy,
                parent_ids=(parent_a.id, parent_b.id),
            )
            child.species_id = parent_a.species_id

            self.population.add(child)
            self.lineage.register_birth(child, self.tick_count)
            self._total_births += 1

            log_event(BIRTH,
                f"{parent_a.name.short()} ♀♂ {parent_b.name.short()} → {child.name.short()}",
                self.tick_count)
            self._log_structural(child, mut)
            if self._total_births in self._milestone_births:
                log_event(MILESTONE, f"Birth #{self._total_births:,} reached!", self.tick_count)

        # ── Asexual reproduction ──────────────────────────────────────────────
        for parent in self.resolver.get_asexual_queue():
            if self.population.count >= self.cfg.max_population:
                break
            if not parent.is_alive():
                continue
            if not parent.can_reproduce(self.cfg.reproduce_energy_threshold):
                continue

            # Single parent pays the full cost
            parent.apply_energy_cost(self.cfg.reproduce_cost)

            # Clone + mutate (no crossover)
            child_genome = parent.genome.copy()
            mut = mutate(child_genome, self.cfg, INNOVATION_REGISTRY)

            child_name = inherit_name(
                parent.name, parent.name,
                mutation_rate=self.cfg.syllable_mutation_rate,
                add_syllable=mut["structural"],
                min_syllables=self.cfg.name_min_syllables,
                max_syllables=self.cfg.name_max_syllables,
            )

            offset = np.array([random.uniform(-30, 30), random.uniform(-30, 30)],
                              dtype=np.float32)
            child_pos = self.world.wrap_position(parent.pos + offset)

            child = Creature(
                genome=child_genome, name=child_name, pos=child_pos,
                energy=self.cfg.initial_energy,
                parent_ids=(parent.id, parent.id),   # both slots = same parent
            )
            child.species_id = parent.species_id

            self.population.add(child)
            self.lineage.register_birth(child, self.tick_count)
            self._total_births += 1

            log_event(BIRTH,
                f"{parent.name.short()} ✦ → {child.name.short()} (asexual)",
                self.tick_count)
            self._log_structural(child, mut)
            if self._total_births in self._milestone_births:
                log_event(MILESTONE, f"Birth #{self._total_births:,} reached!", self.tick_count)

    def _log_structural(self, child: "Creature", mut: dict) -> None:
        if mut["add_node"]:
            log_event(ADD_NODE,
                f"{child.name.short()} grew a node ({child.genome.weight_count()} weights)",
                self.tick_count)
        if mut["add_conn"]:
            log_event(ADD_CONN,
                f"{child.name.short()} grew a connection ({child.genome.weight_count()} weights)",
                self.tick_count)

    # ── Random injection ──────────────────────────────────────────────────────

    def _inject_random_creatures(self, count: int) -> None:
        global _lineage_counter
        for _ in range(count):
            genome = make_minimal_genome(
                self.cfg.neat_input_size,
                self.cfg.neat_output_size,
                INNOVATION_REGISTRY,
                self.cfg.weight_init_range,
            )
            genome.lineage_id = _next_lineage_id()
            name = random_name(self.cfg.name_min_syllables, self.cfg.name_max_syllables)
            pos = self.population.random_spawn_position()
            creature = Creature(genome, name, pos, self.cfg.initial_energy)
            self.population.add(creature)
            self.lineage.register_birth(creature, self.tick_count)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total_births(self) -> int:
        return self._total_births

    @property
    def total_deaths(self) -> int:
        return self._total_deaths


# ── Lineage id counter ────────────────────────────────────────────────────────
_lineage_counter: int = 0

def _next_lineage_id() -> int:
    global _lineage_counter
    _lineage_counter += 1
    return _lineage_counter
