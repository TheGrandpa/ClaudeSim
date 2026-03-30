"""
Sensors — build the 55-element input vector for a creature's neural network.

Layout (matches config.neat_input_size = 55):

  [0:24]   Ray sensors  — 6 rays × 4 values each
               (wall_hit, creature_hit, food_hit, normalized_distance)
               Ray length comes from creature.genome.behavior.ray_length.

  [24:33]  Nearest 3 food items — 3 × (sin_angle, cos_angle, norm_dist)

  [33:48]  Nearest 3 creatures  — 3 × (sin_angle, cos_angle, norm_dist,
                                        energy_ratio, kin_foe_signal)
               kin_foe_signal: +1.0 = same species (kin)
                                0.0 = unknown/neutral different species
                               -1..+1 = prior interaction score if any

  [48:50]  Hunger direction — (sin, cos) toward nearest food,
               scaled by (1 - energy_norm) * food_priority
               Strong when hungry, silent when full.

  [50:55]  Self state — (speed_norm, energy_norm, years_norm,
                          heading_sin, heading_cos)
               years_norm = min(years / 5, 1.0)  (saturates at 5 years)
"""

from __future__ import annotations

import math
from typing import List, TYPE_CHECKING

import numpy as np

from config import SimConfig, CONFIG
from core.world import World

if TYPE_CHECKING:
    from core.creature import Creature
    from core.population import Population


# ── Layout constants ──────────────────────────────────────────────────────────

SENSOR_SIZE      = 55
RAY_COUNT        = 6
NEAREST_FOOD     = 3
NEAREST_CREATURE = 3

RAY_OFFSET      = 0        # 6 × 4 = 24
FOOD_OFFSET     = 24       # 3 × 3 = 9
CREATURE_OFFSET = 33       # 3 × 5 = 15
HUNGER_OFFSET   = 48       # 2
SELF_OFFSET     = 50       # 5


def build_sensor_vector(
    creature: "Creature",
    world: World,
    population: "Population",
    cfg: SimConfig = CONFIG,
) -> np.ndarray:
    """Construct the full 55-element sensor vector for one creature."""
    vec = np.zeros(SENSOR_SIZE, dtype=np.float32)
    cx, cy = float(creature.pos[0]), float(creature.pos[1])
    angle  = creature.angle
    beh    = creature.genome.behavior

    # ── Ray sensors ───────────────────────────────────────────────────────────
    ray_len = beh.ray_length
    ray_angles = [angle + i * (2 * math.pi / RAY_COUNT) for i in range(RAY_COUNT)]
    for i, ray_angle in enumerate(ray_angles):
        wall_hit, creature_hit, food_hit, dist_norm = _cast_ray(
            cx, cy, ray_angle, ray_len, world, population, creature.id, cfg
        )
        base = RAY_OFFSET + i * 4
        vec[base + 0] = wall_hit
        vec[base + 1] = creature_hit
        vec[base + 2] = food_hit
        vec[base + 3] = dist_norm

    # ── Nearest food ──────────────────────────────────────────────────────────
    nearby_food = world.get_nearby_resources(cx, cy, cfg.sense_radius)
    nearest_food_dir = None   # (sin, cos) to closest food — used for hunger signal below

    for i in range(NEAREST_FOOD):
        base = FOOD_OFFSET + i * 3
        if i < len(nearby_food):
            f = nearby_food[i]
            dx, dy = world._wrap_delta(f.x - cx, f.y - cy)
            dist = math.sqrt(dx * dx + dy * dy) + 1e-6
            rel_angle = math.atan2(dy, dx) - angle
            vec[base + 0] = math.sin(rel_angle)
            vec[base + 1] = math.cos(rel_angle)
            vec[base + 2] = 1.0 - min(dist / cfg.sense_radius, 1.0)
            if i == 0:
                nearest_food_dir = (math.sin(rel_angle), math.cos(rel_angle))

    # ── Nearest creatures (with kin/foe signal) ───────────────────────────────
    nearby_ids = world.get_nearby_creature_ids(cx, cy, cfg.sense_radius)
    nearby_creatures = _sort_nearby_creatures(
        creature.id, cx, cy, nearby_ids, population, cfg.sense_radius, world
    )
    for i in range(NEAREST_CREATURE):
        base = CREATURE_OFFSET + i * 5
        if i < len(nearby_creatures):
            other, dist = nearby_creatures[i]
            dx, dy = world._wrap_delta(float(other.pos[0]) - cx, float(other.pos[1]) - cy)
            rel_angle = math.atan2(dy, dx) - angle
            energy_ratio = min(other.energy / max(creature.energy, 1.0), 2.0) / 2.0

            # Kin/foe signal
            if other.species_id == creature.species_id and creature.species_id != 0:
                kin_foe = 1.0   # same species = kin
            else:
                # Prior interaction score in [-1, 1]; 0 = unknown
                kin_foe = creature._species_interactions.get(other.species_id, 0.0)

            vec[base + 0] = math.sin(rel_angle)
            vec[base + 1] = math.cos(rel_angle)
            vec[base + 2] = 1.0 - min(dist / cfg.sense_radius, 1.0)
            vec[base + 3] = energy_ratio
            vec[base + 4] = kin_foe

    # ── Hunger direction ──────────────────────────────────────────────────────
    energy_norm    = min(creature.energy / creature.max_energy, 1.0)
    hunger         = 1.0 - energy_norm                      # 0=full, 1=starving
    hunger_strength = hunger * beh.food_priority            # scaled by genetic drive
    if nearest_food_dir is not None:
        vec[HUNGER_OFFSET + 0] = nearest_food_dir[0] * hunger_strength
        vec[HUNGER_OFFSET + 1] = nearest_food_dir[1] * hunger_strength
    # else: stays 0 — no food in sense range

    # ── Self state ────────────────────────────────────────────────────────────
    speed = math.sqrt(float(creature.vel[0] ** 2 + creature.vel[1] ** 2))
    vec[SELF_OFFSET + 0] = min(speed / creature.effective_max_speed, 1.0)
    vec[SELF_OFFSET + 1] = energy_norm
    vec[SELF_OFFSET + 2] = min(creature.years / 5.0, 1.0)  # saturates at 5 years
    vec[SELF_OFFSET + 3] = math.sin(angle)
    vec[SELF_OFFSET + 4] = math.cos(angle)

    return vec


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cast_ray(
    ox: float, oy: float, angle: float, length: float,
    world: World, population: "Population", self_id: int,
    cfg: SimConfig,
) -> tuple:
    """
    Step along ray in increments.  Returns (wall_hit, creature_hit, food_hit, dist_norm).
    All hit values are 0 or 1; dist_norm in [0, 1] (1 = hit at origin, 0 = no hit / far).
    """
    step  = cfg.creature_radius * 1.5
    steps = max(1, int(length / step))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    for s in range(1, steps + 1):
        t = s * step
        # Wrap ray position toroidally — no walls, rays travel through edges
        x = (ox + cos_a * t) % world.width
        y = (oy + sin_a * t) % world.height

        # Creature check — larger creatures subtend a bigger detection sphere
        nearby_ids = world.get_nearby_creature_ids(x, y, cfg.creature_radius * 3)
        for cid in nearby_ids:
            if cid == self_id:
                continue
            other = population.get_by_id(cid)
            if other is None:
                continue
            dx, dy = world._wrap_delta(float(other.pos[0]) - x, float(other.pos[1]) - y)
            detect_r = cfg.creature_radius * 2 * other.genome.behavior.size
            if dx * dx + dy * dy < detect_r * detect_r:
                dist_norm = 1.0 - t / length
                return (0.0, 1.0, 0.0, dist_norm)

        # Food check
        nearby_food = world.get_nearby_resources(x, y, cfg.resource_radius * 2)
        if nearby_food:
            dist_norm = 1.0 - t / length
            return (0.0, 0.0, 1.0, dist_norm)

    return (0.0, 0.0, 0.0, 0.0)


def _sort_nearby_creatures(
    self_id: int, cx: float, cy: float,
    nearby_ids: set, population: "Population", max_dist: float,
    world: "World",
) -> List[tuple]:
    result = []
    max_dist_sq = max_dist * max_dist
    for cid in nearby_ids:
        if cid == self_id:
            continue
        other = population.get_by_id(cid)
        if other is None:
            continue
        dx, dy = world._wrap_delta(float(other.pos[0]) - cx, float(other.pos[1]) - cy)
        dist_sq = dx * dx + dy * dy
        if dist_sq <= max_dist_sq:
            result.append((other, math.sqrt(dist_sq)))
    result.sort(key=lambda x: x[1])
    return result
