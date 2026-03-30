"""
Physics — movement integration and wall collision.
"""

from __future__ import annotations

import math
import numpy as np

from config import SimConfig, CONFIG
from core.creature import Creature
from core.world import World


def integrate(creature: Creature, thrust: float, turn: float, cfg: SimConfig = CONFIG) -> None:
    """
    Update creature position and velocity for one tick.

    thrust  ∈ [-1, 1]  — mapped to [-max_speed, max_speed]
    turn    ∈ [-1, 1]  — mapped to [-max_turn_rate, max_turn_rate] rad/tick
    """
    # Update heading
    creature.angle += turn * cfg.max_turn_rate
    creature.angle %= (2 * math.pi)

    # Size scales max speed: large creatures are slower (size^-0.5)
    effective_max_speed = cfg.max_speed * (creature.genome.behavior.size ** -0.5)

    # Compute desired velocity delta from thrust
    speed = thrust * effective_max_speed
    dx = math.cos(creature.angle) * speed
    dy = math.sin(creature.angle) * speed

    # Apply thrust as acceleration (add to velocity)
    creature.vel[0] = creature.vel[0] * cfg.velocity_damping + dx * (1.0 - cfg.velocity_damping)
    creature.vel[1] = creature.vel[1] * cfg.velocity_damping + dy * (1.0 - cfg.velocity_damping)

    # Clamp velocity magnitude to size-adjusted cap
    speed_sq = creature.vel[0] ** 2 + creature.vel[1] ** 2
    if speed_sq > effective_max_speed ** 2:
        scale = effective_max_speed / math.sqrt(speed_sq)
        creature.vel[0] *= scale
        creature.vel[1] *= scale

    # Integrate position
    old_x, old_y = float(creature.pos[0]), float(creature.pos[1])
    creature.pos[0] += creature.vel[0]
    creature.pos[1] += creature.vel[1]

    # Toroidal wrap — seamlessly cross world edges
    creature.pos[0] = creature.pos[0] % cfg.world_width
    creature.pos[1] = creature.pos[1] % cfg.world_height

    return old_x, old_y


def movement_energy_cost(creature: Creature, cfg: SimConfig = CONFIG) -> float:
    """Energy cost proportional to speed squared, scaled by sqrt(size) — big bodies cost more."""
    speed_sq = float(creature.vel[0] ** 2 + creature.vel[1] ** 2)
    size_factor = creature.genome.behavior.size ** 0.5
    return speed_sq * cfg.movement_cost_factor * size_factor
