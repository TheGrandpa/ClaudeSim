"""
Physics — movement integration and wall collision.

Stamina model
─────────────
Each creature has a stamina pool (size controlled by stamina_capacity gene).

  effort   = thrust²          (0 at rest, 1 at full sprint)
  drain    = effort × cfg.stamina_sprint_drain × creature_size
  recovery = (1 − effort) × beh.stamina_recovery
  net      = recovery − drain   (clamped to [0, stamina_capacity])

Speed is scaled by a stamina fraction so exhausted creatures slow down:
  stamina_scale = speed_floor + (1 − speed_floor) × (stamina / capacity)

At stamina=0 the creature moves at stamina_speed_floor × full_speed.
This creates natural predator/prey chase resolution and rest behaviour.
"""

from __future__ import annotations

import math
import numpy as np

from config import SimConfig, CONFIG
from core.creature import Creature
from core.world import World


def integrate(creature: Creature, thrust: float, turn: float, cfg: SimConfig = CONFIG) -> None:
    """
    Update creature position, velocity, and stamina for one tick.

    thrust  ∈ [-1, 1]  — mapped to [−effective_max_speed, +effective_max_speed]
    turn    ∈ [-1, 1]  — mapped to [−max_turn_rate, +max_turn_rate] rad/tick
    """
    beh = creature.genome.behavior

    # ── Stamina update ────────────────────────────────────────────────────────
    effort   = thrust * thrust                          # [0, 1]
    drain    = effort * cfg.stamina_sprint_drain * beh.size
    recovery = (1.0 - effort) * beh.stamina_recovery
    creature.stamina = max(
        0.0,
        min(beh.stamina_capacity, creature.stamina + recovery - drain)
    )

    # ── Effective speed (size + stamina) ─────────────────────────────────────
    stamina_frac  = creature.stamina / max(beh.stamina_capacity, 1.0)
    stamina_scale = cfg.stamina_speed_floor + (1.0 - cfg.stamina_speed_floor) * stamina_frac
    effective_max_speed = cfg.max_speed * (beh.size ** -0.5) * stamina_scale

    # ── Heading ───────────────────────────────────────────────────────────────
    creature.angle += turn * cfg.max_turn_rate
    creature.angle %= (2 * math.pi)

    # ── Velocity ──────────────────────────────────────────────────────────────
    speed = thrust * effective_max_speed
    dx = math.cos(creature.angle) * speed
    dy = math.sin(creature.angle) * speed

    creature.vel[0] = creature.vel[0] * cfg.velocity_damping + dx * (1.0 - cfg.velocity_damping)
    creature.vel[1] = creature.vel[1] * cfg.velocity_damping + dy * (1.0 - cfg.velocity_damping)

    # Clamp to stamina-scaled cap
    speed_sq = creature.vel[0] ** 2 + creature.vel[1] ** 2
    if speed_sq > effective_max_speed ** 2:
        scale = effective_max_speed / math.sqrt(speed_sq)
        creature.vel[0] *= scale
        creature.vel[1] *= scale

    # ── Position ──────────────────────────────────────────────────────────────
    old_x, old_y = float(creature.pos[0]), float(creature.pos[1])
    creature.pos[0] += creature.vel[0]
    creature.pos[1] += creature.vel[1]

    # Toroidal wrap
    creature.pos[0] = creature.pos[0] % cfg.world_width
    creature.pos[1] = creature.pos[1] % cfg.world_height

    return old_x, old_y


def movement_energy_cost(creature: Creature, cfg: SimConfig = CONFIG) -> float:
    """Energy cost proportional to speed squared, scaled by sqrt(size)."""
    speed_sq = float(creature.vel[0] ** 2 + creature.vel[1] ** 2)
    size_factor = creature.genome.behavior.size ** 0.5
    return speed_sq * cfg.movement_cost_factor * size_factor
