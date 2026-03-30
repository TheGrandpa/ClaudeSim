"""
BehaviorGene — heritable behavioral parameters separate from the NEAT weight genome.

Traits:
  Sensing
    ray_length        how far vision rays reach (80–400)

  Action thresholds  (NN output must exceed these to trigger the action)
    eat_threshold         (0.2–0.9)
    attack_threshold      (0.3–0.95)
    reproduce_threshold   (0.5–0.95)
    flee_threshold        (0.2–0.9)

  Food drive
    food_priority     scales hunger-direction signal strength (0.1–3.0)
                      high value → strong directional pull toward food when hungry

All continuous traits mutate via Gaussian perturbation.
Crossover: each trait independently picked from parent A or B (uniform).
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class BehaviorGene:
    # Sensing
    ray_length:           float = 200.0

    # Action thresholds
    eat_threshold:        float = 0.50
    attack_threshold:     float = 0.70
    reproduce_threshold:  float = 0.80
    flee_threshold:       float = 0.60

    # Food-seeking drive
    food_priority:        float = 1.0

    # Reproduction mode: 0.0 = always asexual, 1.0 = always sexual
    # Values in between reproduce sexually with that probability each attempt.
    # Asexual offspring are clones with mutation; parent pays full reproduce_cost.
    sexual_bias:          float = 0.8

    # Diet: 0.0 = pure herbivore (energy from food),
    #        1.0 = pure carnivore (energy from attacking creatures).
    # Scales energy gained from each source continuously.
    carnivore_bias:       float = 0.05

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "ray_length":          self.ray_length,
            "eat_threshold":       self.eat_threshold,
            "attack_threshold":    self.attack_threshold,
            "reproduce_threshold": self.reproduce_threshold,
            "flee_threshold":      self.flee_threshold,
            "food_priority":       self.food_priority,
            "sexual_bias":         self.sexual_bias,
            "carnivore_bias":      self.carnivore_bias,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BehaviorGene":
        return cls(
            ray_length          = d.get("ray_length",          200.0),
            eat_threshold       = d.get("eat_threshold",        0.50),
            attack_threshold    = d.get("attack_threshold",     0.70),
            reproduce_threshold = d.get("reproduce_threshold",  0.80),
            flee_threshold      = d.get("flee_threshold",       0.60),
            food_priority       = d.get("food_priority",        1.0),
            sexual_bias         = d.get("sexual_bias",          0.8),
            carnivore_bias      = d.get("carnivore_bias",       0.05),
        )

    def copy(self) -> "BehaviorGene":
        return BehaviorGene(**self.to_dict())


# ── Factory ───────────────────────────────────────────────────────────────────

def random_behavior() -> BehaviorGene:
    return BehaviorGene(
        ray_length          = random.uniform(80.0, 400.0),
        eat_threshold       = random.uniform(0.2, 0.9),
        attack_threshold    = random.uniform(0.3, 0.95),
        reproduce_threshold = random.uniform(0.5, 0.95),
        flee_threshold      = random.uniform(0.2, 0.9),
        food_priority       = random.uniform(0.1, 3.0),
        sexual_bias         = random.uniform(0.0, 1.0),
        carnivore_bias      = random.uniform(0.0, 0.3),  # start mostly herbivorous
    )


# ── Crossover ─────────────────────────────────────────────────────────────────

def crossover_behavior(a: BehaviorGene, b: BehaviorGene) -> BehaviorGene:
    """Uniform crossover: each trait independently picked from either parent."""
    def pick(va, vb):
        return va if random.random() < 0.5 else vb

    return BehaviorGene(
        ray_length          = pick(a.ray_length,          b.ray_length),
        eat_threshold       = pick(a.eat_threshold,       b.eat_threshold),
        attack_threshold    = pick(a.attack_threshold,    b.attack_threshold),
        reproduce_threshold = pick(a.reproduce_threshold, b.reproduce_threshold),
        flee_threshold      = pick(a.flee_threshold,      b.flee_threshold),
        food_priority       = pick(a.food_priority,       b.food_priority),
        sexual_bias         = pick(a.sexual_bias,         b.sexual_bias),
        carnivore_bias      = pick(a.carnivore_bias,      b.carnivore_bias),
    )


# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate_behavior(beh: BehaviorGene, rate: float = 0.3) -> None:
    """Perturb behavior gene in-place."""
    import numpy as np

    def perturb(v: float, lo: float, hi: float, strength: float = 0.05) -> float:
        if random.random() < rate:
            v += float(np.random.normal(0, strength * (hi - lo)))
            return max(lo, min(hi, v))
        return v

    beh.ray_length          = perturb(beh.ray_length,          80.0, 400.0,  0.08)
    beh.eat_threshold       = perturb(beh.eat_threshold,        0.2,  0.9,   0.06)
    beh.attack_threshold    = perturb(beh.attack_threshold,     0.3,  0.95,  0.06)
    beh.reproduce_threshold = perturb(beh.reproduce_threshold,  0.5,  0.95,  0.06)
    beh.flee_threshold      = perturb(beh.flee_threshold,       0.2,  0.9,   0.06)
    beh.food_priority       = perturb(beh.food_priority,        0.1,  3.0,   0.08)
    beh.sexual_bias         = perturb(beh.sexual_bias,          0.0,  1.0,   0.08)
    beh.carnivore_bias      = perturb(beh.carnivore_bias,       0.0,  1.0,   0.06)
