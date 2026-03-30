"""
AppearanceGene — heritable visual traits separate from the NEAT weight genome.

Traits:
  Color
    hue           primary hue  (0.0–1.0)
    saturation    color richness (0.55–1.0)
    brightness    color value  (0.55–1.0)
    belly_hue     secondary / belly color hue (0.0–1.0)
    belly_ratio   how much belly color shows (0.0–0.6)

  Body shape   (all are multipliers on the base creature radius)
    body_length   nose projection forward   (1.2–2.4)
    body_width    half-width at widest      (0.6–1.4)
    tail_length   tail projection backward  (0.5–1.2)
    tail_fork     how split the tail is     (0.0–0.7)

  Fins
    fin_count     0, 1, or 2 dorsal/ventral fins
    fin_length    fin projection sideways   (0.2–0.9)
    fin_position  where along body fins sit (0.0–1.0, 0=mid, 1=back)

All continuous traits mutate via Gaussian perturbation.
Discrete traits (fin_count) mutate via ±1 with low probability.
Crossover: each trait independently picked from parent A or B.
"""

from __future__ import annotations

import colorsys
import random
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AppearanceGene:
    # Color
    hue:          float = 0.5
    saturation:   float = 0.85
    brightness:   float = 0.90
    belly_hue:    float = 0.5
    belly_ratio:  float = 0.25

    # Body
    body_length:  float = 1.6
    body_width:   float = 0.95
    tail_length:  float = 0.8
    tail_fork:    float = 0.15

    # Fins
    fin_count:    int   = 1
    fin_length:   float = 0.45
    fin_position: float = 0.5

    # ── Derived colors ────────────────────────────────────────────────────────

    def primary_rgb(self, energy_ratio: float = 1.0) -> Tuple[int, int, int]:
        v = self.brightness * (0.35 + 0.65 * energy_ratio)
        r, g, b = colorsys.hsv_to_rgb(self.hue, self.saturation, min(v, 1.0))
        return (int(r * 255), int(g * 255), int(b * 255))

    def belly_rgb(self, energy_ratio: float = 1.0) -> Tuple[int, int, int]:
        v = self.brightness * 0.75 * (0.35 + 0.65 * energy_ratio)
        r, g, b = colorsys.hsv_to_rgb(self.belly_hue, self.saturation * 0.7, min(v, 1.0))
        return (int(r * 255), int(g * 255), int(b * 255))

    def outline_rgb(self) -> Tuple[int, int, int]:
        r, g, b = colorsys.hsv_to_rgb(self.hue, self.saturation * 0.4, 0.95)
        return (int(r * 255), int(g * 255), int(b * 255))

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "hue": self.hue, "saturation": self.saturation,
            "brightness": self.brightness, "belly_hue": self.belly_hue,
            "belly_ratio": self.belly_ratio, "body_length": self.body_length,
            "body_width": self.body_width, "tail_length": self.tail_length,
            "tail_fork": self.tail_fork, "fin_count": self.fin_count,
            "fin_length": self.fin_length, "fin_position": self.fin_position,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AppearanceGene":
        return cls(**d)

    def copy(self) -> "AppearanceGene":
        return AppearanceGene(**self.to_dict())


# ── Factory ───────────────────────────────────────────────────────────────────

def random_appearance() -> AppearanceGene:
    return AppearanceGene(
        hue          = random.random(),
        saturation   = random.uniform(0.60, 1.0),
        brightness   = random.uniform(0.65, 1.0),
        belly_hue    = random.random(),
        belly_ratio  = random.uniform(0.10, 0.45),
        body_length  = random.uniform(1.2, 2.4),
        body_width   = random.uniform(0.6, 1.4),
        tail_length  = random.uniform(0.5, 1.2),
        tail_fork    = random.uniform(0.0, 0.7),
        fin_count    = random.randint(0, 2),
        fin_length   = random.uniform(0.2, 0.9),
        fin_position = random.uniform(0.1, 0.9),
    )


# ── Crossover ─────────────────────────────────────────────────────────────────

def crossover_appearance(a: AppearanceGene, b: AppearanceGene) -> AppearanceGene:
    """Uniform crossover: each trait independently picked from either parent."""
    def pick(va, vb):
        return va if random.random() < 0.5 else vb

    return AppearanceGene(
        hue          = pick(a.hue, b.hue),
        saturation   = pick(a.saturation, b.saturation),
        brightness   = pick(a.brightness, b.brightness),
        belly_hue    = pick(a.belly_hue, b.belly_hue),
        belly_ratio  = pick(a.belly_ratio, b.belly_ratio),
        body_length  = pick(a.body_length, b.body_length),
        body_width   = pick(a.body_width, b.body_width),
        tail_length  = pick(a.tail_length, b.tail_length),
        tail_fork    = pick(a.tail_fork, b.tail_fork),
        fin_count    = pick(a.fin_count, b.fin_count),
        fin_length   = pick(a.fin_length, b.fin_length),
        fin_position = pick(a.fin_position, b.fin_position),
    )


# ── Mutation ──────────────────────────────────────────────────────────────────

_PERTURB_STRENGTH = 0.06

def mutate_appearance(app: AppearanceGene, rate: float = 0.3) -> None:
    """Perturb appearance gene in-place."""
    import numpy as np

    def perturb_hue(v: float) -> float:
        if random.random() < rate:
            return (v + np.random.normal(0, _PERTURB_STRENGTH)) % 1.0
        return v

    def perturb(v: float, lo: float, hi: float) -> float:
        if random.random() < rate:
            v += float(np.random.normal(0, _PERTURB_STRENGTH * (hi - lo)))
            return max(lo, min(hi, v))
        return v

    app.hue          = perturb_hue(app.hue)
    app.saturation   = perturb(app.saturation, 0.55, 1.0)
    app.brightness   = perturb(app.brightness, 0.55, 1.0)
    app.belly_hue    = perturb_hue(app.belly_hue)
    app.belly_ratio  = perturb(app.belly_ratio, 0.0, 0.6)
    app.body_length  = perturb(app.body_length, 1.1, 2.5)
    app.body_width   = perturb(app.body_width, 0.5, 1.5)
    app.tail_length  = perturb(app.tail_length, 0.4, 1.3)
    app.tail_fork    = perturb(app.tail_fork, 0.0, 0.75)
    app.fin_length   = perturb(app.fin_length, 0.15, 1.0)
    app.fin_position = perturb(app.fin_position, 0.05, 0.95)

    # Discrete: fin_count can shift ±1
    if random.random() < rate * 0.3:
        app.fin_count = max(0, min(2, app.fin_count + random.choice([-1, 1])))
