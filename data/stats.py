"""
StatsCollector — rolling statistics over the simulation.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.population import Population
    from simulation.loop import SimulationLoop


class StatsCollector:
    def __init__(self, window: int = 300) -> None:
        self.window = window
        self.population_history: Deque[int]   = deque(maxlen=window)
        self.avg_energy_history: Deque[float] = deque(maxlen=window)
        self.avg_age_history: Deque[float]    = deque(maxlen=window)
        self.species_history: Deque[int]      = deque(maxlen=window)

    def record(self, population: "Population", loop: "SimulationLoop") -> None:
        creatures = population.creatures
        if not creatures:
            self.population_history.append(0)
            self.avg_energy_history.append(0.0)
            self.avg_age_history.append(0.0)
            return

        self.population_history.append(len(creatures))
        self.avg_energy_history.append(sum(c.energy for c in creatures) / len(creatures))
        self.avg_age_history.append(sum(c.age for c in creatures) / len(creatures))
        self.species_history.append(loop.speciator.species_count())

    def snapshot(self) -> dict:
        def avg(d): return sum(d) / len(d) if d else 0.0
        return {
            "avg_population": avg(self.population_history),
            "avg_energy":     avg(self.avg_energy_history),
            "avg_age":        avg(self.avg_age_history),
            "avg_species":    avg(self.species_history),
        }
