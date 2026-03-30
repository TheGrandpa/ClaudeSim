"""
Syllable-based heritable naming system.

Each creature carries a list of syllables. At birth:
  - syllables are drawn from both parents (crossover)
  - structural NEAT mutations (add_node, add_connection) can append a new syllable
  - any syllable can randomly drift to a nearby variant (mutation)

Syllables are short phonetically plausible strings built from onset + nucleus
combinations so names sound natural when read aloud.
"""

import random
from typing import List, Optional

# ── Syllable pool ─────────────────────────────────────────────────────────────
# Organised as (onset, nucleus) pairs. Onset may be empty for vowel-start syllables.

_ONSETS = [
    "r", "v", "k", "m", "n", "th", "dr", "vel", "mor", "rath",
    "kir", "das", "ven", "sol", "thr", "eld", "cor", "ash", "bel",
    "fyr", "gal", "hex", "isk", "jor", "lun", "nox", "orr", "pyr",
    "qan", "sar", "tav", "ulm", "var", "wyr", "xen", "yon", "zel",
    "br", "cr", "fr", "gr", "pr", "str", "tr", "",
]

_NUCLEI = [
    "ath", "el", "in", "or", "un", "ys", "en", "al", "ix", "on",
    "ar", "ev", "im", "og", "ur", "an", "eth", "il", "os", "um",
    "ael", "ein", "oir", "uen", "aer", "ion", "ith", "oth", "iel",
]

# Pre-build a flat pool so random choice is O(1)
_SYLLABLE_POOL: List[str] = []
for _o in _ONSETS:
    for _n in _NUCLEI:
        s = _o + _n
        if 2 <= len(s) <= 6:
            _SYLLABLE_POOL.append(s)

# Deduplicate while preserving order
_seen: set = set()
_SYLLABLE_POOL_UNIQUE: List[str] = []
for _s in _SYLLABLE_POOL:
    if _s not in _seen:
        _seen.add(_s)
        _SYLLABLE_POOL_UNIQUE.append(_s)
_SYLLABLE_POOL = _SYLLABLE_POOL_UNIQUE


def random_syllable() -> str:
    return random.choice(_SYLLABLE_POOL)


def _mutate_syllable(syllable: str) -> str:
    """Return a nearby syllable (same index ± small offset in the pool)."""
    idx = _SYLLABLE_POOL.index(syllable) if syllable in _SYLLABLE_POOL else 0
    delta = random.randint(-8, 8)
    new_idx = max(0, min(len(_SYLLABLE_POOL) - 1, idx + delta))
    return _SYLLABLE_POOL[new_idx]


# ── Name class ────────────────────────────────────────────────────────────────

class CreatureName:
    """
    Immutable value object representing a creature's name.

    Attributes:
        syllables   list of syllable strings
        generation  generation depth (0 = primordial)
    """

    __slots__ = ("syllables", "generation")

    def __init__(self, syllables: List[str], generation: int = 0):
        self.syllables = list(syllables)
        self.generation = generation

    # ── Display ───────────────────────────────────────────────────────────────

    def short(self) -> str:
        """e.g. 'Rath-vel-kin'"""
        return "-".join(s.capitalize() for s in self.syllables)

    def full(self) -> str:
        """e.g. 'Rath-vel-kin  (Gen 7)'"""
        return f"{self.short()}  (Gen {self.generation})"

    def __repr__(self) -> str:
        return f"CreatureName({self.full()!r})"

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"syllables": self.syllables, "generation": self.generation}

    @classmethod
    def from_dict(cls, d: dict) -> "CreatureName":
        return cls(d["syllables"], d["generation"])


# ── Factory functions ─────────────────────────────────────────────────────────

def random_name(min_syllables: int = 2, max_syllables: int = 4) -> CreatureName:
    """Generate a completely random primordial name."""
    n = random.randint(min_syllables, max_syllables)
    return CreatureName([random_syllable() for _ in range(n)], generation=0)


def inherit_name(
    parent_a: CreatureName,
    parent_b: CreatureName,
    mutation_rate: float = 0.15,
    add_syllable: bool = False,
    min_syllables: int = 2,
    max_syllables: int = 4,
) -> CreatureName:
    """
    Produce a child name by combining syllables from two parents.

    Algorithm:
      1. Interleave both parent syllable lists (uniform crossover per position).
      2. Target length = avg of parent lengths, clamped to [min, max].
      3. Each syllable mutates independently with `mutation_rate`.
      4. If `add_syllable` is True (structural NEAT mutation occurred), append
         a new random syllable — this is how names grow over evolutionary time.
    """
    # Build candidate pool interleaving both parents
    max_len = max(len(parent_a.syllables), len(parent_b.syllables))
    a_padded = parent_a.syllables + [random_syllable()] * (max_len - len(parent_a.syllables))
    b_padded = parent_b.syllables + [random_syllable()] * (max_len - len(parent_b.syllables))

    target_len = round((len(parent_a.syllables) + len(parent_b.syllables)) / 2)
    target_len = max(min_syllables, min(max_syllables, target_len))

    syllables: List[str] = []
    for i in range(target_len):
        chosen = a_padded[i] if random.random() < 0.5 else b_padded[i]
        if random.random() < mutation_rate:
            chosen = _mutate_syllable(chosen)
        syllables.append(chosen)

    if add_syllable and len(syllables) < max_syllables:
        syllables.append(random_syllable())

    generation = max(parent_a.generation, parent_b.generation) + 1
    return CreatureName(syllables, generation)
