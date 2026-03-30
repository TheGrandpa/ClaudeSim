"""
NEAT genome representation.

NodeGene     — an input, hidden, or output node
ConnectionGene — a weighted, directed connection between two nodes
InnovationRegistry — global singleton that assigns stable innovation numbers
Genome       — a creature's complete genetic description
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from core.appearance import AppearanceGene, random_appearance
from core.behavior import BehaviorGene, random_behavior


# ── Node types ────────────────────────────────────────────────────────────────

class NodeType(Enum):
    INPUT = auto()
    HIDDEN = auto()
    OUTPUT = auto()


@dataclass
class NodeGene:
    node_id: int
    node_type: NodeType
    # Activation: 'tanh', 'relu', 'sigmoid', 'linear'
    activation: str = "tanh"

    def copy(self) -> "NodeGene":
        return NodeGene(self.node_id, self.node_type, self.activation)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.name,
            "activation": self.activation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NodeGene":
        return cls(d["node_id"], NodeType[d["node_type"]], d["activation"])


@dataclass
class ConnectionGene:
    in_node: int
    out_node: int
    weight: float
    enabled: bool
    innovation: int

    def copy(self) -> "ConnectionGene":
        return ConnectionGene(
            self.in_node, self.out_node, self.weight, self.enabled, self.innovation
        )

    def to_dict(self) -> dict:
        return {
            "in_node": self.in_node,
            "out_node": self.out_node,
            "weight": self.weight,
            "enabled": self.enabled,
            "innovation": self.innovation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConnectionGene":
        return cls(d["in_node"], d["out_node"], d["weight"], d["enabled"], d["innovation"])


# ── Innovation registry ───────────────────────────────────────────────────────

class InnovationRegistry:
    """
    Global singleton.  Assigns a stable innovation number to each unique
    (in_node, out_node) structural event.  Two creatures that independently
    discover the same connection in the same generation share one number,
    enabling correct crossover alignment.

    Reset per-generation via `new_generation()` so within-generation
    duplicates collapse, but cross-generation events keep their numbers.
    """

    def __init__(self) -> None:
        self._global_counter: int = 0
        self._node_counter: int = 0
        # Maps (in_node, out_node) -> innovation number, reset each generation
        self._generation_cache: Dict[Tuple[int, int], int] = {}

    def new_generation(self) -> None:
        self._generation_cache.clear()

    def get_innovation(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self._generation_cache:
            self._global_counter += 1
            self._generation_cache[key] = self._global_counter
        return self._generation_cache[key]

    def next_node_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def set_counters(self, innovation: int, node: int) -> None:
        """Used when loading a saved simulation."""
        self._global_counter = max(self._global_counter, innovation)
        self._node_counter = max(self._node_counter, node)

    def to_dict(self) -> dict:
        return {
            "global_counter": self._global_counter,
            "node_counter": self._node_counter,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InnovationRegistry":
        reg = cls()
        reg._global_counter = d["global_counter"]
        reg._node_counter = d["node_counter"]
        return reg


# Module-level singleton
INNOVATION_REGISTRY = InnovationRegistry()


# ── Genome ────────────────────────────────────────────────────────────────────

class Genome:
    """
    A creature's NEAT genome.

    nodes       — dict node_id -> NodeGene
    connections — dict innovation -> ConnectionGene
    """

    def __init__(self) -> None:
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}
        self.lineage_id: int = 0    # stable across mutation, set at birth
        self.appearance: AppearanceGene = random_appearance()
        self.behavior: BehaviorGene = random_behavior()

    # ── Accessors ─────────────────────────────────────────────────────────────

    def input_ids(self) -> List[int]:
        return [n.node_id for n in self.nodes.values() if n.node_type == NodeType.INPUT]

    def output_ids(self) -> List[int]:
        return [n.node_id for n in self.nodes.values() if n.node_type == NodeType.OUTPUT]

    def hidden_ids(self) -> List[int]:
        return [n.node_id for n in self.nodes.values() if n.node_type == NodeType.HIDDEN]

    def enabled_connections(self) -> List[ConnectionGene]:
        return [c for c in self.connections.values() if c.enabled]

    def weight_count(self) -> int:
        return len(self.enabled_connections())

    # ── Compatibility distance (for speciation) ───────────────────────────────

    def compatibility_distance(
        self, other: "Genome", c1: float, c2: float, c3: float
    ) -> float:
        """
        δ = c1·E/N + c2·D/N + c3·W̄
        E = excess genes, D = disjoint genes, W̄ = avg weight diff of matching genes
        N = max(len(self.connections), len(other.connections)), min-clamped to 1
        """
        keys_a = set(self.connections.keys())
        keys_b = set(other.connections.keys())

        if not keys_a and not keys_b:
            return 0.0

        max_innov_a = max(keys_a, default=0)
        max_innov_b = max(keys_b, default=0)

        matching_weight_diffs: List[float] = []
        disjoint = 0
        excess = 0

        all_keys = keys_a | keys_b
        for k in all_keys:
            in_a = k in keys_a
            in_b = k in keys_b
            if in_a and in_b:
                matching_weight_diffs.append(
                    abs(self.connections[k].weight - other.connections[k].weight)
                )
            else:
                # Determine disjoint vs excess
                if in_a:
                    if k > max_innov_b:
                        excess += 1
                    else:
                        disjoint += 1
                else:
                    if k > max_innov_a:
                        excess += 1
                    else:
                        disjoint += 1

        # Cap N at 20: prevents large genomes from diluting structural differences
        # to near-zero (original NEAT paper recommends N=1 for small genomes).
        n = min(max(len(keys_a), len(keys_b), 1), 20)
        w_bar = sum(matching_weight_diffs) / len(matching_weight_diffs) if matching_weight_diffs else 0.0

        return c1 * excess / n + c2 * disjoint / n + c3 * w_bar

    # ── Copy ──────────────────────────────────────────────────────────────────

    def copy(self) -> "Genome":
        g = Genome()
        g.lineage_id = self.lineage_id
        g.nodes = {nid: n.copy() for nid, n in self.nodes.items()}
        g.connections = {inn: c.copy() for inn, c in self.connections.items()}
        g.appearance = self.appearance.copy()
        g.behavior = self.behavior.copy()
        return g

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "lineage_id": self.lineage_id,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "connections": [c.to_dict() for c in self.connections.values()],
            "appearance": self.appearance.to_dict(),
            "behavior": self.behavior.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        g = cls()
        g.lineage_id = d["lineage_id"]
        for nd in d["nodes"]:
            node = NodeGene.from_dict(nd)
            g.nodes[node.node_id] = node
        for cd in d["connections"]:
            conn = ConnectionGene.from_dict(cd)
            g.connections[conn.innovation] = conn
        if "appearance" in d:
            g.appearance = AppearanceGene.from_dict(d["appearance"])
        if "behavior" in d:
            g.behavior = BehaviorGene.from_dict(d["behavior"])
        return g


# ── Minimal genome factory ────────────────────────────────────────────────────

def make_minimal_genome(
    n_inputs: int,
    n_outputs: int,
    registry: InnovationRegistry = INNOVATION_REGISTRY,
    weight_range: float = 2.0,
) -> Genome:
    """
    Create a minimal NEAT genome: all inputs directly connected to all outputs,
    no hidden nodes.  This is the standard NEAT starting point.
    """
    g = Genome()

    # Reserve node ids: 1..n_inputs for inputs, n_inputs+1..n_inputs+n_outputs for outputs
    for i in range(1, n_inputs + 1):
        g.nodes[i] = NodeGene(i, NodeType.INPUT, "linear")

    for i in range(n_inputs + 1, n_inputs + n_outputs + 1):
        g.nodes[i] = NodeGene(i, NodeType.OUTPUT, "tanh")

    # Register these node ids so future hidden nodes don't collide
    registry.set_counters(0, n_inputs + n_outputs)

    # Connect every input to every output
    for in_id in range(1, n_inputs + 1):
        for out_id in range(n_inputs + 1, n_inputs + n_outputs + 1):
            innov = registry.get_innovation(in_id, out_id)
            weight = random.uniform(-weight_range, weight_range)
            g.connections[innov] = ConnectionGene(in_id, out_id, weight, True, innov)

    return g
