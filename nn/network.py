"""
NEAT neural network — graph-based forward pass.

Each creature has its own NeuralNetwork instance built from its Genome.
Since topologies differ between creatures, we cannot batch across all
creatures with a single matmul.  Instead we evaluate each network via a
cached topological sort of its connection graph.

The sort is recomputed only when the genome structure changes (add_node /
add_connection mutation).  Between structural changes, forward() is just
numpy dot products along the sorted activation order.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from core.genome import Genome, NodeType


# ── Activation functions ──────────────────────────────────────────────────────

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _linear(x: np.ndarray) -> np.ndarray:
    return x

_ACTIVATIONS = {
    "tanh": _tanh,
    "relu": _relu,
    "sigmoid": _sigmoid,
    "linear": _linear,
}


# ── Network ───────────────────────────────────────────────────────────────────

class NeuralNetwork:
    """
    Evaluates a NEAT genome as a directed graph.

    Attributes:
        genome          the source genome (not copied — mutate genome then call rebuild())
        _order          topologically sorted list of node ids (inputs first, outputs last)
        _dirty          True when structural changes require a resort
    """

    def __init__(self, genome: Genome) -> None:
        self.genome = genome
        self._order: List[int] = []
        self._dirty: bool = True

    def mark_dirty(self) -> None:
        """Call after any structural mutation to trigger a re-sort on next forward()."""
        self._dirty = True

    def _build_order(self) -> None:
        """
        Topological sort using Kahn's algorithm (BFS from nodes with in-degree 0).
        Input nodes are always first; output nodes always last.
        Recurrent connections (cycles) are detected and skipped gracefully.
        """
        nodes = self.genome.nodes
        enabled = self.genome.enabled_connections()

        # Build adjacency and in-degree, but only for enabled connections
        in_degree: Dict[int, int] = {nid: 0 for nid in nodes}
        successors: Dict[int, List[int]] = {nid: [] for nid in nodes}

        for conn in enabled:
            if conn.in_node == conn.out_node:
                continue  # skip self-loops
            if conn.in_node in nodes and conn.out_node in nodes:
                in_degree[conn.out_node] += 1
                successors[conn.in_node].append(conn.out_node)

        # Force inputs first
        queue = [nid for nid in nodes if nodes[nid].node_type == NodeType.INPUT]
        visited = set(queue)
        order = list(queue)

        # BFS
        i = 0
        while i < len(order):
            current = order[i]
            for nxt in successors[current]:
                if nxt in visited:
                    continue
                in_degree[nxt] -= 1
                if in_degree[nxt] <= 0:
                    order.append(nxt)
                    visited.add(nxt)
            i += 1

        # Any nodes not reached (cycle remainder) appended at end
        for nid in nodes:
            if nid not in visited:
                order.append(nid)

        self._order = order
        self._dirty = False

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run a forward pass.

        Parameters:
            inputs   1-D array of length == number of input nodes

        Returns:
            1-D array of length == number of output nodes (tanh-activated)
        """
        if self._dirty:
            self._build_order()

        nodes = self.genome.nodes
        activations: Dict[int, float] = {}

        # Load inputs
        input_ids = [nid for nid in self._order if nodes[nid].node_type == NodeType.INPUT]
        for idx, nid in enumerate(input_ids):
            activations[nid] = float(inputs[idx]) if idx < len(inputs) else 0.0

        # Build lookup: out_node -> list of (in_node, weight)
        incoming: Dict[int, List[tuple]] = {nid: [] for nid in nodes}
        for conn in self.genome.enabled_connections():
            if conn.in_node in nodes and conn.out_node in nodes:
                incoming[conn.out_node].append((conn.in_node, conn.weight))

        # Evaluate in topological order
        for nid in self._order:
            if nodes[nid].node_type == NodeType.INPUT:
                continue  # already loaded
            raw = sum(
                activations.get(src, 0.0) * w for src, w in incoming[nid]
            )
            act_fn = _ACTIVATIONS.get(nodes[nid].activation, _tanh)
            activations[nid] = float(act_fn(np.array([raw]))[0])

        # Collect outputs in stable order
        output_ids = sorted(
            [nid for nid in nodes if nodes[nid].node_type == NodeType.OUTPUT]
        )
        return np.array([activations.get(nid, 0.0) for nid in output_ids], dtype=np.float32)

    def weight_count(self) -> int:
        return self.genome.weight_count()
