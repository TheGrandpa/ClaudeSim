"""
NEAT crossover — produce a child genome from two parents.

Gene alignment uses innovation numbers:
  matching genes   → randomly inherit from either parent
  disjoint/excess  → inherit from the more fit parent
                     (in this sim, fitness = current energy)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from core.appearance import crossover_appearance
from core.behavior import crossover_behavior
from core.genome import Genome, NodeGene, ConnectionGene, NodeType

if TYPE_CHECKING:
    from core.creature import Creature


def crossover(parent_a: "Creature", parent_b: "Creature") -> Genome:
    """
    Produce a child Genome.  The more energetic parent is treated as the
    'fitter' parent for disjoint/excess gene inheritance.
    """
    if parent_a.energy >= parent_b.energy:
        fit, weak = parent_a.genome, parent_b.genome
    else:
        fit, weak = parent_b.genome, parent_a.genome

    child = Genome()
    child.lineage_id = fit.lineage_id   # child inherits lineage from fitter parent

    # ── Connections ───────────────────────────────────────────────────────────
    all_innovations = set(fit.connections.keys()) | set(weak.connections.keys())

    for innov in all_innovations:
        in_fit  = innov in fit.connections
        in_weak = innov in weak.connections

        if in_fit and in_weak:
            # Matching gene — pick randomly
            source = fit.connections[innov] if random.random() < 0.5 else weak.connections[innov]
            gene = source.copy()
            # If either parent has it disabled, 75% chance child also disables it
            if not fit.connections[innov].enabled or not weak.connections[innov].enabled:
                if random.random() < 0.75:
                    gene.enabled = False
            child.connections[innov] = gene

        elif in_fit:
            # Excess/disjoint from fitter parent — always inherit
            child.connections[innov] = fit.connections[innov].copy()

        # Genes only in weak parent are not inherited (standard NEAT rule)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    # Include all nodes referenced by child connections, plus all input/output nodes
    referenced_ids = set()
    for conn in child.connections.values():
        referenced_ids.add(conn.in_node)
        referenced_ids.add(conn.out_node)

    # Always include input and output nodes from fitter parent
    for nid, node in fit.nodes.items():
        if node.node_type in (NodeType.INPUT, NodeType.OUTPUT):
            child.nodes[nid] = node.copy()
            referenced_ids.discard(nid)

    # Add any remaining referenced hidden nodes
    for nid in referenced_ids:
        if nid in fit.nodes:
            child.nodes[nid] = fit.nodes[nid].copy()
        elif nid in weak.nodes:
            child.nodes[nid] = weak.nodes[nid].copy()

    # ── Appearance ────────────────────────────────────────────────────────────
    child.appearance = crossover_appearance(fit.appearance, weak.appearance)

    # ── Behavior ──────────────────────────────────────────────────────────────
    child.behavior = crossover_behavior(fit.behavior, weak.behavior)

    return child
