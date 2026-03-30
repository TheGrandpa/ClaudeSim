"""
NEAT mutation operators.

  mutate_weights      — perturb or scramble connection weights
  add_connection      — add a new connection gene between two unconnected nodes
  add_node            — split a connection, inserting a hidden node
  disable_connection  — randomly disable an enabled connection
"""

from __future__ import annotations

import random
from typing import Optional

from config import SimConfig, CONFIG
from core.appearance import mutate_appearance
from core.behavior import mutate_behavior
from core.genome import (
    Genome, NodeGene, ConnectionGene, NodeType, INNOVATION_REGISTRY, InnovationRegistry
)


def mutate(
    genome: Genome,
    cfg: SimConfig = CONFIG,
    registry: InnovationRegistry = INNOVATION_REGISTRY,
) -> dict:
    """
    Apply all mutation operators to genome in-place.
    Returns a dict with keys 'structural' (bool), 'add_node' (bool), 'add_conn' (bool).
    """
    result = {"structural": False, "add_node": False, "add_conn": False}

    mutate_weights(genome, cfg)

    if random.random() < cfg.add_connection_rate:
        if add_connection(genome, cfg, registry):
            result["structural"] = True
            result["add_conn"] = True

    if random.random() < cfg.add_node_rate:
        if add_node(genome, cfg, registry):
            result["structural"] = True
            result["add_node"] = True

    if random.random() < cfg.disable_connection_rate:
        disable_connection(genome)

    mutate_appearance(genome.appearance)
    mutate_behavior(genome.behavior)

    return result


def mutate_weights(genome: Genome, cfg: SimConfig = CONFIG) -> None:
    for conn in genome.connections.values():
        if not conn.enabled:
            continue
        r = random.random()
        if r < cfg.weight_scramble_rate:
            conn.weight = random.uniform(-cfg.weight_init_range, cfg.weight_init_range)
        elif r < cfg.weight_perturb_rate:
            import numpy as np
            conn.weight += float(np.random.normal(0, cfg.weight_perturb_strength))
            conn.weight = max(-cfg.weight_init_range * 2, min(cfg.weight_init_range * 2, conn.weight))


def add_connection(
    genome: Genome,
    cfg: SimConfig = CONFIG,
    registry: InnovationRegistry = INNOVATION_REGISTRY,
) -> bool:
    """
    Pick two nodes with no existing enabled connection and add one.
    Returns True if successful.
    """
    node_ids = list(genome.nodes.keys())
    if len(node_ids) < 2:
        return False

    # Existing enabled connections set
    existing = {(c.in_node, c.out_node) for c in genome.connections.values() if c.enabled}

    # Try random pairs (give up after N attempts to avoid hot loop)
    for _ in range(20):
        in_id = random.choice(node_ids)
        out_id = random.choice(node_ids)
        if in_id == out_id:
            continue
        # Don't connect output -> input or output -> output (keeps it mostly feedforward)
        if genome.nodes[in_id].node_type == NodeType.OUTPUT:
            continue
        if genome.nodes[out_id].node_type == NodeType.INPUT:
            continue
        if (in_id, out_id) in existing:
            continue

        innov = registry.get_innovation(in_id, out_id)
        weight = random.uniform(-cfg.weight_init_range, cfg.weight_init_range)
        genome.connections[innov] = ConnectionGene(in_id, out_id, weight, True, innov)
        return True

    return False


def add_node(
    genome: Genome,
    cfg: SimConfig = CONFIG,
    registry: InnovationRegistry = INNOVATION_REGISTRY,
) -> bool:
    """
    Split an existing enabled connection by inserting a hidden node.
    The original connection is disabled; two new connections are added.
    Returns True if successful.
    """
    if len(genome.hidden_ids()) >= cfg.max_hidden_nodes:
        return False

    enabled = genome.enabled_connections()
    if not enabled:
        return False

    # Prefer splitting connections not already split (those with weight ≈ 1.0)
    conn = random.choice(enabled)
    conn.enabled = False

    new_node_id = registry.next_node_id()
    # Avoid id collision with existing nodes
    while new_node_id in genome.nodes:
        new_node_id = registry.next_node_id()

    genome.nodes[new_node_id] = NodeGene(new_node_id, NodeType.HIDDEN, "tanh")

    # in -> new_node  with weight 1.0  (preserves function)
    innov_a = registry.get_innovation(conn.in_node, new_node_id)
    genome.connections[innov_a] = ConnectionGene(
        conn.in_node, new_node_id, 1.0, True, innov_a
    )

    # new_node -> out  with original weight
    innov_b = registry.get_innovation(new_node_id, conn.out_node)
    genome.connections[innov_b] = ConnectionGene(
        new_node_id, conn.out_node, conn.weight, True, innov_b
    )

    return True


def disable_connection(genome: Genome) -> None:
    enabled = genome.enabled_connections()
    if enabled:
        random.choice(enabled).enabled = False
