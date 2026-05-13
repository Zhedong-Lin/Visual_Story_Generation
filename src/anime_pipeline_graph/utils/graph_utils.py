"""Graph helpers."""

from __future__ import annotations

from typing import List

import networkx as nx

from anime_pipeline_graph.domain.models import SkillGraph
from anime_pipeline_graph.planner.typed_graph import normalized_edges


def topological_steps(graph: SkillGraph) -> List[str]:
    """Return topologically sorted step ids."""
    g = nx.DiGraph()
    valid_nodes = set()
    for step in graph.steps:
        g.add_node(step.step_id)
        valid_nodes.add(step.step_id)
    for src, dst in normalized_edges(graph):
        if src in valid_nodes and dst in valid_nodes:
            g.add_edge(src, dst)
    return list(nx.topological_sort(g))
