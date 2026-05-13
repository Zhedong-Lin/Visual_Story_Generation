"""Typed graph adapters and helpers."""

from __future__ import annotations

from typing import Iterable, List, Tuple

from anime_pipeline_graph.domain.models import GraphEdge, GraphStep, SkillGraph


def edge_to_tuple(edge: tuple[str, str] | GraphEdge) -> tuple[str, str]:
    """Normalize edge shape to tuple(source, target)."""
    if isinstance(edge, GraphEdge):
        return edge.source, edge.target
    return edge


def edge_from_tuple(source: str, target: str, edge_type: str = "data") -> GraphEdge:
    """Build typed edge from tuple parts."""
    return GraphEdge(source=source, target=target, edge_type=edge_type)


def normalized_edges(graph: SkillGraph) -> List[tuple[str, str]]:
    """Return all edges as source/target tuples."""
    return [edge_to_tuple(e) for e in graph.edges]


def as_typed_edges(edges: Iterable[tuple[str, str]]) -> List[GraphEdge]:
    """Convert tuples to typed edges."""
    return [edge_from_tuple(src, dst) for src, dst in edges]


def with_typed_defaults(step: GraphStep) -> GraphStep:
    """Ensure a step has typed node defaults populated."""
    updates = {}
    if not step.skill_name:
        updates["skill_name"] = step.skill
    if not step.outputs_produced and step.outputs:
        updates["outputs_produced"] = list(step.outputs)
    if not step.inputs_required:
        updates["inputs_required"] = list(step.inputs.keys())
    return step.model_copy(update=updates) if updates else step


def to_typed_graph(graph: SkillGraph) -> SkillGraph:
    """Return a graph with typed nodes/edges populated while preserving compatibility."""
    typed_steps = [with_typed_defaults(s) for s in graph.steps]
    typed_edges = []
    for edge in graph.edges:
        src, dst = edge_to_tuple(edge)
        if isinstance(edge, GraphEdge):
            typed_edges.append(edge)
        else:
            typed_edges.append(GraphEdge(source=src, target=dst, edge_type="data"))
    return graph.model_copy(update={"steps": typed_steps, "edges": typed_edges})


def to_legacy_graph(graph: SkillGraph) -> SkillGraph:
    """Convert graph edges to tuple format for existing executor code paths."""
    return graph.model_copy(update={"edges": [edge_to_tuple(e) for e in graph.edges]})


def dedup_edges(edges: Iterable[tuple[str, str]]) -> List[tuple[str, str]]:
    """Deduplicate edges while preserving order."""
    seen: set[Tuple[str, str]] = set()
    out: List[tuple[str, str]] = []
    for edge in edges:
        if edge in seen:
            continue
        seen.add(edge)
        out.append(edge)
    return out

