"""Planning-time models for candidate graphs, violations and scores."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from anime_pipeline_graph.domain.models import SkillGraph, TaskSpec


class Violation(BaseModel):
    """Constraint violation record."""

    code: str
    message: str
    severity: str = "error"
    related_nodes: List[str] = Field(default_factory=list)
    suggested_repairs: List[Dict[str, Any]] = Field(default_factory=list)


class GraphScore(BaseModel):
    """Score breakdown for one candidate graph."""

    total_score: float
    coverage_score: float
    validity_score: float
    prior_score: float
    cost_score: float
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class CandidateGraph(BaseModel):
    """One candidate graph with provenance and diagnostics."""

    graph: SkillGraph
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    violations: List[Violation] = Field(default_factory=list)
    score: GraphScore | None = None


class CandidateGraphPlan(BaseModel):
    """Planner output that contains multiple candidate graphs."""

    task_spec: TaskSpec
    candidates: List[CandidateGraph]

