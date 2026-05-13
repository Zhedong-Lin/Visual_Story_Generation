"""Graph repair as lightweight best-first edit search."""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from anime_pipeline_graph.domain.models import GraphStep, SkillGraph, TaskSpec
from anime_pipeline_graph.planner.constraints import ConstraintValidator
from anime_pipeline_graph.planner.planning_models import Violation
from anime_pipeline_graph.planner.skill_library import SkillLibrary
from anime_pipeline_graph.planner.typed_graph import dedup_edges, normalized_edges, to_typed_graph


class GraphEditOperator(ABC):
    """Abstract graph edit operation."""

    name: str
    cost: float = 1.0

    @abstractmethod
    def apply(self, graph: SkillGraph) -> SkillGraph:
        """Return edited graph."""


@dataclass
class AddNode(GraphEditOperator):
    """Add a node by skill name."""

    skill: str
    node_id: str | None = None
    cost: float = 1.2
    name: str = "AddNode"

    def apply(self, graph: SkillGraph) -> SkillGraph:
        g = to_typed_graph(graph)
        existing = {s.step_id for s in g.steps}
        base = self.node_id or f"{self.skill}_repair"
        node_id = base
        idx = 1
        while node_id in existing:
            node_id = f"{base}_{idx}"
            idx += 1

        outputs = ["judge_report"] if self.skill == "judge" else []
        g.steps.append(
            GraphStep(
                step_id=node_id,
                skill=self.skill,
                skill_name=self.skill,
                skill_type="runtime_skill",
                frame_scope="all",
                inputs_required=[],
                outputs_produced=outputs,
                optional=False,
                inputs={},
                params={},
                outputs=outputs,
            )
        )
        return g


@dataclass
class RemoveNode(GraphEditOperator):
    """Remove one node and related edges."""

    node_id: str
    cost: float = 1.0
    name: str = "RemoveNode"

    def apply(self, graph: SkillGraph) -> SkillGraph:
        g = to_typed_graph(graph)
        g.steps = [s for s in g.steps if s.step_id != self.node_id]
        kept = {s.step_id for s in g.steps}
        g.edges = [(s, t) for s, t in normalized_edges(g) if s in kept and t in kept]
        return g


@dataclass
class AddEdge(GraphEditOperator):
    """Add one directed edge."""

    source: str
    target: str
    cost: float = 0.6
    name: str = "AddEdge"

    def apply(self, graph: SkillGraph) -> SkillGraph:
        g = to_typed_graph(graph)
        edges = normalized_edges(g)
        edges.append((self.source, self.target))
        g.edges = dedup_edges(edges)
        return g


@dataclass
class RemoveEdge(GraphEditOperator):
    """Remove one directed edge."""

    source: str
    target: str
    cost: float = 0.5
    name: str = "RemoveEdge"

    def apply(self, graph: SkillGraph) -> SkillGraph:
        g = to_typed_graph(graph)
        g.edges = [(s, t) for s, t in normalized_edges(g) if not (s == self.source and t == self.target)]
        return g


@dataclass
class ReplaceNode(GraphEditOperator):
    """Replace one node skill while keeping node id/edges."""

    node_id: str
    skill: str
    cost: float = 1.3
    name: str = "ReplaceNode"

    def apply(self, graph: SkillGraph) -> SkillGraph:
        g = to_typed_graph(graph)
        out = []
        for s in g.steps:
            if s.step_id == self.node_id:
                out.append(s.model_copy(update={"skill": self.skill, "skill_name": self.skill}))
            else:
                out.append(s)
        g.steps = out
        return g


@dataclass
class InsertMotif(GraphEditOperator):
    """Insert a motif chain if missing."""

    steps: List[str]
    cost: float = 2.0
    name: str = "InsertMotif"

    def apply(self, graph: SkillGraph) -> SkillGraph:
        g = to_typed_graph(graph)
        existing_skills = {s.skill for s in g.steps}
        new_nodes = []
        for idx, skill in enumerate(self.steps, start=1):
            if skill in existing_skills:
                continue
            nid = f"{skill}_motif_{idx}"
            new_nodes.append(
                GraphStep(
                    step_id=nid,
                    skill=skill,
                    skill_name=skill,
                    skill_type="runtime_skill",
                    frame_scope="all",
                    inputs_required=[],
                    outputs_produced=[],
                    optional=False,
                    inputs={},
                    params={},
                    outputs=[],
                )
            )
        if not new_nodes:
            return g
        g.steps.extend(new_nodes)
        order = [n.step_id for n in new_nodes]
        edges = normalized_edges(g)
        for i in range(len(order) - 1):
            edges.append((order[i], order[i + 1]))
        g.edges = dedup_edges(edges)
        return g


class BestFirstGraphRepair:
    """Best-first search over graph edits to satisfy constraints."""

    def __init__(self, validator: ConstraintValidator, beam_width: int = 5, max_iters: int = 30) -> None:
        self.validator = validator
        self.beam_width = beam_width
        self.max_iters = max_iters

    def repair(
        self,
        graph: SkillGraph,
        task_spec: TaskSpec,
        skill_library: SkillLibrary,
        initial_violations: List[Violation] | None = None,
    ) -> tuple[SkillGraph, List[Violation], Dict[str, float]]:
        """Search edited graphs and return best valid-ish candidate."""
        start = to_typed_graph(graph)
        v0 = initial_violations if initial_violations is not None else self.validator.validate(start, task_spec, skill_library)

        best_graph = start
        best_v = v0
        best_cost = self._objective(start, v0, edit_cost=0.0)

        pq: List[Tuple[float, int, SkillGraph, float, int]] = []
        # (objective, tie, graph, edit_cost, depth)
        counter = 0
        heapq.heappush(pq, (best_cost, counter, start, 0.0, 0))
        seen = {self._graph_signature(start)}

        for _ in range(self.max_iters):
            if not pq:
                break
            _, _, curr, curr_edit_cost, depth = heapq.heappop(pq)
            curr_v = self.validator.validate(curr, task_spec, skill_library)
            curr_obj = self._objective(curr, curr_v, curr_edit_cost)
            if curr_obj < best_cost:
                best_graph, best_v, best_cost = curr, curr_v, curr_obj
            if not curr_v:
                best_graph, best_v, best_cost = curr, curr_v, curr_obj
                break
            if depth >= 4:
                continue

            ops = self._propose_ops(curr, curr_v, skill_library)
            frontier = []
            for op in ops:
                nxt = op.apply(curr)
                sig = self._graph_signature(nxt)
                if sig in seen:
                    continue
                seen.add(sig)
                nxt_edit_cost = curr_edit_cost + op.cost
                nxt_v = self.validator.validate(nxt, task_spec, skill_library)
                obj = self._objective(nxt, nxt_v, nxt_edit_cost)
                counter += 1
                frontier.append((obj, counter, nxt, nxt_edit_cost, depth + 1))

            frontier.sort(key=lambda x: x[0])
            for item in frontier[: self.beam_width]:
                heapq.heappush(pq, item)

        stats = {
            "best_objective": best_cost,
            "remaining_violations": float(len(best_v)),
            "beam_width": float(self.beam_width),
        }
        return best_graph, best_v, stats

    def _objective(self, graph: SkillGraph, violations: List[Violation], edit_cost: float) -> float:
        sev_penalty = 0.0
        for v in violations:
            if v.severity == "error":
                sev_penalty += 3.0
            elif v.severity == "warning":
                sev_penalty += 1.2
            else:
                sev_penalty += 0.5
        complexity = len(graph.steps) * 0.08 + len(graph.edges) * 0.04
        return sev_penalty + edit_cost + complexity

    def _graph_signature(self, graph: SkillGraph) -> str:
        skills = ",".join(sorted(f"{s.step_id}:{s.skill}" for s in graph.steps))
        edges = ",".join(sorted(f"{a}->{b}" for a, b in normalized_edges(graph)))
        return skills + "|" + edges

    def _propose_ops(
        self,
        graph: SkillGraph,
        violations: List[Violation],
        skill_library: SkillLibrary,
    ) -> List[GraphEditOperator]:
        ops: List[GraphEditOperator] = []
        steps = {s.step_id: s for s in graph.steps}
        sinks = self._sink_nodes(graph)

        for v in violations:
            if v.code == "must_have_judge":
                ops.append(AddNode(skill="judge", node_id="judge_autofix"))
                for s in sinks:
                    ops.append(AddEdge(source=s, target="judge_autofix"))

            elif v.code == "multiframe_needs_generation":
                ops.append(AddNode(skill="prompt_pack_builder", node_id="prompt_pack_builder_autofix"))
                ops.append(AddNode(skill="base_generation", node_id="base_generation_autofix"))
                ops.append(AddEdge(source="prompt_pack_builder_autofix", target="base_generation_autofix"))

            elif v.code == "multiframe_needs_story_decompose":
                ops.append(AddNode(skill="story_decompose", node_id="story_decompose_autofix"))
                for s in graph.steps:
                    if s.skill in {
                        "character_bind",
                        "scene_condition",
                        "transition_plan",
                        "shot_plan",
                        "pose_plan",
                        "expression_plan",
                        "continuity_state_tracker",
                        "visual_style_plan",
                        "prompt_pack_builder",
                    }:
                        ops.append(AddEdge(source="story_decompose_autofix", target=s.step_id))

            elif v.code == "prompt_builder_before_generation":
                for n in v.related_nodes:
                    if n in steps:
                        ops.append(AddNode(skill="prompt_pack_builder", node_id="prompt_pack_builder_autofix"))
                        ops.append(AddEdge(source="prompt_pack_builder_autofix", target=n))

            elif v.code == "graph_must_be_dag":
                cyc_nodes = v.related_nodes
                if len(cyc_nodes) >= 2:
                    ops.append(RemoveEdge(source=cyc_nodes[-2], target=cyc_nodes[-1]))

            elif v.code == "type_compatible_edges":
                for rep in v.suggested_repairs:
                    if rep.get("op") == "RemoveEdge":
                        src = rep.get("source")
                        dst = rep.get("target")
                        if src and dst:
                            ops.append(RemoveEdge(source=src, target=dst))

            elif v.code == "no_orphan_critical_nodes":
                for node_id in v.related_nodes:
                    for src in sinks:
                        if src != node_id:
                            ops.append(AddEdge(source=src, target=node_id))

            elif v.code == "preconditions_satisfied":
                for node_id in v.related_nodes:
                    if node_id in steps and steps[node_id].skill == "base_generation":
                        ops.append(AddNode(skill="prompt_pack_builder", node_id="prompt_pack_builder_autofix"))
                        ops.append(AddEdge(source="prompt_pack_builder_autofix", target=node_id))

            # Skill-card driven second-layer repair hints.
            for node_id in v.related_nodes:
                if node_id not in steps:
                    continue
                skill = steps[node_id].skill
                failure_modes = self._violation_to_failure_modes(v.code)
                for spec in skill_library.get_repairs_for_skill(skill, failure_modes):
                    op = self._op_from_spec(spec, node_id=node_id, sinks=sinks)
                    if op is not None:
                        ops.append(op)

        # Generic minimal fallback edits.
        if not ops:
            ops.append(AddNode(skill="judge", node_id="judge_autofix"))
        return self._dedup_ops(ops)

    def _sink_nodes(self, graph: SkillGraph) -> List[str]:
        outgoing = {s.step_id: 0 for s in graph.steps}
        for src, _ in normalized_edges(graph):
            if src in outgoing:
                outgoing[src] += 1
        sinks = [nid for nid, out in outgoing.items() if out == 0]
        return sinks or list(outgoing.keys())[:1]

    def _violation_to_failure_modes(self, code: str) -> List[str]:
        mapping = {
            "preconditions_satisfied": ["missing_preconditions", "missing_conditioning", "prompt_underfit"],
            "type_compatible_edges": ["edge_compatibility_error"],
            "no_orphan_critical_nodes": ["orphan_critical_node"],
            "prompt_builder_before_generation": ["missing_prompt_builder"],
            "multiframe_needs_generation": ["missing_generation_path"],
            "multiframe_needs_story_decompose": ["missing_story_decompose"],
            "must_have_judge": ["missing_judge"],
            "graph_must_be_dag": ["graph_cycle"],
        }
        return mapping.get(code, [])

    def _op_from_spec(self, spec: Dict[str, Any], node_id: str, sinks: List[str]) -> GraphEditOperator | None:
        op = str(spec.get("op", "")).strip()
        if op == "AddNode":
            skill = str(spec.get("skill", "")).strip()
            if not skill:
                return None
            return AddNode(skill=skill, node_id=spec.get("node_id"))
        if op == "RemoveNode":
            target = str(spec.get("node_id", node_id)).strip()
            return RemoveNode(node_id=target)
        if op == "ReplaceNode":
            skill = str(spec.get("skill", "")).strip()
            target = str(spec.get("node_id", node_id)).strip()
            if not skill:
                return None
            return ReplaceNode(node_id=target, skill=skill)
        if op == "AddEdge":
            source = str(spec.get("source", "")).strip()
            target = str(spec.get("target", node_id)).strip()
            if not source:
                source = sinks[0] if sinks else ""
            if not source or not target:
                return None
            return AddEdge(source=source, target=target)
        if op == "RemoveEdge":
            source = str(spec.get("source", "")).strip()
            target = str(spec.get("target", "")).strip()
            if not source or not target:
                return None
            return RemoveEdge(source=source, target=target)
        if op == "InsertMotif":
            steps = spec.get("steps", [])
            if not isinstance(steps, list):
                return None
            return InsertMotif(steps=[str(x) for x in steps if str(x)])
        return None

    def _dedup_ops(self, ops: List[GraphEditOperator]) -> List[GraphEditOperator]:
        seen = set()
        out = []
        for op in ops:
            key = repr(op)
            if key in seen:
                continue
            seen.add(key)
            out.append(op)
        return out
