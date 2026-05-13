"""Explicit graph constraint system for planning-time validation."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, List

import networkx as nx

from anime_pipeline_graph.domain.models import SkillGraph, TaskSpec
from anime_pipeline_graph.planner.planning_models import Violation
from anime_pipeline_graph.planner.skill_library import SkillLibrary
from anime_pipeline_graph.planner.typed_graph import edge_to_tuple, normalized_edges, to_typed_graph


class GraphConstraint(ABC):
    """Base class for one graph constraint."""

    code: str

    @abstractmethod
    def check(self, graph: SkillGraph, task_spec: TaskSpec, skill_library: SkillLibrary) -> List[Violation]:
        """Return violations caused by this constraint."""


class MustHaveJudgeConstraint(GraphConstraint):
    code = "must_have_judge"

    def check(self, graph: SkillGraph, task_spec: TaskSpec, skill_library: SkillLibrary) -> List[Violation]:
        if os.getenv("ENABLE_JUDGE", "0").strip().lower() not in {"1", "true", "yes", "y", "on", "enabled"}:
            return []
        if any(step.skill == "judge" for step in graph.steps):
            return []
        return [
            Violation(
                code=self.code,
                message="Graph must contain a judge step.",
                severity="error",
                related_nodes=[],
                suggested_repairs=[{"op": "AddNode", "skill": "judge"}],
            )
        ]


class MultiFrameNeedsGenerationConstraint(GraphConstraint):
    code = "multiframe_needs_generation"

    def check(self, graph: SkillGraph, task_spec: TaskSpec, skill_library: SkillLibrary) -> List[Violation]:
        is_multiframe = task_spec.num_frames > 1 or task_spec.task_type.value == "storyboard"
        if not is_multiframe:
            return []
        has_generation = any(step.skill in {"base_generation", "edit", "fill_edit"} for step in graph.steps)
        if has_generation:
            return []
        return [
            Violation(
                code=self.code,
                message="Storyboard / multi-frame tasks require generation or edit path.",
                severity="error",
                suggested_repairs=[{"op": "AddNode", "skill": "base_generation"}],
            )
        ]


class MultiFrameNeedsStoryDecomposeConstraint(GraphConstraint):
    code = "multiframe_needs_story_decompose"

    def check(self, graph: SkillGraph, task_spec: TaskSpec, skill_library: SkillLibrary) -> List[Violation]:
        is_multiframe = task_spec.num_frames > 1 or task_spec.task_type.value == "storyboard"
        if not is_multiframe:
            return []

        step_by_id = {s.step_id: s for s in graph.steps}
        story_nodes = [s for s in graph.steps if s.skill == "story_decompose"]
        if not story_nodes:
            pp_nodes = [s.step_id for s in graph.steps if s.skill == "prompt_pack_builder"]
            suggested = [{"op": "AddNode", "skill": "story_decompose", "node_id": "story_decompose_autofix"}]
            if pp_nodes:
                suggested.append(
                    {"op": "AddEdge", "source": "story_decompose_autofix", "target": pp_nodes[0]}
                )
            return [
                Violation(
                    code=self.code,
                    message="Storyboard / multi-frame tasks must include story_decompose before prompt building.",
                    severity="error",
                    related_nodes=pp_nodes,
                    suggested_repairs=suggested,
                )
            ]

        # Require at least one story_decompose node to be upstream of prompt_pack_builder.
        prompt_nodes = [s.step_id for s in graph.steps if s.skill == "prompt_pack_builder"]
        if not prompt_nodes:
            return []

        reverse_incoming: Dict[str, List[str]] = {}
        for src, dst in normalized_edges(graph):
            reverse_incoming.setdefault(dst, []).append(src)

        story_ids = {s.step_id for s in story_nodes}
        for pp in prompt_nodes:
            stack = [pp]
            seen = set()
            found = False
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                if cur in story_ids:
                    found = True
                    break
                stack.extend(reverse_incoming.get(cur, []))
            if not found:
                return [
                    Violation(
                        code=self.code,
                        message=f"prompt_pack_builder step '{pp}' must have story_decompose upstream in storyboard/multi-frame tasks.",
                        severity="error",
                        related_nodes=[pp],
                        suggested_repairs=[
                            {"op": "AddNode", "skill": "story_decompose", "node_id": "story_decompose_autofix"},
                            {"op": "AddEdge", "source": "story_decompose_autofix", "target": pp},
                        ],
                    )
                ]
        return []


class PromptBuilderBeforeGenerationConstraint(GraphConstraint):
    code = "prompt_builder_before_generation"

    def check(self, graph: SkillGraph, task_spec: TaskSpec, skill_library: SkillLibrary) -> List[Violation]:
        incoming: Dict[str, List[str]] = {}
        for src, dst in normalized_edges(graph):
            incoming.setdefault(dst, []).append(src)

        violations: List[Violation] = []
        steps_by_id = {s.step_id: s for s in graph.steps}
        for step in graph.steps:
            if step.skill != "base_generation":
                continue
            upstream_ids = incoming.get(step.step_id, [])
            upstream_skills = {steps_by_id[sid].skill for sid in upstream_ids if sid in steps_by_id}
            if "prompt_pack_builder" not in upstream_skills:
                violations.append(
                    Violation(
                        code=self.code,
                        message=f"base_generation step '{step.step_id}' must have prompt_pack_builder upstream.",
                        severity="error",
                        related_nodes=[step.step_id],
                        suggested_repairs=[{"op": "AddNode", "skill": "prompt_pack_builder"}],
                    )
                )
        return violations


class TypeCompatibleEdgesConstraint(GraphConstraint):
    code = "type_compatible_edges"

    def check(self, graph: SkillGraph, task_spec: TaskSpec, skill_library: SkillLibrary) -> List[Violation]:
        violations: List[Violation] = []
        step_by_id = {s.step_id: s for s in graph.steps}

        for src, dst in normalized_edges(graph):
            if src not in step_by_id or dst not in step_by_id:
                continue
            src_skill = step_by_id[src].skill
            dst_skill = step_by_id[dst].skill
            src_card = skill_library.get_skill(src_skill)
            dst_card = skill_library.get_skill(dst_skill)
            if src_card and src_card.allowed_successors and dst_skill not in src_card.allowed_successors:
                violations.append(
                    Violation(
                        code=self.code,
                        message=f"Edge {src}->{dst} violates allowed_successors of {src_skill}.",
                        severity="warning",
                        related_nodes=[src, dst],
                        suggested_repairs=[{"op": "RemoveEdge", "source": src, "target": dst}],
                    )
                )
            if dst_card and dst_card.allowed_predecessors and src_skill not in dst_card.allowed_predecessors:
                violations.append(
                    Violation(
                        code=self.code,
                        message=f"Edge {src}->{dst} violates allowed_predecessors of {dst_skill}.",
                        severity="warning",
                        related_nodes=[src, dst],
                        suggested_repairs=[{"op": "RemoveEdge", "source": src, "target": dst}],
                    )
                )
        return violations


class PreconditionsSatisfiedConstraint(GraphConstraint):
    code = "preconditions_satisfied"

    @staticmethod
    def _task_precondition_satisfied(precondition: str, task_spec: TaskSpec) -> bool:
        cond = str(precondition).strip().lower()
        if cond == "num_frames>1":
            return int(task_spec.num_frames) > 1
        if cond == "character_exists":
            return bool(task_spec.character_names) or bool(task_spec.has_character_reference)
        if cond == "image_exists":
            return bool(task_spec.source_image)
        # Keep unknown preconditions for graph-effect based resolution.
        return False

    def check(self, graph: SkillGraph, task_spec: TaskSpec, skill_library: SkillLibrary) -> List[Violation]:
        step_by_id = {s.step_id: s for s in graph.steps}
        incoming: Dict[str, List[str]] = {}
        for src, dst in normalized_edges(graph):
            incoming.setdefault(dst, []).append(src)

        violations: List[Violation] = []
        for step in graph.steps:
            card = skill_library.get_skill(step.skill)
            if not card or not card.preconditions:
                continue

            upstream_effects = set()
            for src_id in incoming.get(step.step_id, []):
                src_step = step_by_id.get(src_id)
                if not src_step:
                    continue
                src_card = skill_library.get_skill(src_step.skill)
                if src_card:
                    upstream_effects.update(src_card.effects)

            missing = [
                p
                for p in card.preconditions
                if p not in upstream_effects and not self._task_precondition_satisfied(p, task_spec)
            ]
            if missing:
                violations.append(
                    Violation(
                        code=self.code,
                        message=f"Step '{step.step_id}' missing preconditions: {missing}",
                        severity="warning",
                        related_nodes=[step.step_id],
                        suggested_repairs=[{"op": "ReplaceNode", "node_id": step.step_id}],
                    )
                )
        return violations


class GraphMustBeDAGConstraint(GraphConstraint):
    code = "graph_must_be_dag"

    def check(self, graph: SkillGraph, task_spec: TaskSpec, skill_library: SkillLibrary) -> List[Violation]:
        g = nx.DiGraph()
        for step in graph.steps:
            g.add_node(step.step_id)
        for src, dst in normalized_edges(graph):
            g.add_edge(src, dst)

        if nx.is_directed_acyclic_graph(g):
            return []

        cyc = []
        try:
            cyc = nx.find_cycle(g)
        except Exception:
            pass

        nodes = []
        for src, dst in cyc:
            nodes.extend([src, dst])

        return [
            Violation(
                code=self.code,
                message="Graph must be DAG.",
                severity="error",
                related_nodes=sorted(set(nodes)),
                suggested_repairs=[{"op": "RemoveEdge"}],
            )
        ]


class NoOrphanCriticalNodesConstraint(GraphConstraint):
    code = "no_orphan_critical_nodes"

    def check(self, graph: SkillGraph, task_spec: TaskSpec, skill_library: SkillLibrary) -> List[Violation]:
        critical = {"prompt_pack_builder", "base_generation", "edit", "fill_edit", "judge"}
        incident = {s.step_id: 0 for s in graph.steps}
        for src, dst in normalized_edges(graph):
            if src in incident:
                incident[src] += 1
            if dst in incident:
                incident[dst] += 1

        violations: List[Violation] = []
        for step in graph.steps:
            if step.skill in critical and incident.get(step.step_id, 0) == 0 and len(graph.steps) > 1:
                violations.append(
                    Violation(
                        code=self.code,
                        message=f"Critical step '{step.step_id}' is orphaned.",
                        severity="warning",
                        related_nodes=[step.step_id],
                        suggested_repairs=[{"op": "AddEdge", "target": step.step_id}],
                    )
                )
        return violations


class ConstraintValidator:
    """Constraint runner that emits explicit violations."""

    def __init__(self, constraints: List[GraphConstraint] | None = None) -> None:
        self.constraints = constraints or [
            MustHaveJudgeConstraint(),
            MultiFrameNeedsGenerationConstraint(),
            MultiFrameNeedsStoryDecomposeConstraint(),
            PromptBuilderBeforeGenerationConstraint(),
            TypeCompatibleEdgesConstraint(),
            PreconditionsSatisfiedConstraint(),
            GraphMustBeDAGConstraint(),
            NoOrphanCriticalNodesConstraint(),
        ]

    def validate(self, graph: SkillGraph, task_spec: TaskSpec, skill_library: SkillLibrary) -> List[Violation]:
        """Run all constraints and collect violations."""
        typed = to_typed_graph(graph)
        out: List[Violation] = []
        for constraint in self.constraints:
            out.extend(constraint.check(typed, task_spec, skill_library))
        return out


def build_task_spec_from_graph_metadata(graph: SkillGraph) -> TaskSpec:
    """Build minimal TaskSpec from graph metadata (validator compatibility path)."""
    meta = graph.metadata or {}
    task_type = str(meta.get("task_type", "single_image"))
    return TaskSpec(
        task_id=str(meta.get("task_id", "task_from_graph")),
        task_type=task_type,
        num_frames=int(meta.get("num_frames", 1) or 1),
        num_characters=len(meta.get("character_names", []) or []),
        character_names=list(meta.get("character_names", []) or []),
        scene_names=list(meta.get("scene_names", []) or []),
        source_image=meta.get("source_image"),
        has_character_reference=bool(meta.get("has_character_reference", False)),
        has_scene_reference=bool(meta.get("has_scene_reference", False)),
        has_setting_doc=bool(meta.get("has_setting_doc", False)),
        needs_identity_preservation=bool(meta.get("needs_identity_preservation", False)),
        needs_local_editing=bool(meta.get("needs_local_editing", False)),
        needs_pose_control=bool(meta.get("needs_pose_control", False)),
        needs_scene_generation=bool(meta.get("needs_scene_generation", True)),
        needs_layout_control=bool(meta.get("needs_layout_control", False)),
        needs_story_continuity=bool(meta.get("needs_story_continuity", False)),
        needs_multi_character_interaction=bool(meta.get("needs_multi_character_interaction", False)),
        action_intensity=str(meta.get("action_intensity", "low")),
        edit_scope=str(meta.get("edit_scope", "none")),
        scene_strength=str(meta.get("scene_strength", "medium")),
        priority=str(meta.get("priority", "quality")),
        character_constraints=list(meta.get("character_constraints", []) or []),
        scene_constraints=list(meta.get("scene_constraints", []) or []),
        story_constraints=list(meta.get("story_constraints", []) or []),
        frame_descriptions=list(meta.get("frame_descriptions", []) or []),
        risk_flags=list(meta.get("risk_flags", []) or []),
    )
