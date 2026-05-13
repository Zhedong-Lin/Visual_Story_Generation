"""Score candidate skill graphs for selection."""

from __future__ import annotations

import os
from typing import Dict, List

from anime_pipeline_graph.domain.models import SkillGraph, TaskSpec
from anime_pipeline_graph.planner.case_memory import CaseMemory
from anime_pipeline_graph.planner.planning_models import GraphScore, Violation


class GraphScorer:
    """Compute coverage/validity/prior/cost and total score."""

    def __init__(self, case_memory: CaseMemory | None = None) -> None:
        self.case_memory = case_memory

    def score(
        self,
        graph: SkillGraph,
        task_spec: TaskSpec,
        violations: List[Violation],
        motif_name: str | None = None,
    ) -> GraphScore:
        coverage = self._coverage_score(graph, task_spec)
        validity = self._validity_score(violations)
        prior = self._prior_score(graph, task_spec, motif_name)
        cost = self._cost_score(graph, task_spec)

        total = 0.45 * coverage + 0.30 * validity + 0.15 * prior + 0.10 * cost
        diagnostics = {
            "num_steps": len(graph.steps),
            "num_edges": len(graph.edges),
            "violation_codes": [v.code for v in violations],
            "motif_name": motif_name,
        }
        return GraphScore(
            total_score=round(total, 4),
            coverage_score=round(coverage, 4),
            validity_score=round(validity, 4),
            prior_score=round(prior, 4),
            cost_score=round(cost, 4),
            diagnostics=diagnostics,
        )

    def _coverage_score(self, graph: SkillGraph, task_spec: TaskSpec) -> float:
        skills = {s.skill for s in graph.steps}
        required = {"judge"} if os.getenv("ENABLE_JUDGE", "0").strip().lower() in {"1", "true", "yes", "y", "on", "enabled"} else set()
        optional_story_enhancers = set()

        enable_edit = os.getenv("ENABLE_EDIT", "0").strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}
        if enable_edit and task_spec.needs_local_editing and task_spec.source_image:
            required.add("edit")
        else:
            required.update({"prompt_pack_builder", "base_generation"})

        if task_spec.needs_identity_preservation:
            required.add("character_bind")
        if task_spec.needs_pose_control:
            required.add("pose_plan")
        if task_spec.num_frames > 1 or task_spec.needs_story_continuity:
            required.update({"story_decompose", "expression_plan"})
            optional_story_enhancers.update(
                {"shot_plan", "transition_plan", "continuity_state_tracker", "visual_style_plan"}
            )

        hit = len(required & skills)
        base = hit / max(len(required), 1)
        if not optional_story_enhancers:
            return base
        opt_hit = len(optional_story_enhancers & skills)
        opt_ratio = opt_hit / max(len(optional_story_enhancers), 1)
        # Soft bonus for richer storyboard planning skills; keeps backward compatibility.
        return min(1.0, base * 0.88 + opt_ratio * 0.12)

    def _validity_score(self, violations: List[Violation]) -> float:
        if not violations:
            return 1.0
        penalty = 0.0
        for v in violations:
            if v.severity == "error":
                penalty += 0.25
            elif v.severity == "warning":
                penalty += 0.10
            else:
                penalty += 0.05
        return max(0.0, 1.0 - penalty)

    def _prior_score(self, graph: SkillGraph, task_spec: TaskSpec, motif_name: str | None) -> float:
        base = 0.5
        if motif_name:
            base += 0.05
        if not self.case_memory:
            return min(1.0, base)

        similar = self.case_memory.retrieve_similar_cases(task_spec, top_k=5)
        if not similar:
            return min(1.0, base)

        graph_skills = {s.skill for s in graph.steps}
        weighted = []
        for case in similar:
            case_skills = set(case.graph_summary.get("skills", []))
            overlap = len(graph_skills & case_skills) / max(len(graph_skills | case_skills), 1)
            success = 1.0 if case.outcome.get("success") else 0.3
            final_score = case.outcome.get("final_score")
            quality = float(final_score) if isinstance(final_score, (int, float)) else 0.7
            weighted.append((0.5 * overlap + 0.3 * success + 0.2 * quality))

        if not weighted:
            return min(1.0, base)
        return min(1.0, base + sum(weighted) / len(weighted) * 0.35)

    def _cost_score(self, graph: SkillGraph, task_spec: TaskSpec) -> float:
        n = len(graph.steps)
        e = len(graph.edges)

        # Lightweight complexity heuristic: enough structure but avoid bloated graphs.
        target_steps = 4
        if task_spec.num_frames > 1:
            target_steps = 7
        if task_spec.needs_local_editing and task_spec.source_image:
            target_steps = 4

        step_penalty = abs(n - target_steps) * 0.08
        edge_penalty = max(0, e - (n + 2)) * 0.03
        return max(0.0, 1.0 - step_penalty - edge_penalty)
