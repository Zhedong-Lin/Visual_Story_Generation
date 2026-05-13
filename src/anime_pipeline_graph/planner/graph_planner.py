"""Candidate-based graph planner wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from anime_pipeline_graph.domain.models import CapabilityPlan, SkillGraph, TaskSpec
from anime_pipeline_graph.planner.candidate_graph_planner import CandidateGraphPlanner
from anime_pipeline_graph.planner.case_memory import CaseMemory
from anime_pipeline_graph.planner.constraints import ConstraintValidator
from anime_pipeline_graph.planner.graph_repair_search import BestFirstGraphRepair
from anime_pipeline_graph.planner.graph_scorer import GraphScorer
from anime_pipeline_graph.planner.planning_models import CandidateGraphPlan
from anime_pipeline_graph.planner.skill_library import SkillLibrary
from anime_pipeline_graph.planner.typed_graph import to_legacy_graph
from anime_pipeline_graph.skills.registry import SkillRegistry


class GraphPlanner:
    """Create dynamic graph from task/capability context via candidate search."""

    def __init__(
        self,
        qwen_client: Any,
        skill_library: SkillLibrary | None = None,
        case_memory: CaseMemory | None = None,
    ) -> None:
        self.qwen_client = qwen_client
        self.skill_library = skill_library or SkillLibrary.from_registry(
            SkillRegistry().skills,
            library_dir=Path(__file__).resolve().parents[1] / "skills" / "library",
        )
        self.case_memory = case_memory
        self.constraint_validator = ConstraintValidator()
        self.scorer = GraphScorer(case_memory=case_memory)
        self.repairer = BestFirstGraphRepair(self.constraint_validator)

    def plan_candidates(self, spec: TaskSpec, cap: CapabilityPlan, registry_payload: Dict[str, dict]) -> CandidateGraphPlan:
        """Return candidate graphs from multiple generation sources."""
        return CandidateGraphPlanner(self.qwen_client).plan(
            spec,
            cap,
            registry_payload,
            self.skill_library,
            case_memory=self.case_memory,
        )

    def _apply_memory_replay_gate(self, plan: CandidateGraphPlan, spec: TaskSpec) -> None:
        """Penalize storyboard memory_replay candidates that miss key cinematic planning skills."""
        is_storyboard = spec.num_frames > 1 or spec.task_type.value == "storyboard"
        if not is_storyboard:
            return
        required = {"shot_plan", "transition_plan"}
        for candidate in plan.candidates:
            if candidate.source != "memory_replay" or candidate.score is None:
                continue
            skills = {s.skill for s in candidate.graph.steps}
            missing = sorted(required - skills)
            if not missing:
                continue
            penalty = 0.12 * len(missing)
            diagnostics = dict(candidate.score.diagnostics or {})
            diagnostics["memory_replay_missing_skills"] = missing
            diagnostics["memory_replay_penalty"] = round(penalty, 4)
            candidate.score = candidate.score.model_copy(
                update={
                    "total_score": round(max(0.0, candidate.score.total_score - penalty), 4),
                    "diagnostics": diagnostics,
                }
            )

    def select_best(self, plan: CandidateGraphPlan, spec: TaskSpec) -> SkillGraph:
        """Score/rank candidates, apply repair-search when needed, and return best graph."""
        for candidate in plan.candidates:
            candidate.violations = self.constraint_validator.validate(candidate.graph, spec, self.skill_library)
            motif_name = str(candidate.metadata.get("motif")) if candidate.metadata else None
            candidate.score = self.scorer.score(candidate.graph, spec, candidate.violations, motif_name=motif_name)
        self._apply_memory_replay_gate(plan, spec)

        ranked = sorted(plan.candidates, key=lambda c: (c.score.total_score if c.score else -1.0), reverse=True)
        best = ranked[0]

        if best.violations:
            repaired_graph, repaired_violations, repair_stats = self.repairer.repair(
                best.graph,
                spec,
                self.skill_library,
                initial_violations=best.violations,
            )
            repaired_score = self.scorer.score(repaired_graph, spec, repaired_violations)
            if repaired_score.total_score >= (best.score.total_score if best.score else -1.0):
                best.graph = repaired_graph
                best.violations = repaired_violations
                best.score = repaired_score
                best.metadata["repair_stats"] = repair_stats

        # Keep existing runtime compatibility (tuple edges expected by executor/tests).
        out = to_legacy_graph(best.graph)
        out.metadata["candidate_source"] = best.source
        out.metadata["candidate_score"] = best.score.model_dump(mode="json") if best.score else {}
        out.metadata["candidate_violations"] = [v.model_dump(mode="json") for v in best.violations]
        out.metadata["candidate_count"] = len(plan.candidates)
        return out

    def plan(self, spec: TaskSpec, cap: CapabilityPlan, registry_payload: Dict[str, dict]) -> SkillGraph:
        """Generate candidates, score/select, and return best executable graph."""
        plan = self.plan_candidates(spec, cap, registry_payload)
        return self.select_best(plan, spec)
