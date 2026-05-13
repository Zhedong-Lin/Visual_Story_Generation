from pathlib import Path

from anime_pipeline_graph.domain.models import CapabilityPlan, GraphStep, SkillGraph, TaskSpec
from anime_pipeline_graph.planner.case_memory import CaseMemory
from anime_pipeline_graph.planner.graph_planner import GraphPlanner
from anime_pipeline_graph.planner.planning_models import CandidateGraph, CandidateGraphPlan, GraphScore
from anime_pipeline_graph.planner.skill_library import SkillLibrary
from anime_pipeline_graph.providers.mock_qwen_client import MockQwenClient
from anime_pipeline_graph.skills.registry import SkillRegistry


def _mk_graph(graph_id: str, skills: list[str]) -> SkillGraph:
    steps = [GraphStep(step_id=f"{s}_{i}", skill=s, inputs={}, params={}, outputs=[]) for i, s in enumerate(skills, 1)]
    edges = [(steps[i].step_id, steps[i + 1].step_id) for i in range(len(steps) - 1)]
    return SkillGraph(graph_id=graph_id, steps=steps, edges=edges, metadata={})


def test_memory_replay_gate_penalizes_missing_shot_transition_for_storyboard():
    spec = TaskSpec(task_id="t_gate", task_type="storyboard", num_frames=6, character_names=["lulu"], scene_names=["city"])
    cap = CapabilityPlan(
        identity_preservation=True,
        local_editing=False,
        pose_control=True,
        scene_reference_conditioning=True,
        story_continuity=True,
        multi_character_interaction=False,
        quality_refinement=True,
    )
    registry = SkillRegistry()
    lib = SkillLibrary.from_registry(registry.skills, library_dir=Path("src/anime_pipeline_graph/skills/library"))
    planner = GraphPlanner(MockQwenClient(), skill_library=lib, case_memory=CaseMemory(items=[]))

    memory_candidate = CandidateGraph(
        graph=_mk_graph(
            "g_mem",
            ["story_decompose", "character_bind", "scene_condition", "pose_plan", "expression_plan", "prompt_pack_builder", "base_generation", "judge"],
        ),
        source="memory_replay",
        score=GraphScore(
            total_score=0.9,
            coverage_score=0.9,
            validity_score=0.9,
            prior_score=0.9,
            cost_score=0.9,
            diagnostics={},
        ),
    )
    llm_candidate = CandidateGraph(
        graph=_mk_graph(
            "g_llm",
            [
                "story_decompose",
                "character_bind",
                "scene_condition",
                "transition_plan",
                "shot_plan",
                "pose_plan",
                "expression_plan",
                "prompt_pack_builder",
                "base_generation",
                "judge",
            ],
        ),
        source="llm_proposal",
        score=GraphScore(
            total_score=0.85,
            coverage_score=0.85,
            validity_score=0.85,
            prior_score=0.85,
            cost_score=0.85,
            diagnostics={},
        ),
    )
    plan = CandidateGraphPlan(task_spec=spec, candidates=[memory_candidate, llm_candidate])
    planner._apply_memory_replay_gate(plan, spec)

    assert plan.candidates[0].score is not None
    assert plan.candidates[0].score.total_score == 0.66
    assert plan.candidates[0].score.diagnostics.get("memory_replay_missing_skills") == ["shot_plan", "transition_plan"]
    assert plan.candidates[1].score is not None
    assert plan.candidates[1].score.total_score == 0.85

