from pathlib import Path

from anime_pipeline_graph.domain.models import CapabilityPlan, TaskSpec
from anime_pipeline_graph.planner.case_memory import CaseMemory, CaseMemoryItem
from anime_pipeline_graph.planner.candidate_graph_planner import CandidateGraphPlanner
from anime_pipeline_graph.planner.skill_library import SkillLibrary
from anime_pipeline_graph.providers.mock_qwen_client import MockQwenClient
from anime_pipeline_graph.skills.registry import SkillRegistry


def test_candidate_graph_planner_produces_three_sources():
    spec = TaskSpec(
        task_id="t2",
        task_type="storyboard",
        num_frames=2,
        character_names=["lulu"],
        scene_names=["cyber_city"],
        needs_identity_preservation=True,
        needs_pose_control=True,
        needs_story_continuity=True,
    )
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
    library = SkillLibrary.from_registry(
        registry.skills,
        library_dir=Path("src/anime_pipeline_graph/skills/library"),
    )

    plan = CandidateGraphPlanner(MockQwenClient()).plan(spec, cap, registry.as_dict(), library)
    assert len(plan.candidates) >= 3
    sources = {c.source for c in plan.candidates}
    assert {"llm_proposal", "motif_assembly", "fallback"}.issubset(sources)

    for c in plan.candidates:
        assert c.graph.steps
        assert c.graph.edges is not None


def test_candidate_graph_planner_adds_memory_replay_when_available():
    spec = TaskSpec(
        task_id="t_mem",
        task_type="storyboard",
        num_frames=6,
        character_names=["lulu"],
        scene_names=["cyber_city"],
        needs_identity_preservation=True,
        needs_pose_control=True,
        needs_story_continuity=True,
    )
    cap = CapabilityPlan(
        identity_preservation=True,
        local_editing=True,
        pose_control=True,
        scene_reference_conditioning=True,
        story_continuity=True,
        multi_character_interaction=False,
        quality_refinement=True,
    )
    registry = SkillRegistry()
    library = SkillLibrary.from_registry(
        registry.skills,
        library_dir=Path("src/anime_pipeline_graph/skills/library"),
    )
    case = CaseMemoryItem(
        run_id="run_hist_001",
        task_spec_summary={
            "task_type": "storyboard",
            "num_frames": 6,
            "character_names": ["lulu"],
            "scene_names": ["cyber_city"],
            "needs_local_editing": True,
            "needs_pose_control": True,
        },
        graph_summary={
            "num_steps": 4,
            "num_edges": 3,
            "skills": ["character_bind", "prompt_pack_builder", "base_generation", "judge"],
        },
        outcome={"success": True, "final_score": 0.93, "repair_count": 0, "failure_modes": []},
        graph_payload={
            "graph_id": "graph_hist_ok",
            "steps": [
                {"step_id": "character_bind_1", "skill": "character_bind", "inputs": {}, "params": {}, "outputs": ["char_anchor"]},
                {"step_id": "prompt_pack_builder_2", "skill": "prompt_pack_builder", "inputs": {}, "params": {}, "outputs": ["prompt_pack"]},
                {"step_id": "base_generation_3", "skill": "base_generation", "inputs": {}, "params": {}, "outputs": ["generated_images"]},
                {"step_id": "judge_4", "skill": "judge", "inputs": {}, "params": {}, "outputs": ["judge_report"]},
            ],
            "edges": [
                ["character_bind_1", "prompt_pack_builder_2"],
                ["prompt_pack_builder_2", "base_generation_3"],
                ["base_generation_3", "judge_4"],
            ],
            "metadata": {"candidate_source": "historical"},
        },
    )
    memory = CaseMemory(items=[case])

    plan = CandidateGraphPlanner(MockQwenClient()).plan(
        spec,
        cap,
        registry.as_dict(),
        library,
        case_memory=memory,
    )
    sources = {c.source for c in plan.candidates}
    assert "memory_replay" in sources
