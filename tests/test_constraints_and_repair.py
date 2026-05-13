from pathlib import Path

from anime_pipeline_graph.domain.models import GraphStep, SkillGraph, TaskSpec
from anime_pipeline_graph.planner.constraints import ConstraintValidator
from anime_pipeline_graph.planner.graph_repair_search import BestFirstGraphRepair
from anime_pipeline_graph.planner.skill_library import SkillLibrary
from anime_pipeline_graph.skills.registry import SkillRegistry


def test_constraints_and_repair_reduce_violations():
    graph = SkillGraph(
        graph_id="g_repair",
        steps=[
            GraphStep(step_id="scene_1", skill="scene_condition", inputs={}, params={}, outputs=["scene_pack"]),
            GraphStep(step_id="gen_1", skill="base_generation", inputs={}, params={}, outputs=["generated_images"]),
        ],
        edges=[("scene_1", "gen_1")],
        metadata={"task_type": "storyboard", "num_frames": 2},
    )
    spec = TaskSpec(task_id="tx", task_type="storyboard", num_frames=2)
    library = SkillLibrary.from_registry(
        SkillRegistry().skills,
        library_dir=Path("src/anime_pipeline_graph/skills/library"),
    )
    validator = ConstraintValidator()

    violations = validator.validate(graph, spec, library)
    assert violations
    codes = {v.code for v in violations}
    assert "must_have_judge" in codes

    repaired, remaining, _ = BestFirstGraphRepair(validator, beam_width=4, max_iters=20).repair(
        graph,
        spec,
        library,
        initial_violations=violations,
    )
    assert len(remaining) < len(violations)
    # Search may prioritize reducing higher total objective first; ensure it improved violations.
    assert repaired.steps


def test_multiframe_requires_story_decompose():
    graph = SkillGraph(
        graph_id="g_multiframe_no_decompose",
        steps=[
            GraphStep(step_id="prompt_1", skill="prompt_pack_builder", inputs={}, params={}, outputs=["prompt_pack"]),
            GraphStep(step_id="gen_1", skill="base_generation", inputs={}, params={}, outputs=["generated_images"]),
            GraphStep(step_id="judge_1", skill="judge", inputs={}, params={}, outputs=["judge_report"]),
        ],
        edges=[("prompt_1", "gen_1"), ("gen_1", "judge_1")],
        metadata={"task_type": "storyboard", "num_frames": 6},
    )
    spec = TaskSpec(task_id="tx2", task_type="storyboard", num_frames=6)
    library = SkillLibrary.from_registry(
        SkillRegistry().skills,
        library_dir=Path("src/anime_pipeline_graph/skills/library"),
    )

    violations = ConstraintValidator().validate(graph, spec, library)
    codes = {v.code for v in violations}
    assert "multiframe_needs_story_decompose" in codes


def test_repair_can_insert_story_decompose_for_multiframe():
    graph = SkillGraph(
        graph_id="g_multiframe_repair_decompose",
        steps=[
            GraphStep(step_id="prompt_1", skill="prompt_pack_builder", inputs={}, params={}, outputs=["prompt_pack"]),
            GraphStep(step_id="gen_1", skill="base_generation", inputs={}, params={}, outputs=["generated_images"]),
            GraphStep(step_id="judge_1", skill="judge", inputs={}, params={}, outputs=["judge_report"]),
        ],
        edges=[("prompt_1", "gen_1"), ("gen_1", "judge_1")],
        metadata={"task_type": "storyboard", "num_frames": 6},
    )
    spec = TaskSpec(task_id="tx3", task_type="storyboard", num_frames=6)
    library = SkillLibrary.from_registry(
        SkillRegistry().skills,
        library_dir=Path("src/anime_pipeline_graph/skills/library"),
    )
    validator = ConstraintValidator()
    violations = validator.validate(graph, spec, library)
    assert any(v.code == "multiframe_needs_story_decompose" for v in violations)

    repaired, remaining, _ = BestFirstGraphRepair(validator, beam_width=4, max_iters=20).repair(
        graph,
        spec,
        library,
        initial_violations=violations,
    )
    assert any(s.skill == "story_decompose" for s in repaired.steps)
    assert not any(v.code == "multiframe_needs_story_decompose" for v in remaining)


def test_story_decompose_to_transition_and_shot_is_type_compatible():
    graph = SkillGraph(
        graph_id="g_story_camera_compat",
        steps=[
            GraphStep(step_id="story_1", skill="story_decompose", inputs={}, params={}, outputs=["frame_specs"]),
            GraphStep(step_id="transition_1", skill="transition_plan", inputs={}, params={}, outputs=["transition_plan_json"]),
            GraphStep(step_id="shot_1", skill="shot_plan", inputs={}, params={}, outputs=["shot_plan_json"]),
            GraphStep(step_id="prompt_1", skill="prompt_pack_builder", inputs={}, params={}, outputs=["prompt_pack"]),
            GraphStep(step_id="gen_1", skill="base_generation", inputs={}, params={}, outputs=["generated_images"]),
            GraphStep(step_id="judge_1", skill="judge", inputs={}, params={}, outputs=["judge_report"]),
        ],
        edges=[
            ("story_1", "transition_1"),
            ("transition_1", "shot_1"),
            ("shot_1", "prompt_1"),
            ("prompt_1", "gen_1"),
            ("gen_1", "judge_1"),
        ],
        metadata={"task_type": "storyboard", "num_frames": 6},
    )
    spec = TaskSpec(task_id="tx4", task_type="storyboard", num_frames=6)
    library = SkillLibrary.from_registry(
        SkillRegistry().skills,
        library_dir=Path("src/anime_pipeline_graph/skills/library"),
    )
    violations = ConstraintValidator().validate(graph, spec, library)
    assert not any(v.code == "type_compatible_edges" for v in violations)
