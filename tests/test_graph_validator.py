from anime_pipeline_graph.domain.models import GraphStep, SkillGraph
from anime_pipeline_graph.planner.graph_validator import GraphValidator
import os


def test_graph_validator_adds_judge():
    graph = SkillGraph(
        graph_id="g1",
        steps=[GraphStep(step_id="a", skill="scene_condition", inputs={}, params={}, outputs=["scene_pack"])],
        edges=[],
        metadata={"num_frames": 1},
    )
    validated, issues = GraphValidator().validate(graph, auto_fix=True)
    assert any(s.skill == "judge" for s in validated.steps)
    assert issues


def test_graph_validator_autofix_missing_step_referenced_by_edge():
    graph = SkillGraph(
        graph_id="g2",
        steps=[
            GraphStep(step_id="scene_condition_1", skill="scene_condition", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="base_generation_1", skill="base_generation", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="judge_1", skill="judge", inputs={}, params={}, outputs=[]),
        ],
        edges=[
            ("scene_condition_1", "prompt_pack_builder"),
            ("prompt_pack_builder", "base_generation_1"),
            ("base_generation_1", "judge_1"),
        ],
        metadata={"num_frames": 2, "character_names": ["lulu"]},
    )
    fixed, issues = GraphValidator().validate(graph, auto_fix=True)
    step_ids = {s.step_id for s in fixed.steps}
    assert "prompt_pack_builder_autofix" in step_ids or "prompt_pack_builder" in step_ids
    pp_id = "prompt_pack_builder_autofix" if "prompt_pack_builder_autofix" in step_ids else "prompt_pack_builder"
    assert ("scene_condition_1", pp_id) in fixed.edges
    assert (pp_id, "base_generation_1") in fixed.edges
    assert any("prompt_pack_builder" in x for x in issues)


def test_graph_validator_prunes_unnecessary_steps_for_single_image():
    graph = SkillGraph(
        graph_id="g3",
        steps=[
            GraphStep(step_id="story_decompose_1", skill="story_decompose", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="scene_condition_1", skill="scene_condition", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="pose_plan_1", skill="pose_plan", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="character_bind_1", skill="character_bind", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="prompt_pack_builder_1", skill="prompt_pack_builder", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="base_generation_1", skill="base_generation", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="judge_1", skill="judge", inputs={}, params={}, outputs=[]),
        ],
        edges=[
            ("story_decompose_1", "pose_plan_1"),
            ("scene_condition_1", "prompt_pack_builder_1"),
            ("pose_plan_1", "prompt_pack_builder_1"),
            ("character_bind_1", "prompt_pack_builder_1"),
            ("prompt_pack_builder_1", "base_generation_1"),
            ("base_generation_1", "judge_1"),
        ],
        metadata={
            "task_type": "single_image",
            "num_frames": 1,
            "has_scene_reference": False,
            "needs_pose_control": False,
            "needs_story_continuity": False,
            "needs_local_editing": False,
            "character_names": ["jiddo"],
            "has_character_reference": True,
        },
    )
    fixed, issues = GraphValidator().validate(graph, auto_fix=True)
    skills = {s.skill for s in fixed.steps}
    assert "story_decompose" not in skills
    assert "scene_condition" not in skills
    assert "pose_plan" not in skills
    assert "character_bind" in skills
    assert "prompt_pack_builder" in skills
    assert "base_generation" in skills
    assert "judge" in skills
    assert any("pruned unnecessary steps for task shape" in x for x in issues)


def test_graph_validator_inserts_prompt_pack_builder_for_base_generation():
    graph = SkillGraph(
        graph_id="g4",
        steps=[
            GraphStep(step_id="character_bind_1", skill="character_bind", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="base_generation_1", skill="base_generation", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="judge_1", skill="judge", inputs={}, params={}, outputs=[]),
        ],
        edges=[("character_bind_1", "base_generation_1"), ("base_generation_1", "judge_1")],
        metadata={
            "task_type": "single_image",
            "num_frames": 1,
            "has_scene_reference": False,
            "needs_pose_control": False,
            "needs_story_continuity": False,
            "needs_local_editing": False,
            "character_names": ["jiddo"],
            "has_character_reference": True,
        },
    )
    fixed, issues = GraphValidator().validate(graph, auto_fix=True)
    ids = {s.step_id for s in fixed.steps}
    skills = {s.skill for s in fixed.steps}
    assert "prompt_pack_builder" in skills
    pp_id = "prompt_pack_builder_autofix"
    assert pp_id in ids
    assert (pp_id, "base_generation_1") in fixed.edges
    assert any("inserted prompt_pack_builder" in x for x in issues)


def test_graph_validator_inserts_edit_for_local_edit_task_with_source_image():
    graph = SkillGraph(
        graph_id="g5",
        steps=[
            GraphStep(step_id="character_bind_1", skill="character_bind", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="judge_1", skill="judge", inputs={}, params={}, outputs=[]),
        ],
        edges=[],
        metadata={
            "task_type": "single_image",
            "num_frames": 1,
            "has_scene_reference": False,
            "needs_pose_control": False,
            "needs_story_continuity": False,
            "needs_local_editing": True,
            "source_image": "/tmp/jiddo.png",
            "character_names": ["jiddo"],
            "has_character_reference": True,
        },
    )
    fixed, issues = GraphValidator().validate(graph, auto_fix=True)
    skills = {s.skill for s in fixed.steps}
    assert "edit" in skills
    assert "base_generation" in skills
    assert "prompt_pack_builder" in skills
    step_ids = {s.step_id for s in fixed.steps}
    assert "edit_autofix" in step_ids
    assert ("base_generation_autofix", "edit_autofix") in fixed.edges
    assert ("edit_autofix", "judge_1") in fixed.edges
    assert any("inserted edit step" in x for x in issues)


def test_graph_validator_keeps_pose_for_edit_when_needs_pose_control_true():
    graph = SkillGraph(
        graph_id="g6",
        steps=[
            GraphStep(step_id="pose_plan_1", skill="pose_plan", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="character_bind_1", skill="character_bind", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="edit_1", skill="edit", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="judge_1", skill="judge", inputs={}, params={}, outputs=[]),
        ],
        edges=[("pose_plan_1", "edit_1"), ("character_bind_1", "edit_1"), ("edit_1", "judge_1")],
        metadata={
            "task_type": "single_image",
            "num_frames": 1,
            "has_scene_reference": False,
            "needs_pose_control": True,
            "needs_story_continuity": False,
            "needs_local_editing": True,
            "source_image": "/tmp/jiddo.png",
            "character_names": ["jiddo"],
            "has_character_reference": True,
        },
    )
    fixed, _ = GraphValidator().validate(graph, auto_fix=True)
    skills = {s.skill for s in fixed.steps}
    assert "pose_plan" in skills


def test_graph_validator_connects_story_decompose_as_upstream_for_storyboard():
    graph = SkillGraph(
        graph_id="g7",
        steps=[
            GraphStep(step_id="character_bind_1", skill="character_bind", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="scene_condition_1", skill="scene_condition", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="pose_plan_1", skill="pose_plan", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="expression_plan_1", skill="expression_plan", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="prompt_pack_builder_1", skill="prompt_pack_builder", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="base_generation_1", skill="base_generation", inputs={}, params={}, outputs=[]),
            GraphStep(step_id="judge_1", skill="judge", inputs={}, params={}, outputs=[]),
        ],
        edges=[
            ("character_bind_1", "prompt_pack_builder_1"),
            ("scene_condition_1", "prompt_pack_builder_1"),
            ("pose_plan_1", "prompt_pack_builder_1"),
            ("expression_plan_1", "prompt_pack_builder_1"),
            ("prompt_pack_builder_1", "base_generation_1"),
            ("base_generation_1", "judge_1"),
        ],
        metadata={
            "task_type": "storyboard",
            "num_frames": 6,
            "needs_story_continuity": True,
            "character_names": ["lulu"],
            "has_character_reference": True,
        },
    )
    fixed, issues = GraphValidator().validate(graph, auto_fix=True)
    story_ids = [s.step_id for s in fixed.steps if s.skill == "story_decompose"]
    assert story_ids
    sid = story_ids[0]
    assert (sid, "pose_plan_1") in fixed.edges
    assert (sid, "expression_plan_1") in fixed.edges
    assert (sid, "prompt_pack_builder_1") in fixed.edges
    assert any("story_decompose" in x for x in issues)


def test_graph_validator_strict_mainchain_enforces_linear_storyboard_chain():
    old = os.environ.get("STRICT_MAINCHAIN_DAG")
    os.environ["STRICT_MAINCHAIN_DAG"] = "1"
    try:
        graph = SkillGraph(
            graph_id="g8",
            steps=[
                GraphStep(step_id="story_decompose_1", skill="story_decompose", inputs={}, params={}, outputs=[]),
                GraphStep(step_id="character_bind_1", skill="character_bind", inputs={}, params={}, outputs=[]),
                GraphStep(step_id="scene_condition_1", skill="scene_condition", inputs={}, params={}, outputs=[]),
                GraphStep(step_id="pose_plan_1", skill="pose_plan", inputs={}, params={}, outputs=[]),
                GraphStep(step_id="expression_plan_1", skill="expression_plan", inputs={}, params={}, outputs=[]),
                GraphStep(step_id="prompt_pack_builder_1", skill="prompt_pack_builder", inputs={}, params={}, outputs=[]),
                GraphStep(step_id="base_generation_1", skill="base_generation", inputs={}, params={}, outputs=[]),
                GraphStep(step_id="judge_1", skill="judge", inputs={}, params={}, outputs=[]),
                GraphStep(step_id="transition_plan_1", skill="transition_plan", inputs={}, params={}, outputs=[]),
            ],
            edges=[
                ("story_decompose_1", "pose_plan_1"),
                ("story_decompose_1", "expression_plan_1"),
                ("story_decompose_1", "prompt_pack_builder_1"),
                ("character_bind_1", "prompt_pack_builder_1"),
                ("scene_condition_1", "prompt_pack_builder_1"),
                ("prompt_pack_builder_1", "base_generation_1"),
                ("base_generation_1", "judge_1"),
                ("base_generation_1", "transition_plan_1"),
            ],
            metadata={
                "task_type": "storyboard",
                "num_frames": 4,
                "needs_story_continuity": True,
                "character_names": ["dog"],
                "has_character_reference": False,
                "needs_local_editing": False,
            },
        )
        fixed, issues = GraphValidator().validate(graph, auto_fix=True)
        fixed_ids = [s.step_id for s in fixed.steps]
        assert fixed_ids[0] == "story_decompose_1"
        assert fixed_ids[-1] == "judge_1"
        assert len(fixed.edges) == max(0, len(fixed.steps) - 1)
        assert ("base_generation_1", "judge_1") in fixed.edges
        assert any("strict mainchain DAG enforced" in x for x in issues)
    finally:
        if old is None:
            os.environ.pop("STRICT_MAINCHAIN_DAG", None)
        else:
            os.environ["STRICT_MAINCHAIN_DAG"] = old


def test_graph_validator_connects_continuity_tracker_to_prompt_pack_in_non_strict_mode():
    old = os.environ.get("STRICT_MAINCHAIN_DAG")
    if "STRICT_MAINCHAIN_DAG" in os.environ:
        os.environ.pop("STRICT_MAINCHAIN_DAG", None)
    try:
        graph = SkillGraph(
            graph_id="g9",
            steps=[
                GraphStep(step_id="story_decompose_1", skill="story_decompose", inputs={}, params={}, outputs=[]),
                GraphStep(step_id="continuity_state_tracker_1", skill="continuity_state_tracker", inputs={}, params={}, outputs=[]),
                GraphStep(step_id="prompt_pack_builder_1", skill="prompt_pack_builder", inputs={}, params={}, outputs=[]),
                GraphStep(step_id="base_generation_1", skill="base_generation", inputs={}, params={}, outputs=[]),
                GraphStep(step_id="judge_1", skill="judge", inputs={}, params={}, outputs=[]),
            ],
            edges=[
                ("story_decompose_1", "prompt_pack_builder_1"),
                ("prompt_pack_builder_1", "base_generation_1"),
                ("base_generation_1", "judge_1"),
            ],
            metadata={
                "task_type": "storyboard",
                "num_frames": 4,
                "needs_story_continuity": True,
                "character_names": ["dog"],
                "has_character_reference": False,
                "needs_local_editing": False,
            },
        )
        fixed, issues = GraphValidator().validate(graph, auto_fix=True)
        assert ("continuity_state_tracker_1", "prompt_pack_builder_1") in fixed.edges
        assert any("planning nodes into prompt_pack_builder" in x for x in issues)
    finally:
        if old is None:
            os.environ.pop("STRICT_MAINCHAIN_DAG", None)
        else:
            os.environ["STRICT_MAINCHAIN_DAG"] = old
