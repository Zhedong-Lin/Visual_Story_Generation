from pathlib import Path

from anime_pipeline_graph.domain.models import TaskSpec
from anime_pipeline_graph.planner.skill_library import SkillLibrary
from anime_pipeline_graph.skills.registry import SkillRegistry


def _story_spec() -> TaskSpec:
    return TaskSpec(
        task_id="t1",
        task_type="storyboard",
        num_frames=3,
        character_names=["lulu", "jiddo"],
        scene_names=["cyber_city"],
        needs_identity_preservation=True,
        needs_pose_control=True,
        needs_story_continuity=True,
    )


def test_skill_library_load_and_retrieve():
    lib = SkillLibrary.from_registry(
        SkillRegistry().skills,
        library_dir=Path("src/anime_pipeline_graph/skills/library"),
    )
    assert lib.get_skill("base_generation") is not None

    skills = [c.name for c in lib.retrieve_candidates(_story_spec(), top_k=8)]
    assert "judge" in skills
    assert "prompt_pack_builder" in skills

    motifs = lib.retrieve_motifs(_story_spec(), top_k=2)
    assert motifs
    assert "steps" in motifs[0]

