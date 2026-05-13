from pathlib import Path

from anime_pipeline_graph.planner.skill_library import SkillLibrary
from anime_pipeline_graph.skills.registry import SkillRegistry


def test_skill_library_failure_mode_repairs_loaded():
    lib = SkillLibrary.from_registry(
        SkillRegistry().skills,
        library_dir=Path("src/anime_pipeline_graph/skills/library"),
    )

    reps = lib.get_repairs_for_skill("prompt_pack_builder", ["missing_prompt_builder"])
    assert reps
    assert any(r.get("op") == "AddNode" and r.get("skill") == "prompt_pack_builder" for r in reps)

    reps2 = lib.get_repairs_for_skill("base_generation", ["prompt_underfit"])
    assert any(r.get("op") == "AddNode" and r.get("skill") == "prompt_pack_builder" for r in reps2)

