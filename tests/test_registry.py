from anime_pipeline_graph.skills.registry import SkillRegistry


def test_registry_has_required_skills():
    registry = SkillRegistry().skills
    required = {
        "base_generation",
        "character_bind",
        "edit",
        "fill_edit",
        "scene_condition",
        "pose_plan",
        "pose_extract",
        "story_decompose",
        "judge",
        "refine",
        "prompt_pack_builder",
        "asset_normalizer",
    }
    assert required.issubset(set(registry.keys()))
