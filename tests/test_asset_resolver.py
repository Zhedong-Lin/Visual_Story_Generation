from pathlib import Path

from anime_pipeline_graph.domain.models import InputBundle, TaskSpec
from anime_pipeline_graph.domain.enums import TaskType
from anime_pipeline_graph.parser.asset_resolver import AssetResolver


def test_asset_resolver_soft_fail_on_missing(tmp_path: Path):
    chars = tmp_path / "chars"
    scenes = tmp_path / "scenes"
    chars.mkdir()
    scenes.mkdir()
    resolver = AssetResolver(chars, scenes)

    spec = TaskSpec(task_id="x", task_type=TaskType.SINGLE_IMAGE, character_names=["foo"], scene_names=["bar"])
    bundle, report = resolver.resolve(InputBundle(user_text="x"), spec)
    assert bundle.character_references == {}
    assert "foo" in report.missing_characters
    assert "bar" in report.missing_scenes
