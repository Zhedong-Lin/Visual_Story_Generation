from anime_pipeline_graph.domain.models import InputBundle
from anime_pipeline_graph.parser.task_parser import TaskParser
from anime_pipeline_graph.providers.mock_qwen_client import MockQwenClient


class _MultiImageQwenClient(MockQwenClient):
    def parse_task(self, payload):
        out = super().parse_task(payload)
        out["task_type"] = "multi_image"
        out["num_frames"] = max(2, out.get("num_frames", 1))
        return out


class _NoisyEntityQwenClient(MockQwenClient):
    def parse_task(self, payload):
        out = super().parse_task(payload)
        out["task_type"] = "storyboard"
        out["num_frames"] = 4
        out["character_names"] = ["dog", "crosses", "bridge", "bone"]
        out["scene_names"] = []
        out["num_characters"] = 4
        out["needs_multi_character_interaction"] = True
        return out


def test_parser_outputs_fields():
    parser = TaskParser(MockQwenClient())
    spec = parser.parse(InputBundle(user_text="lulu在cyber_city被jiddo追"))
    assert spec.task_id
    assert spec.num_frames >= 1
    assert isinstance(spec.character_names, list)
    assert isinstance(spec.scene_names, list)


def test_parser_infers_local_edit_when_source_image_available():
    parser = TaskParser(MockQwenClient(), known_character_names=["jiddo"], known_scene_names=["cyber_city"])
    bundle = InputBundle(
        user_text="Edit the existing Jiddo image: only change black shoes to white and replace glasses with sunglasses.",
        constraints={"edit_source_map": {"jiddo": "/tmp/jiddo.png"}},
    )
    spec = parser.parse(bundle)
    assert spec.needs_local_editing is True
    assert spec.edit_scope != "none"
    assert spec.source_image == "/tmp/jiddo.png"


def test_parser_edit_task_enables_pose_control_when_pose_words_present():
    parser = TaskParser(MockQwenClient(), known_character_names=["jiddo"], known_scene_names=["cyber_city"])
    bundle = InputBundle(
        user_text="Edit the existing Jiddo image: keep identity, make him sitting with sunglasses and white shoes.",
        constraints={"edit_source_map": {"jiddo": "/tmp/jiddo.png"}},
    )
    spec = parser.parse(bundle)
    assert spec.needs_local_editing is True
    assert spec.source_image == "/tmp/jiddo.png"
    assert spec.needs_pose_control is True


def test_parser_source_image_without_edit_verbs_does_not_force_edit():
    parser = TaskParser(MockQwenClient(), known_character_names=["jiddo"], known_scene_names=["cyber_city"])
    bundle = InputBundle(
        user_text="Generate a portrait of Jiddo in cyber_city at night.",
        constraints={"edit_source_map": {"jiddo": "/tmp/jiddo.png"}},
    )
    spec = parser.parse(bundle)
    assert spec.source_image == "/tmp/jiddo.png"
    assert spec.needs_local_editing is False
    assert spec.edit_scope == "none"


def test_parser_normalizes_multi_image_task_type():
    parser = TaskParser(_MultiImageQwenClient())
    spec = parser.parse(InputBundle(user_text="Create 3 panels of lulu in cyber_city"))
    assert spec.task_type.value == "storyboard"


def test_parser_supports_explicit_ten_frame_storyboard_in_dry_run():
    parser = TaskParser(MockQwenClient())
    spec = parser.parse(InputBundle(user_text="请画10帧分镜，lulu在cyber_city里奔跑然后回家"))
    assert spec.task_type.value == "storyboard"
    assert spec.num_frames == 10


def test_parser_clamps_frame_count_to_supported_max():
    parser = TaskParser(MockQwenClient())
    spec = parser.parse(InputBundle(user_text="Create 16 frames of lulu in cyber_city"))
    assert spec.task_type.value == "storyboard"
    assert spec.num_frames == 15


def test_parser_uses_frame_description_count_over_lower_model_count():
    class _LowerCountQwenClient(MockQwenClient):
        def parse_task(self, payload):
            out = super().parse_task(payload)
            out["task_type"] = "single_image"
            out["num_frames"] = 8
            out["frame_descriptions"] = [f"Beat {i}" for i in range(1, 11)]
            return out

    parser = TaskParser(_LowerCountQwenClient())
    spec = parser.parse(InputBundle(user_text="Create 10 frames of lulu in cyber_city"))
    assert spec.task_type.value == "storyboard"
    assert spec.num_frames == 10


def test_parser_infers_count_from_frame_labels():
    parser = TaskParser(MockQwenClient())
    spec = parser.parse(
        InputBundle(
            user_text=(
                "Anime 2D style. Frame 1: Lulu starts walking. "
                "Frame 2: Lulu finds a seed. Frame 10: Lulu gives Jiddo a flower."
            )
        )
    )
    assert spec.task_type.value == "storyboard"
    assert spec.num_frames == 10


def test_parser_uses_semantic_storyboard_inference_when_no_explicit_frame_count():
    parser = TaskParser(MockQwenClient(), known_character_names=["lulu"], known_scene_names=["cyber_city"])
    bundle = InputBundle(
        user_text=(
            "Lulu walks in cyber_city, then suddenly heavy rain starts. "
            "She gets frightened and runs away."
        )
    )
    spec = parser.parse(bundle)
    assert spec.task_type.value == "storyboard"
    assert spec.num_frames >= 2


def test_parser_filters_action_object_words_from_character_names():
    parser = TaskParser(_NoisyEntityQwenClient())
    bundle = InputBundle(
        user_text=(
            "A dog crosses a bridge with a bone in its mouth. "
            "Looking into the water, it mistakes reflection for another dog."
        )
    )
    spec = parser.parse(bundle)
    assert spec.character_names == ["dog"]
    assert spec.num_characters == 1
    assert spec.needs_multi_character_interaction is False
