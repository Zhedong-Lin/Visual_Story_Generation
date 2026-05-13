from types import SimpleNamespace

from anime_pipeline_graph.skills import prompt_pack_builder


class _Store:
    def __init__(self, state):
        self.state = dict(state)

    def get(self, key, default=None):
        return self.state.get(key, default)

    def set(self, key, value):
        self.state[key] = value


def test_prompt_pack_builder_infers_scene_when_scene_names_empty():
    spec = SimpleNamespace(
        num_frames=2,
        character_names=["dog"],
        scene_names=[],
        frame_descriptions=[
            "A dog walks across a wooden bridge with a bone in its mouth.",
            "The dog drops the bone into the river and looks down at the water.",
        ],
    )
    bundle = SimpleNamespace(
        user_text="A dog walks on a bridge and drops a bone in the river.",
        constraints={},
    )
    store = _Store(
        {
            "task_spec": spec,
            "input_bundle": bundle,
            "scene_pack": {},
            "char_anchor": {},
            "pose_plan_json": {},
            "shot_plan_json": {},
            "transition_plan_json": {},
            "expression_plan_json": {},
            "continuity_state": {},
            "visual_style_plan": {},
            "frame_specs": [],
        }
    )
    out = prompt_pack_builder.execute(None, store, None)
    pack = out["outputs"]["prompt_pack"]
    frame0 = pack["frame_prompt_payloads"][0]["prompt"]
    assert "urban night street" not in frame0
    assert "scene bridge, river" in frame0 or "scene river, bridge" in frame0
