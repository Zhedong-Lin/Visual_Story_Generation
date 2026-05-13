"""Story decomposition skill."""

from __future__ import annotations


def execute(step, store, ctx):
    """Create frame specs for multi-frame story tasks."""
    spec = store.get("task_spec")
    bundle = store.get("input_bundle")
    story = ctx.qwen_client.decompose_story({"user_text": bundle.user_text, "num_frames": spec.num_frames})
    frames = story.get("frames", [])
    store.set("frame_specs", frames)
    return {"outputs": {"frame_specs": frames}, "artifacts": []}
