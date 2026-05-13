"""Visual style bible planning skill."""

from __future__ import annotations


def execute(step, store, ctx):
    """Create sequence-level style bible for stable visual tone."""
    spec = store.get("task_spec")
    bundle = store.get("input_bundle")
    payload = {
        "user_text": getattr(bundle, "user_text", ""),
        "task_type": getattr(getattr(spec, "task_type", None), "value", "single_image"),
        "scene_names": getattr(spec, "scene_names", []),
        "character_names": getattr(spec, "character_names", []),
        "num_frames": int(getattr(spec, "num_frames", 1) or 1),
    }
    planner = getattr(ctx.qwen_client, "visual_style_plan", None)
    if callable(planner):
        data = planner(payload)
    else:
        data = {"style_bible": {}}
    style_bible = data.get("style_bible", {}) if isinstance(data, dict) else {}
    out = {"style_bible": style_bible}
    store.set("visual_style_plan", out)
    return {"outputs": {"visual_style_plan": out}, "artifacts": []}

