"""Shot/camera planning skill."""

from __future__ import annotations


def execute(step, store, ctx):
    """Plan per-frame cinematic shot language."""
    spec = store.get("task_spec")
    payload = {
        "character_names": getattr(spec, "character_names", []),
        "frame_descriptions": getattr(spec, "frame_descriptions", []),
        "num_frames": int(getattr(spec, "num_frames", 1) or 1),
        "scene_names": getattr(spec, "scene_names", []),
    }
    planner = getattr(ctx.qwen_client, "shot_plan", None)
    if callable(planner):
        data = planner(payload)
    else:
        data = {"shots": []}
    shots = data.get("shots", []) if isinstance(data, dict) else []
    store.set("shot_plan_json", {"shots": shots})
    return {"outputs": {"shot_plan_json": {"shots": shots}}, "artifacts": []}

