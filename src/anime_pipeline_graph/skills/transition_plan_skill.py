"""Transition planning skill."""

from __future__ import annotations


def execute(step, store, ctx):
    """Plan frame-to-frame transition semantics."""
    spec = store.get("task_spec")
    payload = {
        "frame_descriptions": getattr(spec, "frame_descriptions", []),
        "num_frames": int(getattr(spec, "num_frames", 1) or 1),
        "task_type": getattr(getattr(spec, "task_type", None), "value", "single_image"),
    }
    planner = getattr(ctx.qwen_client, "transition_plan", None)
    if callable(planner):
        data = planner(payload)
    else:
        data = {"transitions": []}
    transitions = data.get("transitions", []) if isinstance(data, dict) else []
    store.set("transition_plan_json", {"transitions": transitions})
    return {"outputs": {"transition_plan_json": {"transitions": transitions}}, "artifacts": []}

