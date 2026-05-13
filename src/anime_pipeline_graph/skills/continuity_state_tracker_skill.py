"""Continuity state tracking skill."""

from __future__ import annotations


def execute(step, store, ctx):
    """Build cross-frame continuity ledger (characters/scene/objects/narrative)."""
    spec = store.get("task_spec")
    shot_plan = store.get("shot_plan_json", {})
    transition_plan = store.get("transition_plan_json", {})
    expression_plan = store.get("expression_plan_json", {})
    payload = {
        "character_names": getattr(spec, "character_names", []),
        "scene_names": getattr(spec, "scene_names", []),
        "frame_descriptions": getattr(spec, "frame_descriptions", []),
        "num_frames": int(getattr(spec, "num_frames", 1) or 1),
        "shot_plan": shot_plan,
        "transition_plan": transition_plan,
        "expression_plan": expression_plan,
    }
    planner = getattr(ctx.qwen_client, "continuity_state", None)
    if callable(planner):
        data = planner(payload)
    else:
        data = {"continuity_ledger": []}
    ledger = data.get("continuity_ledger", []) if isinstance(data, dict) else []
    out = {"continuity_ledger": ledger}
    store.set("continuity_state", out)
    return {"outputs": {"continuity_state": out}, "artifacts": []}

