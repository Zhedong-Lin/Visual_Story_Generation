"""Pose planning skill."""

from __future__ import annotations


def execute(step, store, ctx):
    """Generate structured pose plan JSON via Qwen API."""
    spec = store.get("task_spec")
    bundle = store.get("input_bundle")
    pose_plan = ctx.qwen_client.pose_plan(
        {
            "user_text": bundle.user_text,
            "character_names": spec.character_names,
            "num_frames": spec.num_frames,
        }
    )
    store.set("pose_plan_json", pose_plan)
    return {"outputs": {"pose_plan_json": pose_plan}, "artifacts": []}
