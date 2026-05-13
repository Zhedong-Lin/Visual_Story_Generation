"""Expression plan skill."""

from __future__ import annotations


def execute(step, store, ctx):
    """Plan per-frame facial expressions with Qwen."""
    spec = store.get("task_spec")
    frame_specs = store.get("frame_specs", [])
    frame_descriptions = []
    if isinstance(frame_specs, list) and frame_specs:
        for frame in frame_specs:
            if isinstance(frame, dict):
                frame_descriptions.append(frame.get("description") or frame.get("desc") or "")
            elif isinstance(frame, str):
                frame_descriptions.append(frame)
    if not frame_descriptions and spec and getattr(spec, "frame_descriptions", None):
        frame_descriptions = list(spec.frame_descriptions)

    expression_plan = ctx.qwen_client.expression_plan(
        {
            "character_names": spec.character_names if spec else [],
            "num_frames": spec.num_frames if spec else max(len(frame_descriptions), 1),
            "frame_descriptions": frame_descriptions,
            "prompt": frame_descriptions[0] if frame_descriptions else "",
        }
    )
    store.set("expression_plan_json", expression_plan)
    return {"outputs": {"expression_plan_json": expression_plan}, "artifacts": []}

