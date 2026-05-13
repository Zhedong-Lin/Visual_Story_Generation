"""Pose extract skill."""

from __future__ import annotations

from pathlib import Path


def execute(step, store, ctx):
    """Extract skeleton from action reference when available."""
    bundle = store.get("input_bundle")
    out = store.images_dir() / "pose_skeleton.png"
    result = ctx.pose_extractor.extract(bundle.action_reference, out)
    store.set("pose_skeleton", str(result) if result else None)
    return {"outputs": {"pose_skeleton": str(result) if result else None}, "artifacts": [str(result)] if result else []}
