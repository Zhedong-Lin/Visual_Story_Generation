"""Scene condition skill."""

from __future__ import annotations


def execute(step, store, ctx):
    """Build scene condition pack from references and text."""
    bundle = store.get("input_bundle")
    spec = store.get("task_spec")
    scene_pack = {
        "scene_refs": {k: str(v) for k, v in bundle.scene_references.items()},
        "scene_names": spec.scene_names,
        "scene_constraints": spec.scene_constraints,
        "scene_strength": spec.scene_strength,
        "setting_docs": bundle.setting_docs,
    }
    store.set("scene_pack", scene_pack)
    return {"outputs": {"scene_pack": scene_pack}, "artifacts": []}
