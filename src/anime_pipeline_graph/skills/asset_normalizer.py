"""Asset normalizer skill."""

from __future__ import annotations

from pathlib import Path

from anime_pipeline_graph.utils.images import normalize_to_png


def execute(step, store, ctx):
    """Normalize reference images into PNG format."""
    bundle = store.get("input_bundle")
    out_dir = store.images_dir() / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized = {"characters": {}, "scenes": {}}
    for name, path in bundle.character_references.items():
        out = out_dir / f"character_{name}.png"
        normalized["characters"][name] = str(normalize_to_png(Path(path), out))
    for name, path in bundle.scene_references.items():
        out = out_dir / f"scene_{name}.png"
        normalized["scenes"][name] = str(normalize_to_png(Path(path), out))

    store.set("normalized_assets", normalized)
    return {"outputs": {"normalized_assets": normalized}, "artifacts": list(normalized["characters"].values()) + list(normalized["scenes"].values())}
