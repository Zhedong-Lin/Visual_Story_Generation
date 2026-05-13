"""Fill edit skill with edit fallback."""

from __future__ import annotations

from pathlib import Path

from anime_pipeline_graph.providers.flux_fill_provider import MissingMaskError


def execute(step, store, ctx):
    """Run masked edit; fallback to no-mask edit when mask missing."""
    generated = store.get("generated_images", [])
    spec = store.get("task_spec")
    source_image = step.params.get("source_image") or (getattr(spec, "source_image", None) if spec else None)
    if not generated and source_image:
        generated = [source_image]
    mask_path = step.params.get("mask_path")
    instruction = step.params.get("instruction", "")
    if not instruction:
        frame_desc = getattr(spec, "frame_descriptions", []) if spec else []
        instruction = frame_desc[0] if frame_desc else "local detail correction"
    if not generated:
        return {"outputs": {"edited_images": []}, "artifacts": []}
    out_dir = store.images_dir() / "fill_edited"
    out_dir.mkdir(parents=True, exist_ok=True)
    edited = []

    for i, image_path in enumerate(generated):
        try:
            img = ctx.fill_provider.fill_edit(Path(image_path), Path(mask_path) if mask_path else None, instruction)
        except MissingMaskError:
            img = ctx.kontext_provider.edit_image(Path(image_path), instruction)
        out = out_dir / f"fill_edited_{i+1}.png"
        img.save(out)
        edited.append(str(out))

    store.set("generated_images", edited)
    return {"outputs": {"edited_images": edited}, "artifacts": edited}
