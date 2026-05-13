"""Edit skill."""

from __future__ import annotations

from pathlib import Path


def execute(step, store, ctx):
    """Run no-mask edit with Kontext provider."""
    generated = store.get("generated_images", [])
    spec = store.get("task_spec")
    source_image = step.params.get("source_image") or (getattr(spec, "source_image", None) if spec else None)
    if not generated and source_image:
        generated = [source_image]
    instruction = step.params.get("instruction", "")
    if not instruction:
        frame_desc = getattr(spec, "frame_descriptions", []) if spec else []
        instruction = frame_desc[0] if frame_desc else "polish details and improve anime style"
    if not generated:
        return {"outputs": {"edited_images": []}, "artifacts": []}
    out_dir = store.images_dir() / "edited"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply available character LoRA for identity-consistent edits.
    char_anchor = store.get("char_anchor", {}) or {}
    lora_map = char_anchor.get("lora_map", {}) if isinstance(char_anchor, dict) else {}
    lora_path = char_anchor.get("lora_path") if isinstance(char_anchor, dict) else None
    loras = list(dict.fromkeys([v for v in lora_map.values() if v])) or ([lora_path] if lora_path else [])
    if loras:
        try:
            ctx.kontext_provider.apply_optional_loras(loras)
        except Exception:
            pass

    edited = []
    for i, image_path in enumerate(generated):
        img = ctx.kontext_provider.edit_image(Path(image_path), instruction)
        out = out_dir / f"edited_{i+1}.png"
        img.save(out)
        edited.append(str(out))

    store.set("generated_images", edited)
    return {"outputs": {"edited_images": edited}, "artifacts": edited}
