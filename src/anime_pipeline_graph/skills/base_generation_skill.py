"""Base generation skill."""

from __future__ import annotations

import os
import re
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - fallback for dry-run environments
    class _TorchStub:
        OutOfMemoryError = RuntimeError

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def empty_cache() -> None:
                return None

    torch = _TorchStub()  # type: ignore[assignment]


def execute(step, store, ctx):
    """Generate base images from all pre-built conditions."""
    prompt_pack = store.get("prompt_pack")
    spec = store.get("task_spec")
    bundle = store.get("input_bundle")
    if not isinstance(prompt_pack, dict):
        # Fallback when planner omitted prompt_pack_builder.
        frame_desc = []
        if spec and getattr(spec, "frame_descriptions", None):
            frame_desc = list(spec.frame_descriptions)
        if not frame_desc:
            user_text = getattr(bundle, "user_text", "anime character portrait")
            frame_desc = [user_text]
        num_frames = max(1, int(getattr(spec, "num_frames", len(frame_desc) or 1)))
        if len(frame_desc) < num_frames:
            frame_desc.extend([frame_desc[-1]] * (num_frames - len(frame_desc)))
        frame_prompts = frame_desc[:num_frames]
        prompt_pack = {
            "prompt": frame_prompts[0],
            "prompt_2": frame_prompts[0],
            "positive_prompt": frame_prompts[0],
            "frame_prompts": frame_prompts,
            "frame_prompt_2": frame_prompts,
            "frame_prompt_payloads": [{"prompt": p, "prompt_2": p} for p in frame_prompts],
            "frame_key_characters": list(getattr(spec, "character_names", []))[:1] * num_frames,
            "frame_present_characters": [list(getattr(spec, "character_names", [])) for _ in range(num_frames)],
            "continuity_blocks": [
                {
                    "previous_frame_summary": frame_prompts[i - 1] if i > 0 else "",
                    "must_keep_scene_layout": i > 0,
                    "allowed_delta": "character pose and camera shift only",
                }
                for i in range(num_frames)
            ],
            "num_frames": num_frames,
            "num_inference_steps": 12,
            "max_sequence_length": 256,
        }
        store.set("prompt_pack", prompt_pack)
    out_dir = store.images_dir() / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_prompts = prompt_pack.get("frame_prompts", [])
    frame_prompt_2 = prompt_pack.get("frame_prompt_2", [])
    frame_prompt_payloads = prompt_pack.get("frame_prompt_payloads", [])
    frame_key_characters = prompt_pack.get("frame_key_characters", [])
    frame_present_characters = prompt_pack.get("frame_present_characters", [])
    continuity_blocks = prompt_pack.get("continuity_blocks", [])
    scene_refs = list(bundle.scene_references.values()) if bundle else []
    char_ref_map = bundle.character_references if bundle else {}
    char_refs = list(char_ref_map.values()) if bundle else []
    safe_mode = os.getenv("REAL_SAFE_MODE", "").strip() == "1"
    force_base_only = os.getenv("FORCE_BASE_ONLY", "").strip() == "1"
    force_no_kontext = os.getenv("FORCE_NO_KONTEXT", "").strip() == "1"

    char_anchor = store.get("char_anchor", {}) or {}
    lora_path = char_anchor.get("lora_path")
    lora_map = char_anchor.get("lora_map", {}) if isinstance(char_anchor, dict) else {}
    has_lora = bool(lora_path)
    lora_per_frame = os.getenv("LORA_PER_FRAME", "1").strip() == "1"
    single_lora_per_frame = os.getenv("SINGLE_LORA_PER_FRAME", "0").strip() == "1"
    # Default profile is conservative for stability.
    high_quality_profile = os.getenv("REAL_HIGH_QUALITY", "").strip() == "1"
    conservative_size = (640, 896) if high_quality_profile and not safe_mode else (576, 832)
    conservative_steps = int(prompt_pack.get("num_inference_steps", 12))
    max_sequence_length = int(prompt_pack.get("max_sequence_length", 256))
    kontext_strength_first = float(os.getenv("KONTEXT_STRENGTH_FIRST", "0.62"))
    kontext_strength_chain = float(os.getenv("KONTEXT_STRENGTH_CHAIN", "0.78"))
    relax_prev_ref_on_motion = os.getenv("RELAX_PREV_REF_ON_MOTION", "0").strip() == "1"

    def _is_motion_critical_frame(payload: dict) -> bool:
        """Detect frames where action change should override strict previous-frame anchoring."""
        text = f"{payload.get('prompt', '')} {payload.get('prompt_2', '')}".lower()
        # Frame labels like "frame 2/5" can create false positives if only "2" matches.
        text = re.sub(r"\bframe\s+\d+/\d+\b", " ", text)
        keywords = (
            "fall",
            "falling",
            "drop",
            "dropped",
            "descend",
            "descending",
            "hit",
            "hits",
            "strik",
            "impact",
            "thud",
            "collision",
            "contact",
        )
        return any(k in text for k in keywords)

    def _is_scene_change_frame(payload: dict) -> bool:
        """Detect frames that should not be anchored to the previous generated image."""
        text = f"{payload.get('prompt', '')} {payload.get('prompt_2', '')}".lower()
        markers = (
            "transition: scene_change",
            "scene_change",
            "scene shifts",
            "scene changes",
            "space changes",
            "空间切换",
            "空间跳转",
            "空间发生跳跃",
            "场景切换",
        )
        return any(marker in text for marker in markers)

    def _release_kontext_gpu() -> None:
        """Release Kontext pipeline from GPU before fallback."""
        pipe = getattr(ctx.kontext_provider, "pipe", None)
        if pipe is not None:
            try:
                pipe.to("cpu")
            except Exception:
                pass
            # Drop the pipe reference to avoid hidden GPU re-allocation in hooks.
            try:
                ctx.kontext_provider.pipe = None
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _pick_frame_loras(i: int) -> list[str]:
        if not has_lora:
            return []
        present = frame_present_characters[i] if i < len(frame_present_characters) else []
        paths = [lora_map[name] for name in present if name in lora_map]
        if paths:
            dedup = list(dict.fromkeys(paths))
            if single_lora_per_frame:
                return [dedup[0]]
            return dedup
        key_char = frame_key_characters[i] if i < len(frame_key_characters) else None
        if lora_per_frame and key_char and key_char in lora_map:
            return [lora_map[key_char]]
        if key_char and key_char in lora_map:
            return [lora_map[key_char]]
        if spec.num_frames <= 1 and lora_path:
            return [lora_path]
        return []

    def _build_frame_payload(i: int) -> dict:
        base_payload = dict(prompt_pack)
        if i < len(frame_prompt_payloads) and isinstance(frame_prompt_payloads[i], dict):
            base_payload["prompt"] = frame_prompt_payloads[i].get("prompt", base_payload.get("prompt", ""))
            base_payload["prompt_2"] = frame_prompt_payloads[i].get("prompt_2", base_payload.get("prompt_2", ""))
        else:
            if i < len(frame_prompts):
                base_payload["prompt"] = frame_prompts[i]
            if i < len(frame_prompt_2):
                base_payload["prompt_2"] = frame_prompt_2[i]
        base_payload["positive_prompt"] = base_payload.get("prompt", base_payload.get("positive_prompt", ""))
        base_payload["num_inference_steps"] = conservative_steps
        base_payload["max_sequence_length"] = max_sequence_length
        return base_payload

    outputs = []
    per_frame_meta = []
    prev_frame_path: Path | None = None
    for i in range(spec.num_frames):
        frame_payload = _build_frame_payload(i)
        continuity = continuity_blocks[i] if i < len(continuity_blocks) and isinstance(continuity_blocks[i], dict) else {}
        frame_loras = _pick_frame_loras(i)
        frame_lora = frame_loras[0] if frame_loras else None
        key_char = frame_key_characters[i] if i < len(frame_key_characters) else None

        # Chain strategy: frame1 uses scene reference; frame2+ uses previous generated frame.
        ref: Path | None = None
        used_prev_frame = None
        used_scene_ref = None
        key_char_ref = None
        support_char_ref = None
        frame_refs: list[Path] = []
        if i == 0:
            if scene_refs:
                ref = Path(scene_refs[0])
                used_scene_ref = str(ref)
            elif char_refs:
                ref = Path(char_refs[0])
                used_scene_ref = str(ref)
        elif prev_frame_path is not None and prev_frame_path.exists():
            ref = prev_frame_path
            used_prev_frame = str(prev_frame_path)
        if scene_refs:
            scene_ref = Path(scene_refs[0])
            frame_refs.append(scene_ref)
            if not used_scene_ref:
                used_scene_ref = str(scene_ref)
        key_ref_path: Path | None = None
        support_ref_path: Path | None = None
        if key_char and key_char in char_ref_map:
            key_char_ref = str(char_ref_map[key_char])
            key_ref_path = Path(char_ref_map[key_char])
            frame_refs.append(key_ref_path)
        present_names = frame_present_characters[i] if i < len(frame_present_characters) else []
        support_ref_paths: list[Path] = []
        for cname in present_names:
            if cname == key_char:
                continue
            if cname in char_ref_map:
                support_ref_paths.append(Path(char_ref_map[cname]))
        # Fallback for older prompt packs without frame_present_characters.
        if not support_ref_paths:
            for cname in spec.character_names:
                if cname == key_char:
                    continue
                if cname in char_ref_map:
                    support_ref_paths.append(Path(char_ref_map[cname]))
                    break
        if support_ref_paths:
            support_ref_path = support_ref_paths[0]
            support_char_ref = str(support_ref_path)
            frame_refs.extend(support_ref_paths)

        used_backend_mode = "base_text2img"
        retries = []
        scene_ref_path = Path(scene_refs[0]) if scene_refs else None
        primary_ref: Path | None = None
        extra_refs: list[Path] = []

        motion_critical = _is_motion_critical_frame(frame_payload)
        if i == 0:
            primary_ref = scene_ref_path or key_ref_path or support_ref_path
        else:
            if relax_prev_ref_on_motion and motion_critical:
                # For strong action beats (e.g., object falling/impact), avoid over-locking to previous frame.
                primary_ref = scene_ref_path or key_ref_path or support_ref_path
                used_prev_frame = None
            else:
                primary_ref = prev_frame_path or scene_ref_path or key_ref_path or support_ref_path

        ordered_ref_candidates = [scene_ref_path, key_ref_path, *support_ref_paths]
        for r in ordered_ref_candidates:
            if r and (primary_ref is None or r != primary_ref):
                extra_refs.append(r)
        # De-duplicate while preserving order.
        seen = set()
        dedup_extra = []
        for r in extra_refs:
            sr = str(r)
            if sr in seen:
                continue
            seen.add(sr)
            dedup_extra.append(r)
        extra_refs = dedup_extra

        if ref is not None and not force_base_only and not force_no_kontext:
            try:
                used_backend_mode = "kontext_chain_prev" if i > 0 else "kontext_ref_init"
                frame_strength = kontext_strength_chain if i > 0 else kontext_strength_first
                img = ctx.kontext_provider.render_from_reference(
                    primary_ref or ref,
                    frame_payload.get("prompt", frame_payload.get("positive_prompt", "")),
                    refs=extra_refs,
                    lora_path=frame_lora,
                    lora_paths=frame_loras,
                    prompt_2=frame_payload.get("prompt_2", ""),
                    max_sequence_length=max_sequence_length,
                    size=conservative_size,
                    num_inference_steps=conservative_steps,
                    strength=frame_strength,
                )
            except torch.OutOfMemoryError:
                _release_kontext_gpu()
                retries.append("kontext_oom_8_steps")
                try:
                    frame_strength = kontext_strength_chain if i > 0 else kontext_strength_first
                    img = ctx.kontext_provider.render_from_reference(
                        primary_ref or ref,
                        frame_payload.get("prompt", frame_payload.get("positive_prompt", "")),
                        refs=extra_refs,
                        lora_path=frame_lora,
                        lora_paths=frame_loras,
                        prompt_2=frame_payload.get("prompt_2", ""),
                        max_sequence_length=max_sequence_length,
                        size=conservative_size,
                        num_inference_steps=6,
                        strength=frame_strength,
                    )
                    used_backend_mode = "kontext_retry_6"
                except torch.OutOfMemoryError:
                    retries.append("kontext_oom_6_steps")
                    force_base_only = True
                    ctx.base_provider.apply_optional_loras(frame_loras)
                    fallback_pack = dict(frame_payload)
                    fallback_pack["num_inference_steps"] = 6
                    img = ctx.base_provider.generate(fallback_pack, size=conservative_size)
                    used_backend_mode = "base_fallback_after_kontext_oom"
        else:
            ctx.base_provider.apply_optional_loras(frame_loras)
            fallback_pack = dict(frame_payload)
            fallback_pack["num_inference_steps"] = conservative_steps
            # Keep reference-chain logic in API/base fallback as well.
            api_refs: dict[str, Path] = {}
            if primary_ref is not None:
                api_refs["primary"] = primary_ref
            for ridx, r in enumerate(extra_refs):
                api_refs[f"extra_{ridx+1}"] = r
            img = ctx.base_provider.generate(
                fallback_pack,
                refs=api_refs if api_refs else None,
                size=conservative_size,
            )
            used_backend_mode = "base_forced"
        out = out_dir / f"frame_{i+1}.png"
        img.save(out)
        outputs.append(str(out))
        prev_frame_path = out
        per_frame_meta.append(
            {
                "frame_index": i + 1,
                "used_prev_frame": used_prev_frame,
                "used_scene_ref": used_scene_ref,
                "used_key_char_ref": key_char_ref,
                "used_support_char_ref": support_char_ref,
                "used_lora": frame_loras,
                "used_backend_mode": used_backend_mode,
                "used_primary_reference": str(primary_ref) if primary_ref else None,
                "used_extra_references": [str(x) for x in extra_refs],
                "retries": retries,
                "size": [conservative_size[0], conservative_size[1]],
                "num_inference_steps": conservative_steps,
                "max_sequence_length": max_sequence_length,
            }
        )

    store.set("generated_images", outputs)
    return {"outputs": {"generated_images": outputs}, "artifacts": outputs, "metadata": {"per_frame": per_frame_meta}}
