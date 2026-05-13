"""Run baseline with prompt-handling fixes only (no core module changes)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anime_pipeline_graph.constants import MAX_STORYBOARD_FRAMES
from anime_pipeline_graph.config import AppConfig
from anime_pipeline_graph.providers.flux_base_provider import FluxBaseProvider
from anime_pipeline_graph.providers.flux_kontext_provider import FluxKontextProvider
from anime_pipeline_graph.providers.seedream4_api_provider import SeedDream4ApiProvider
from anime_pipeline_graph.utils.io import ensure_dir


DEFAULT_PROMPT = (
    "Lulu is walking on the street in cyber_city. Dark clouds gather, lightning flashes, and rain begins. Lulu looks frightened and reacts to the sudden storm. Then Lulu is running through the rain in panic. Lulu notices her home in the distance and runs toward it. She reaches her house, soaked and breathing heavily, and quickly steps inside. Under the warm indoor lighting, she bends over slightly, catching her breath, water dripping from her hair and clothes. After calming down, Lulu walks into her room and looks for clean clothes. She finds a set of fresh white clothes and changes into them. Now dressed in clean white clothing, Lulu looks relaxed and smiles softly, feeling safe and comfortable inside her home."
)


def _set_default_env() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("FORCE_CPU_OFFLOAD", "1")


def _load_lora_map(project_root: Path) -> dict[str, str]:
    yaml_path = project_root / "examples" / "lora_map.yaml"
    json_path = project_root / "examples" / "lora_map.json"
    if yaml_path.exists():
        import yaml

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        return {str(k).lower(): str(v) for k, v in data.items()}
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k).lower(): str(v) for k, v in data.items()}
    return {}


def _load_character_profiles(project_root: Path) -> dict[str, dict[str, str]]:
    yaml_path = project_root / "examples" / "character_profiles.yaml"
    json_path = project_root / "examples" / "character_profiles.json"
    if yaml_path.exists():
        import yaml

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    return {}


def _known_character_names(project_root: Path) -> list[str]:
    chars_dir = project_root / "examples" / "assets" / "characters"
    if not chars_dir.exists():
        return []
    return sorted({p.stem.lower() for p in chars_dir.iterdir() if p.is_file()})


def _known_scene_names(project_root: Path) -> list[str]:
    scenes_dir = project_root / "examples" / "assets" / "scenes"
    if not scenes_dir.exists():
        return []
    return sorted({p.stem.lower() for p in scenes_dir.iterdir() if p.is_file()})


def _extract_characters(prompt: str, names: list[str]) -> list[str]:
    text = prompt.lower()
    found: list[str] = []
    for name in names:
        if re.search(rf"\b{re.escape(name)}\b", text):
            found.append(name)
    return found


def _extract_scenes(prompt: str, names: list[str]) -> list[str]:
    text = prompt.lower()
    found: list[str] = []
    for name in names:
        if re.search(rf"\b{re.escape(name)}\b", text):
            found.append(name)
    return found


def _collect_character_refs(project_root: Path, found_chars: list[str]) -> list[Path]:
    chars_dir = project_root / "examples" / "assets" / "characters"
    valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    refs: list[Path] = []
    if not chars_dir.exists():
        return refs
    for name in found_chars:
        for p in chars_dir.iterdir():
            if p.is_file() and p.suffix.lower() in valid_ext and p.stem.lower() == name:
                refs.append(p)
                break
    return refs


def _collect_scene_refs(project_root: Path, found_scenes: list[str]) -> list[Path]:
    scenes_dir = project_root / "examples" / "assets" / "scenes"
    valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    refs: list[Path] = []
    if not scenes_dir.exists():
        return refs
    for name in found_scenes:
        for p in scenes_dir.iterdir():
            if p.is_file() and p.suffix.lower() in valid_ext and p.stem.lower() == name:
                refs.append(p)
                break
    return refs


def _detect_target_frames(prompt: str, default_n: int = 3) -> int:
    text = prompt.lower()
    for pat in [
        r"\b(\d+)\s*-\s*panel\b",
        r"\b(\d+)\s*panels?\b",
        r"\b(\d+)\s*-\s*frame\b",
        r"\b(\d+)\s*frames?\b",
    ]:
        m = re.search(pat, text)
        if m:
            try:
                return max(1, min(int(m.group(1)), MAX_STORYBOARD_FRAMES))
            except ValueError:
                pass
    return max(1, default_n)


def _clean_meta_text(text: str) -> str:
    t = text.strip()
    # Remove only the leading meta phrase; keep the story body.
    t = re.sub(
        r"(?is)^\s*generate\s+a?\s*\d+\s*-\s*panel\s+anime\s+storyboard\b\.?\s*",
        "",
        t,
        count=1,
    )
    # Remove scene-reference meta phrase without deleting nearby story sentences.
    t = re.sub(r"(?i)\busing my scene reference image\s+\w+\b\.?\s*", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _split_frames_exact(prompt: str, target_frames: int) -> list[str]:
    cleaned = _clean_meta_text(prompt)
    markers = list(re.finditer(r"frame\s*\d+\s*:", cleaned, flags=re.IGNORECASE))
    chunks: list[str] = []
    if markers:
        for i, marker in enumerate(markers):
            start = marker.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(cleaned)
            chunk = cleaned[start:end].strip(" .\n\t")
            if chunk:
                chunks.append(chunk)
    else:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        if not sents:
            sents = [cleaned] if cleaned else ["anime character in scene"]
        bucket_size = max(1, (len(sents) + target_frames - 1) // target_frames)
        for i in range(0, len(sents), bucket_size):
            chunks.append(" ".join(sents[i : i + bucket_size]).strip())

    # Force exact frame count.
    if len(chunks) < target_frames:
        last = chunks[-1] if chunks else "story continuation"
        while len(chunks) < target_frames:
            chunks.append(f"{last}. continuation shot {len(chunks)+1}")
    elif len(chunks) > target_frames:
        chunks = chunks[: target_frames - 1] + [" ".join(chunks[target_frames - 1 :])]
    return chunks


def _first_words(text: str, n: int) -> str:
    words = text.replace("\n", " ").split()
    return " ".join(words[:n])


def _compact_profile(name: str, profiles: dict[str, dict[str, str]]) -> str:
    data = profiles.get(name, {}) if isinstance(profiles, dict) else {}
    appearance = _first_words(str(data.get("appearance", "")).replace(",", " "), 8)
    outfit = _first_words(str(data.get("outfit", "")).replace(",", " "), 8)
    return f"{name}: {appearance}; {outfit}".strip()


def _build_prompt_pack(
    frame_text: str,
    frame_idx: int,
    total: int,
    chars: list[str],
    profiles: dict[str, dict[str, str]],
) -> dict:
    char_text = ", ".join(chars) if chars else "main character"
    profile_text = " | ".join(_compact_profile(c, profiles) for c in chars) if chars else ""

    # Keep CLIP-facing prompt short (<77 tokens safety margin).
    short_prompt = (
        f"anime frame {frame_idx}/{total}, {frame_text}. "
        f"show {char_text} clearly, face visible, same outfit, rain mood, dynamic cinematic."
    )
    short_prompt = _first_words(short_prompt, 52)

    long_prompt = (
        f"storyboard frame {frame_idx}/{total}. goal: {frame_text}. "
        f"characters: {char_text}. identity lock: {profile_text}. "
        "keep same face and clothing as reference."
    )
    long_prompt = _first_words(long_prompt, 120)

    return {
        "prompt": short_prompt,
        "prompt_2": long_prompt,
        "negative_prompt": "blurry, low quality, wrong face, different outfit, empty scene, text watermark",
        "num_inference_steps": int(os.getenv("BASELINE_STEPS", "12")),
        "max_sequence_length": int(os.getenv("BASELINE_MAX_SEQ_LEN", "256")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-fixed baseline generation.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA loading for baseline run.")
    parser.add_argument(
        "--backend",
        type=str,
        default=os.getenv("IMAGE_BACKEND", "local"),
        help="Image backend: local / seedream4_api (also accepts ark/doubao_api aliases).",
    )
    args = parser.parse_args()

    _set_default_env()
    config = AppConfig()

    run_id = f"baseline_promptfix_{uuid.uuid4().hex[:8]}"
    run_dir = ensure_dir(PROJECT_ROOT / "runs_baseline" / run_id)
    image_dir = ensure_dir(run_dir / "images")

    prompt = args.prompt.strip()
    target_frames = _detect_target_frames(prompt, default_n=3)
    known_chars = _known_character_names(PROJECT_ROOT)
    known_scenes = _known_scene_names(PROJECT_ROOT)
    found_chars = _extract_characters(prompt, known_chars)
    found_scenes = _extract_scenes(prompt, known_scenes)
    profiles = _load_character_profiles(PROJECT_ROOT)
    frame_texts = _split_frames_exact(prompt, target_frames=target_frames)
    char_refs = _collect_character_refs(PROJECT_ROOT, found_chars)
    scene_refs = _collect_scene_refs(PROJECT_ROOT, found_scenes)

    lora_map = _load_lora_map(PROJECT_ROOT) if not args.no_lora else {}
    active_loras = [lora_map[c] for c in found_chars if c in lora_map]
    active_loras = list(dict.fromkeys(active_loras))
    active_lora = active_loras[0] if active_loras else None

    backend = args.backend.strip().lower()
    use_seedream_api = backend in {"seedream4_api", "seedream_api", "ark_seedream4", "ark", "doubao_api"}
    if use_seedream_api:
        # API path does not support local LoRA injection.
        active_loras = []
        active_lora = None
    if use_seedream_api:
        # API mode is text2img only in this baseline script.
        os.environ["FORCE_NO_KONTEXT"] = "1"
        os.environ["FORCE_BASE_ONLY"] = "1"
        base_provider = SeedDream4ApiProvider(
            model_name=config.ark_seedream_model,
            api_key=config.ark_api_key,
            base_url=config.ark_base_url,
            endpoint=config.ark_images_endpoint,
            timeout_seconds=config.ark_timeout_seconds,
            response_format=config.ark_response_format,
            default_size=config.ark_default_size,
            min_pixels=config.ark_min_pixels,
        )
        kontext_provider = None
    else:
        # Use same model family as main system: base + kontext.
        base_provider = FluxBaseProvider(config.flux_base_model, config.hf_token)
        kontext_provider = FluxKontextProvider(config.flux_kontext_model, config.hf_token)

    if active_loras and not use_seedream_api:
        base_provider.apply_optional_loras(active_loras)

    images: list[str] = []
    frame_prompts: list[dict] = []
    generation_mode = "text2img_base_only"
    prev_frame: Path | None = None
    primary_scene_ref = scene_refs[0] if scene_refs else None
    for i, text in enumerate(frame_texts):
        pack = _build_prompt_pack(text, i + 1, len(frame_texts), found_chars, profiles)
        frame_prompts.append(pack)

        # Baseline stack input: prompt + scene ref + character refs + LoRA.
        primary_ref = primary_scene_ref if i == 0 else (prev_frame or primary_scene_ref)
        extra_refs: list[Path] = []
        if primary_scene_ref is not None:
            extra_refs.append(primary_scene_ref)
        extra_refs.extend(char_refs)
        dedup_refs = []
        seen = set()
        for r in extra_refs:
            k = str(r)
            if k in seen:
                continue
            seen.add(k)
            if primary_ref is None or r != primary_ref:
                dedup_refs.append(r)

        if (primary_ref is not None) and (not use_seedream_api):
            generation_mode = "kontext_ref_plus_lora"
            img = kontext_provider.render_from_reference(
                reference_image_path=primary_ref,
                instruction=pack["prompt"],
                refs=dedup_refs,
                lora_paths=active_loras if active_loras else None,
                prompt_2=pack.get("prompt_2", ""),
                max_sequence_length=pack.get("max_sequence_length", 256),
                num_inference_steps=pack.get("num_inference_steps", 12),
                size=(576, 832),
                strength=0.72 if i > 0 else 0.62,
            )
        else:
            if use_seedream_api:
                generation_mode = "seedream4_api_text2img"
                api_refs: dict[str, Path] = {}
                if primary_ref is not None:
                    api_refs["primary"] = primary_ref
                for ridx, r in enumerate(dedup_refs):
                    api_refs[f"extra_{ridx+1}"] = r
                img = base_provider.generate(
                    pack,
                    refs=api_refs if api_refs else None,
                    seed=args.seed + i,
                    size=(576, 832),
                )
            else:
                img = base_provider.generate(pack, seed=args.seed + i, size=(576, 832))

        out = image_dir / f"frame_{i+1:02d}.png"
        img.save(out)
        images.append(str(out))
        prev_frame = out

    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "baseline": True,
        "prompt_fix_only": True,
        "prompt": prompt,
        "target_frames": target_frames,
        "num_frames": len(frame_texts),
        "detected_characters": found_chars,
        "detected_scenes": found_scenes,
        "active_lora": active_lora,
        "active_loras": active_loras,
        "lora_enabled": (not args.no_lora) and (not use_seedream_api),
        "backend": "seedream4_api" if use_seedream_api else "local",
        "character_references": [str(p) for p in char_refs],
        "scene_references": [str(p) for p in scene_refs],
        "generation_mode": generation_mode,
        "frame_texts": frame_texts,
        "frame_prompts": frame_prompts,
        "images": images,
        "notes": "Baseline prompt+refs+lora: exact frame count + CLIP-safe prompt, no skill graph.",
    }
    (run_dir / "run_record.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"target_frames={target_frames}")
    print(f"num_frames={len(frame_texts)}")
    print(f"detected_characters={found_chars}")
    print(f"active_lora={active_lora}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
