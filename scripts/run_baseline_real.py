"""Run a simple baseline pipeline (no parser/planner/graph)."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from anime_pipeline_graph.constants import MAX_STORYBOARD_FRAMES
from anime_pipeline_graph.config import AppConfig
from anime_pipeline_graph.providers.flux_base_provider import FluxBaseProvider
from anime_pipeline_graph.utils.io import ensure_dir


DEFAULT_PROMPT = (
    "Generate a 4-panel anime storyboard using my scene reference image cyber_city. Lulu is walking on the street in cyber_city. Dark clouds gather, lightning flashes, and rain begins. Lulu looks frightened and reacts to the sudden storm. Then Lulu is running through the rain in panic."
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


def _extract_characters(prompt: str, names: list[str]) -> list[str]:
    text = prompt.lower()
    found: list[str] = []
    for name in names:
        if re.search(rf"\b{re.escape(name)}\b", text):
            found.append(name)
    return found


def _split_frames(prompt: str, fallback_n: int = 3) -> list[str]:
    frame_markers = list(re.finditer(r"frame\s*\d+\s*:", prompt, flags=re.IGNORECASE))
    if frame_markers:
        chunks: list[str] = []
        for i, marker in enumerate(frame_markers):
            start = marker.end()
            end = frame_markers[i + 1].start() if i + 1 < len(frame_markers) else len(prompt)
            chunk = prompt[start:end].strip(" .\n\t")
            if chunk:
                chunks.append(chunk)
        if chunks:
            return chunks

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prompt) if s.strip()]
    if not sentences:
        return [prompt.strip()]
    # Keep chronology by contiguous chunking, do not round-robin shuffle.
    n = min(max(fallback_n, 1), len(sentences))
    out: list[str] = []
    chunk_size = max(1, (len(sentences) + n - 1) // n)
    for i in range(0, len(sentences), chunk_size):
        out.append(" ".join(sentences[i : i + chunk_size]).strip())
    return out[:n]


def _detect_target_frames(prompt: str, default_n: int = 3) -> int:
    """Detect expected frame count from prompt text."""
    text = prompt.lower()
    patterns = [
        r"\b(\d+)\s*-\s*panel\b",
        r"\b(\d+)\s*panels?\b",
        r"\b(\d+)\s*-\s*frame\b",
        r"\b(\d+)\s*frames?\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                n = int(m.group(1))
                return max(1, min(n, MAX_STORYBOARD_FRAMES))
            except ValueError:
                pass
    return max(1, default_n)


def _build_frame_prompt(
    frame_text: str,
    all_chars: list[str],
    scene_hint: str,
    character_profiles: dict[str, dict[str, str]],
) -> dict[str, Any]:
    chars_text = ", ".join(all_chars) if all_chars else "one character"
    bible_lines: list[str] = []
    for name in all_chars:
        profile = character_profiles.get(name, {})
        appearance = str(profile.get("appearance", "")).strip()
        outfit = str(profile.get("outfit", "")).strip()
        must_keep = str(profile.get("must_keep", "")).strip()
        pieces = [x for x in [appearance, outfit, must_keep] if x]
        if pieces:
            bible_lines.append(f"{name}: " + "; ".join(pieces))

    hard_presence = ""
    if len(all_chars) == 1:
        hard_presence = (
            "Must show the character clearly in frame. No empty environment-only shot. "
            "single human character, full body or half body visible, face readable."
        )
    elif len(all_chars) > 1:
        hard_presence = (
            f"Must show all named characters ({chars_text}) in frame. "
            "No empty environment-only shot."
        )

    bible_text = " ".join(bible_lines)
    positive = (
        f"Anime cinematic frame. {frame_text} "
        f"Characters: {chars_text}. "
        f"{hard_presence} "
        f"Character bible: {bible_text} "
        "Keep identity and outfit consistent. Clean lineart, detailed face, dynamic lighting."
    )
    negative = (
        "low quality, blurry, extra limbs, bad anatomy, text, watermark, logo, "
        "wrong face, different outfit, duplicate person, empty street without people"
    )
    prompt_2 = scene_hint.strip()
    return {
        "prompt": positive,
        "prompt_2": prompt_2,
        "negative_prompt": negative,
        "num_inference_steps": int(os.getenv("BASELINE_STEPS", "12")),
        "max_sequence_length": int(os.getenv("BASELINE_MAX_SEQ_LEN", "256")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline generation without dynamic planner.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _set_default_env()
    config = AppConfig()

    run_id = f"baseline_{uuid.uuid4().hex[:8]}"
    run_dir = ensure_dir(PROJECT_ROOT / "runs_baseline" / run_id)
    image_dir = ensure_dir(run_dir / "images")

    prompt = args.prompt.strip()
    known_chars = _known_character_names(PROJECT_ROOT)
    character_profiles = _load_character_profiles(PROJECT_ROOT)
    found_chars = _extract_characters(prompt, known_chars)
    target_frames = _detect_target_frames(prompt, default_n=3)
    frame_texts = _split_frames(prompt, fallback_n=target_frames)

    lora_map = _load_lora_map(PROJECT_ROOT)
    active_lora = lora_map.get(found_chars[0]) if found_chars else None

    # Baseline: use only FLUX base generation, optional single LoRA for the first detected character.
    provider = FluxBaseProvider(config.flux_base_model, config.hf_token)
    if active_lora:
        provider.apply_optional_lora(active_lora)

    results: list[str] = []
    frame_prompts: list[dict[str, Any]] = []
    scene_hint = "cyberpunk city, anime background"
    base_seed = args.seed if args.seed >= 0 else random.randint(0, 2**31 - 1)

    for i, frame_text in enumerate(frame_texts):
        prompt_pack = _build_frame_prompt(frame_text, found_chars, scene_hint, character_profiles)
        frame_prompts.append(prompt_pack)
        img = provider.generate(prompt_pack, seed=base_seed + i)
        out = image_dir / f"frame_{i+1:02d}.png"
        img.save(out)
        results.append(str(out))

    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "baseline": True,
        "prompt": prompt,
        "num_frames": len(frame_texts),
        "target_frames": target_frames,
        "detected_characters": found_chars,
        "active_lora": active_lora,
        "images": results,
        "frame_texts": frame_texts,
        "frame_prompts": frame_prompts,
        "notes": "No parser/planner/skill-graph. Independent frame generation with one optional LoRA.",
    }
    (run_dir / "run_record.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"num_frames={len(frame_texts)}")
    print(f"detected_characters={found_chars}")
    print(f"active_lora={active_lora}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
