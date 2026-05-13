"""Run base_generation directly from pure text prompt (local or SeedDream4 API)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anime_pipeline_graph.config import AppConfig
from anime_pipeline_graph.providers.flux_base_provider import FluxBaseProvider
from anime_pipeline_graph.providers.seedream4_api_provider import SeedDream4ApiProvider
from anime_pipeline_graph.utils.io import ensure_dir

DEFAULT_PROMPT = "anime girl standing in cyberpunk city street at night, cinematic light, dynamic composition"
DEFAULT_PROMPT_2 = "clean anime lineart, detailed face, high contrast lighting"


def _set_default_env(backend: str) -> str:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("FORCE_CPU_OFFLOAD", "1")
    b = backend.strip().lower()
    use_seedream_api = b in {"seedream4_api", "seedream_api", "ark_seedream4", "ark", "doubao_api"}
    os.environ["IMAGE_BACKEND"] = "seedream4_api" if use_seedream_api else "local"
    return os.environ["IMAGE_BACKEND"]


def _load_prompt(prompt: str, prompt_file: str) -> str:
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8").strip()
    return prompt.strip()


def _make_provider(config: AppConfig, backend: str):
    if backend == "seedream4_api":
        return SeedDream4ApiProvider(
            model_name=config.ark_seedream_model,
            api_key=config.ark_api_key,
            base_url=config.ark_base_url,
            endpoint=config.ark_images_endpoint,
            timeout_seconds=config.ark_timeout_seconds,
            response_format=config.ark_response_format,
            default_size=config.ark_default_size,
            min_pixels=config.ark_min_pixels,
        )
    return FluxBaseProvider(config.flux_base_model, config.hf_token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Base generation from text prompt only (no LoRA).")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Main text prompt.")
    parser.add_argument("--prompt-file", type=str, default="", help="UTF-8 txt file for prompt. Overrides --prompt.")
    parser.add_argument("--prompt-2", type=str, default=DEFAULT_PROMPT_2, help="Secondary prompt text.")
    parser.add_argument(
        "--backend",
        type=str,
        default=os.getenv("IMAGE_BACKEND", "local"),
        help="local | seedream4_api (also supports ark/doubao aliases)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=1216)
    args = parser.parse_args()

    backend = _set_default_env(args.backend)
    config = AppConfig()
    provider = _make_provider(config, backend)

    prompt = _load_prompt(args.prompt, args.prompt_file)
    if not prompt:
        raise ValueError("Prompt is empty. Please pass --prompt or --prompt-file.")

    run_id = f"base_generation_{uuid.uuid4().hex[:8]}"
    run_dir = ensure_dir(PROJECT_ROOT / "runs_baseline" / run_id)
    image_dir = ensure_dir(run_dir / "images")
    out_path = image_dir / "frame_01.png"

    prompt_pack = {
        "prompt": prompt,
        "prompt_2": args.prompt_2.strip(),
        "num_inference_steps": int(args.steps),
        "max_sequence_length": int(args.max_seq_len),
    }

    image = provider.generate(
        prompt_pack=prompt_pack,
        seed=args.seed if args.seed >= 0 else None,
        size=(int(args.width), int(args.height)),
    )
    image.save(out_path)

    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "backend": backend,
        "no_lora": True,
        "prompt": prompt,
        "prompt_2": args.prompt_2.strip(),
        "seed": args.seed,
        "steps": args.steps,
        "max_sequence_length": args.max_seq_len,
        "size": [args.width, args.height],
        "image": str(out_path),
    }
    (run_dir / "run_record.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
