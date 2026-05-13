"""Run full pipeline with auto asset/profile/LoRA matching from text prompt."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anime_pipeline_graph.cli.main import run_pipeline


def _set_runtime_env(backend: str, output_dir: str) -> str:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("FORCE_CPU_OFFLOAD", "1")
    if output_dir.strip():
        os.environ["DEFAULT_OUTPUT_DIR"] = output_dir.strip()

    b = backend.strip().lower()
    use_seedream_api = b in {"seedream4_api", "seedream_api", "ark_seedream4", "ark", "doubao_api"}
    if use_seedream_api:
        os.environ["IMAGE_BACKEND"] = "seedream4_api"
        os.environ["FORCE_NO_KONTEXT"] = "1"
        os.environ["FORCE_BASE_ONLY"] = "1"
        # API mode does not support local LoRA injection.
        os.environ["SINGLE_LORA_PER_FRAME"] = "0"
        os.environ["LORA_PER_FRAME"] = "0"
    else:
        os.environ["IMAGE_BACKEND"] = "local"
        os.environ.pop("FORCE_NO_KONTEXT", None)
        os.environ.pop("FORCE_BASE_ONLY", None)
    return os.environ["IMAGE_BACKEND"]


def _load_prompt(prompt: str, prompt_file: str) -> str:
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8").strip()
    return prompt.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run full anime pipeline (parser/planner/graph/executor). "
            "Auto-uses character refs/profile/LoRA when matched; otherwise falls back to normal text2img."
        )
    )
    parser.add_argument("--prompt", type=str, default="", help="Input story prompt.")
    parser.add_argument("--prompt-file", type=str, default="", help="UTF-8 txt file for prompt.")
    parser.add_argument(
        "--backend",
        type=str,
        default=os.getenv("IMAGE_BACKEND", "local"),
        help="local | seedream4_api (also supports ark/doubao aliases)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("DEFAULT_OUTPUT_DIR", "runs"),
        help="Run output dir, relative or absolute. Example: runs / Anime_one / /abs/path/to/runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock providers/clients instead of real APIs/models.",
    )
    args = parser.parse_args()

    prompt = _load_prompt(args.prompt, args.prompt_file)
    if not prompt:
        raise ValueError("Prompt is empty. Please pass --prompt or --prompt-file.")

    active_backend = _set_runtime_env(args.backend, args.output_dir)
    record = run_pipeline(prompt, dry_run=args.dry_run, project_root=PROJECT_ROOT)

    print(f"backend={active_backend}")
    print(f"run_id={record.run_id}")
    print(f"run_dir={record.run_dir}")
    print(record.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
