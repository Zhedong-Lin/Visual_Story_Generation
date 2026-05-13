"""Run real mode for single-character single-LoRA storyboard test."""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anime_pipeline_graph.cli.main import run_pipeline

PROMPT = (
    "Lulu is walking on the street in cyber_city. Dark clouds gather, lightning flashes, and rain begins. Lulu looks frightened and reacts to the sudden storm. Then Lulu is running through the rain in panic. Lulu notices her home in the distance and runs toward it. She reaches her house, soaked and breathing heavily, and quickly steps inside. Under the warm indoor lighting, she bends over slightly, catching her breath, water dripping from her hair and clothes. After calming down, Lulu walks into her room and looks for clean clothes. She finds a set of fresh white clothes and changes into them. Now dressed in clean white clothing, Lulu looks relaxed and smiles softly, feeling safe and comfortable inside her home."
    )


def _set_default_env(backend: str = "local") -> None:
    """Apply stable defaults for single-LoRA single-character testing."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("FORCE_CPU_OFFLOAD", "1")
    b = backend.strip().lower()
    use_seedream_api = b in {"seedream4_api", "seedream_api", "ark_seedream4", "ark", "doubao_api"}
    if use_seedream_api:
        os.environ["IMAGE_BACKEND"] = "seedream4_api"
        # SeedDream4 API path is text2img; disable Kontext to keep pipeline stable.
        os.environ["FORCE_NO_KONTEXT"] = "1"
        os.environ["FORCE_BASE_ONLY"] = "1"
        # API mode does not apply local LoRA.
        os.environ["SINGLE_LORA_PER_FRAME"] = "0"
        os.environ["LORA_PER_FRAME"] = "0"
    else:
        os.environ["IMAGE_BACKEND"] = "local"
        # Keep Kontext enabled unless caller explicitly disables it.
        os.environ.pop("FORCE_NO_KONTEXT", None)
        os.environ.pop("FORCE_BASE_ONLY", None)
        os.environ.setdefault("SINGLE_LORA_PER_FRAME", "1")
        os.environ.setdefault("LORA_PER_FRAME", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-LuLu storyboard in real mode.")
    parser.add_argument(
        "--backend",
        type=str,
        default=os.getenv("IMAGE_BACKEND", "local"),
        help="Image backend: local / seedream4_api (also accepts ark/doubao_api aliases).",
    )
    args = parser.parse_args()

    _set_default_env(args.backend)
    record = run_pipeline(PROMPT, dry_run=False, project_root=PROJECT_ROOT)
    print(record.model_dump_json(indent=2))
