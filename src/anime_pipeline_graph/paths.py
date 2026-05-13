"""Path helpers."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
RUNS_DIR = PROJECT_ROOT / "runs"
PROMPTS_DIR = PROJECT_ROOT / "src" / "anime_pipeline_graph" / "prompts"
