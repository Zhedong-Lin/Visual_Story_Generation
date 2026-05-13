"""Run dry demo with one editable PROMPT variable."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from anime_pipeline_graph.cli.main import run_pipeline

PROMPT = "lulu在cyber_city被jiddo追"


if __name__ == "__main__":
    record = run_pipeline(PROMPT, dry_run=True, project_root=PROJECT_ROOT)
    print(record.model_dump_json(indent=2))
