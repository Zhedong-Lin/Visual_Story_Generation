"""Story demo script focusing on multi-frame output."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from anime_pipeline_graph.cli.main import run_pipeline

PROMPT = "Generate a 3-panel anime storyboard using my scene reference image.Frame 1: Lulu is near a roadside bench when Jiddo, an elderly old man, gives her a flyer. Frame 2: Lulu realizes Jiddo is dangerous, and Jiddo starts chasing Lulu. Frame 3: Lulu escapes to an empty area with no people around"


if __name__ == "__main__":
    record = run_pipeline(PROMPT, dry_run=True, project_root=PROJECT_ROOT)
    print("num_frames:", record.task_spec.num_frames)
    print("frame_descriptions:", record.task_spec.frame_descriptions)
    print(record.model_dump_json(indent=2))
