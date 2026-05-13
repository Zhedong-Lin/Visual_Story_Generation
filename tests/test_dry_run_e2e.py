import json
from pathlib import Path

from anime_pipeline_graph.cli.main import run_pipeline


def test_dry_run_e2e():
    project_root = Path(__file__).resolve().parents[1]
    record = run_pipeline("lulu在cyber_city被jiddo追", dry_run=True, project_root=project_root)
    assert record.run_dir.exists()
    assert record.judge is not None
    assert len(record.graph.steps) >= 1

    # Prompt pack should include dual-channel prompts and continuity blocks.
    prompt_posts = list((record.run_dir / "steps").glob("*/post.json"))
    prompt_payload = None
    gen_payload = None
    for post in prompt_posts:
        data = json.loads(post.read_text(encoding="utf-8"))
        if data.get("skill") == "prompt_pack_builder":
            prompt_payload = data
        if data.get("skill") == "base_generation":
            gen_payload = data

    assert prompt_payload is not None
    prompt_pack = prompt_payload["outputs"]["prompt_pack"]
    assert "frame_prompt_payloads" in prompt_pack
    assert "continuity_blocks" in prompt_pack
    assert prompt_pack.get("max_sequence_length") == 256
    assert len(prompt_pack["frame_prompt_payloads"]) == record.task_spec.num_frames

    assert gen_payload is not None
    per_frame = gen_payload.get("metadata", {}).get("per_frame", [])
    assert len(per_frame) == record.task_spec.num_frames
    assert all("used_backend_mode" in item for item in per_frame)
