"""Judge skill."""

from __future__ import annotations

from anime_pipeline_graph.domain.models import JudgeReport


def execute(step, store, ctx):
    """Run Qwen judge and save judge report."""
    spec = store.get("task_spec")
    images = store.get("generated_images", [])
    judge_payload = {
        "task_spec": spec.model_dump(mode="json"),
        "images": images,
    }
    data = ctx.qwen_client.judge_result(judge_payload)
    source = data
    if isinstance(source, list):
        source = source[0] if source else {}
    if isinstance(source, dict):
        source = source.get("judge_report", source)
    if isinstance(source, list):
        source = source[0] if source else {}
    if not isinstance(source, dict):
        source = {}
    subs = source.get("subscores", source.get("scores", {}))
    normalized = {
        "final_score": source.get("final_score", source.get("overall_score", 0.8)),
        "subscores": {
            "instruction_match": subs.get("instruction_match", subs.get("instruction", 0.8)),
            "identity_preservation": subs.get("identity_preservation", subs.get("identity", 0.8)),
            "costume_accuracy": subs.get("costume_accuracy", subs.get("costume", 0.8)),
            "pose_accuracy": subs.get("pose_accuracy", subs.get("pose", 0.8)),
            "scene_match": subs.get("scene_match", subs.get("scene", 0.8)),
            "story_consistency": subs.get("story_consistency", subs.get("story", 0.8)),
        },
        "failure_tags": source.get("failure_tags", source.get("failures", [])),
        "repair_suggestions": source.get("repair_suggestions", source.get("suggestions", [])),
    }
    if normalized["final_score"] > 1.0:
        normalized["final_score"] = float(normalized["final_score"]) / 5.0
    report = JudgeReport.model_validate(normalized)
    store.set("judge_report", report)
    return {"outputs": {"judge_report": report.model_dump(mode="json")}, "artifacts": []}
