"""Refine skill."""

from __future__ import annotations


def execute(step, store, ctx):
    """Apply a simple refine pass according to judge result."""
    report = store.get("judge_report")
    if report is None or report.final_score >= 0.85:
        return {"outputs": {"refined": False}, "artifacts": []}
    step.params["instruction"] = "improve identity and pose consistency"
    from anime_pipeline_graph.skills.edit_skill import execute as edit_execute

    out = edit_execute(step, store, ctx)
    return {"outputs": {"refined": True, **out.get("outputs", {})}, "artifacts": out.get("artifacts", [])}
