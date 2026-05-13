"""Repair planner."""

from __future__ import annotations

from typing import Any

from anime_pipeline_graph.domain.models import GraphPatch, JudgeReport, TaskSpec


class RepairPlanner:
    """Plan repair actions after judge failure."""

    def __init__(self, qwen_client: Any) -> None:
        self.qwen_client = qwen_client

    def _normalize_patch(self, patch: dict) -> dict:
        """Normalize various repair planner payload shapes to GraphPatch schema."""
        source = patch.get("repair_plan", patch)
        if isinstance(source, list):
            return {"reason": "normalized_from_list", "actions": source}
        if isinstance(source, dict):
            return {
                "reason": source.get("reason", source.get("summary", "normalized_from_dict")),
                "actions": source.get("actions", source.get("steps", source.get("plan", []))),
            }
        return {"reason": "empty_or_unknown_repair_payload", "actions": []}

    def plan(self, task_spec: TaskSpec, judge: JudgeReport) -> GraphPatch:
        """Return a repair patch supporting the first-version action set."""
        payload = {
            "task_spec": task_spec.model_dump(mode="json"),
            "failure_tags": judge.failure_tags,
            "repair_suggestions": judge.repair_suggestions,
        }
        patch = self.qwen_client.plan_repair(payload)
        return GraphPatch.model_validate(self._normalize_patch(patch))
