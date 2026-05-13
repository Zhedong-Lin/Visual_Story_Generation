"""Capability planner wrapper."""

from __future__ import annotations

from typing import Any

from anime_pipeline_graph.domain.models import CapabilityPlan, TaskSpec


class CapabilityPlanner:
    """Plan capability flags from TaskSpec."""

    def __init__(self, qwen_client: Any) -> None:
        self.qwen_client = qwen_client

    def _to_bool(self, value: Any, default: bool = False) -> bool:
        """Convert common LLM output patterns to bool."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y", "on", "enabled"}
        if isinstance(value, dict):
            for key in ("enabled", "value", "use", "required"):
                if key in value:
                    return self._to_bool(value[key], default)
        return default

    def _normalize(self, data: dict, spec: TaskSpec) -> dict:
        """Normalize nested capability payload into CapabilityPlan fields."""
        source = data.get("capability_plan", data)
        normalized = {
            "identity_preservation": self._to_bool(
                source.get("identity_preservation", source.get("identity", spec.needs_identity_preservation)),
                spec.needs_identity_preservation,
            ),
            "local_editing": self._to_bool(
                source.get("local_editing", source.get("editing", spec.needs_local_editing)),
                spec.needs_local_editing,
            ),
            "pose_control": self._to_bool(
                source.get("pose_control", source.get("pose", spec.needs_pose_control)),
                spec.needs_pose_control,
            ),
            "scene_reference_conditioning": self._to_bool(
                source.get("scene_reference_conditioning", source.get("scene_conditioning", True)),
                True,
            ),
            "story_continuity": self._to_bool(
                source.get("story_continuity", source.get("continuity", spec.needs_story_continuity)),
                spec.needs_story_continuity,
            ),
            "multi_character_interaction": self._to_bool(
                source.get(
                    "multi_character_interaction",
                    source.get("multi_character", spec.needs_multi_character_interaction),
                ),
                spec.needs_multi_character_interaction,
            ),
            "quality_refinement": self._to_bool(source.get("quality_refinement", source.get("refinement", True)), True),
        }
        return normalized

    def plan(self, spec: TaskSpec) -> CapabilityPlan:
        """Call Qwen and validate capability plan."""
        payload = {"task_spec": spec.model_dump(mode="json")}
        data = self.qwen_client.plan_capabilities(payload)
        return CapabilityPlan.model_validate(self._normalize(data, spec))
