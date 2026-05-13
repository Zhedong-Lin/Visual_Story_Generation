"""Graph executor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

from anime_pipeline_graph.domain.models import GraphStep, InputBundle, SkillGraph, StepResult, TaskSpec
from anime_pipeline_graph.logging_utils import log_step
from anime_pipeline_graph.utils.graph_utils import topological_steps


@dataclass
class RuntimeContext:
    """Dependencies used by skills at runtime."""

    qwen_client: Any
    base_provider: Any
    kontext_provider: Any
    fill_provider: Any
    pose_extractor: Any
    lora_map: Dict[str, str]
    dry_run: bool = True


class GraphExecutor:
    """Execute dynamic skill graph and persist artifacts."""

    def __init__(self, state_store: Any, context: RuntimeContext) -> None:
        self.state_store = state_store
        self.ctx = context

    def _skill_impl(self, skill_name: str) -> Callable:
        """Load skill execute function by skill name."""
        mapping = {
            "asset_normalizer": "asset_normalizer",
            "character_bind": "character_bind_skill",
            "scene_condition": "scene_condition_skill",
            "pose_plan": "pose_plan_skill",
            "shot_plan": "shot_plan_skill",
            "transition_plan": "transition_plan_skill",
            "expression_plan": "expression_plan_skill",
            "continuity_state_tracker": "continuity_state_tracker_skill",
            "visual_style_plan": "visual_style_plan_skill",
            "pose_extract": "pose_extract_skill",
            "story_decompose": "story_decompose_skill",
            "prompt_pack_builder": "prompt_pack_builder",
            "base_generation": "base_generation_skill",
            "edit": "edit_skill",
            "fill_edit": "fill_edit_skill",
            "judge": "judge_skill",
            "refine": "refine_skill",
        }
        mod_name = mapping[skill_name]
        module = __import__(f"anime_pipeline_graph.skills.{mod_name}", fromlist=["execute"])
        return module.execute

    def execute(self, graph: SkillGraph, bundle: InputBundle, spec: TaskSpec) -> list[StepResult]:
        """Run graph in topological order."""
        step_map = {s.step_id: s for s in graph.steps}
        order = topological_steps(graph)
        self.state_store.set("input_bundle", bundle)
        self.state_store.set("task_spec", spec)
        self.state_store.save_named_json("graph", graph.model_dump(mode="json"))

        results: list[StepResult] = []
        for step_id in order:
            step: GraphStep = step_map[step_id]
            execute_func = self._skill_impl(step.skill)
            pre = {"step": step.model_dump(mode="json"), "state_keys": list(self.state_store.state.keys())}
            self.state_store.save_step_json(step.step_id, "pre", pre)
            log_step("Execute", f"{step.step_id} -> {step.skill}")
            started = datetime.now(timezone.utc)
            out = execute_func(step, self.state_store, self.ctx)
            ended = datetime.now(timezone.utc)
            result = StepResult(
                step_id=step.step_id,
                skill=step.skill,
                status="ok",
                started_at=started,
                ended_at=ended,
                outputs=out.get("outputs", {}),
                artifacts=[Path(p) for p in out.get("artifacts", [])],
                metadata=out.get("metadata", {}),
            )
            self.state_store.save_step_json(step.step_id, "post", result.model_dump(mode="json"))
            results.append(result)
        self.state_store.save_named_json("step_results", [r.model_dump(mode="json") for r in results])
        return results
