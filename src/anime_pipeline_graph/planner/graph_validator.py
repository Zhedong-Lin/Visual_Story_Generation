"""Skill graph validator with explicit constraints and compatibility autofix."""

from __future__ import annotations

import os
from typing import Dict, List

import networkx as nx

from anime_pipeline_graph.domain.models import GraphStep, SkillGraph, TaskSpec
from anime_pipeline_graph.planner.constraints import ConstraintValidator, build_task_spec_from_graph_metadata
from anime_pipeline_graph.planner.graph_repair_search import BestFirstGraphRepair
from anime_pipeline_graph.planner.planning_models import Violation
from anime_pipeline_graph.planner.skill_library import SkillLibrary
from anime_pipeline_graph.planner.typed_graph import dedup_edges, normalized_edges, to_legacy_graph, to_typed_graph
from anime_pipeline_graph.skills.registry import SkillRegistry


class GraphValidator:
    """Validate graph structural constraints and perform compatibility autofix."""

    REQUIRED_FIELDS = {"step_id", "skill", "inputs", "params", "outputs"}
    KNOWN_SKILLS = {
        "asset_normalizer",
        "character_bind",
        "scene_condition",
        "pose_plan",
        "shot_plan",
        "transition_plan",
        "expression_plan",
        "continuity_state_tracker",
        "visual_style_plan",
        "pose_extract",
        "story_decompose",
        "prompt_pack_builder",
        "base_generation",
        "edit",
        "fill_edit",
        "judge",
        "refine",
    }

    def __init__(self, skill_library: SkillLibrary | None = None) -> None:
        self.skill_library = skill_library or SkillLibrary.from_registry(SkillRegistry().skills)
        self.constraint_validator = ConstraintValidator()
        self.repairer = BestFirstGraphRepair(self.constraint_validator)

    @staticmethod
    def _meta_bool(metadata: dict, key: str, default: bool = False) -> bool:
        value = metadata.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y", "on", "enabled"}
        return default

    @staticmethod
    def _flag_enabled(name: str, default: bool = False) -> bool:
        """Parse a boolean-like env flag."""
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}

    @classmethod
    def _disabled_skills(cls) -> set[str]:
        disabled: set[str] = set()
        if not cls._flag_enabled("ENABLE_EDIT", default=False):
            disabled.update({"edit", "fill_edit", "refine"})
        if not cls._flag_enabled("ENABLE_JUDGE", default=False):
            disabled.add("judge")
        return disabled

    def _prune_disabled_steps(self, graph: SkillGraph, issues: List[str]) -> None:
        disabled = self._disabled_skills()
        if not disabled:
            return
        removed = [s.step_id for s in graph.steps if s.skill in disabled]
        if not removed:
            return
        graph.steps = [s for s in graph.steps if s.skill not in disabled]
        kept_ids = {s.step_id for s in graph.steps}
        graph.edges = [(src, dst) for src, dst in normalized_edges(graph) if src in kept_ids and dst in kept_ids]
        issues.append(f"auto-fix: pruned disabled steps: {', '.join(removed)}")

    def _prune_steps_for_task(self, graph: SkillGraph, issues: List[str]) -> None:
        metadata = graph.metadata or {}
        task_type = str(metadata.get("task_type", "")).lower()
        num_frames = int(metadata.get("num_frames", 1) or 1)
        needs_pose_control = self._meta_bool(metadata, "needs_pose_control", num_frames > 1)
        needs_story_continuity = self._meta_bool(metadata, "needs_story_continuity", num_frames > 1)
        has_scene_reference = self._meta_bool(metadata, "has_scene_reference", False)
        needs_local_editing = self._meta_bool(metadata, "needs_local_editing", False)
        has_source_image = bool(metadata.get("source_image"))
        force_pose_in_edit = os.getenv("ENABLE_POSE_IN_EDIT", "").strip() == "1"

        remove_skills: set[str] = set()
        if task_type == "single_image" or num_frames <= 1:
            if not needs_pose_control and not (needs_local_editing and force_pose_in_edit):
                remove_skills.update({"pose_plan", "pose_extract"})
            remove_skills.add("expression_plan")
            remove_skills.add("shot_plan")
            remove_skills.add("transition_plan")
            remove_skills.add("continuity_state_tracker")
            if not needs_story_continuity:
                remove_skills.add("story_decompose")
            if not needs_story_continuity:
                remove_skills.add("visual_style_plan")
            if not has_scene_reference:
                remove_skills.add("scene_condition")
            if needs_local_editing and has_source_image:
                remove_skills.update({"base_generation", "prompt_pack_builder", "refine"})
        if not needs_local_editing:
            remove_skills.update({"edit", "fill_edit"})

        if not remove_skills:
            return

        kept_steps = [s for s in graph.steps if s.skill not in remove_skills]
        removed = [s.step_id for s in graph.steps if s.skill in remove_skills]
        if not removed:
            return

        graph.steps = kept_steps
        kept_ids = {s.step_id for s in graph.steps}
        graph.edges = [(src, dst) for src, dst in normalized_edges(graph) if src in kept_ids and dst in kept_ids]
        issues.append(f"auto-fix: pruned unnecessary steps for task shape: {', '.join(removed)}")

    def _ensure_edit_path_when_needed(self, graph: SkillGraph, issues: List[str]) -> None:
        if "edit" in self._disabled_skills():
            return
        metadata = graph.metadata or {}
        needs_local_editing = self._meta_bool(metadata, "needs_local_editing", False)
        source_image = metadata.get("source_image")
        if not needs_local_editing or not source_image:
            return
        edit_steps = [s for s in graph.steps if s.skill in {"edit", "fill_edit"}]
        if edit_steps:
            edit_id = edit_steps[0].step_id
        else:
            edit_id = "edit_autofix"
            graph.steps.append(
                GraphStep(
                    step_id=edit_id,
                    skill="edit",
                    inputs={},
                    params={"source_image": source_image},
                    outputs=["edited_images"],
                )
            )
            issues.append("auto-fix: inserted edit step for local-edit task with source image")

        # Ensure generation exists before edit (generate-then-edit policy).
        pp_steps = [s for s in graph.steps if s.skill == "prompt_pack_builder"]
        if pp_steps:
            pp_id = pp_steps[0].step_id
        else:
            pp_id = "prompt_pack_builder_autofix"
            graph.steps.append(GraphStep(step_id=pp_id, skill="prompt_pack_builder", inputs={}, params={}, outputs=["prompt_pack"]))
            for s in graph.steps:
                if s.step_id == pp_id:
                    continue
                if s.skill in {"character_bind", "scene_condition", "transition_plan", "shot_plan", "pose_plan", "expression_plan", "story_decompose", "continuity_state_tracker", "visual_style_plan"}:
                    graph.edges.append((s.step_id, pp_id))
            issues.append("auto-fix: inserted prompt_pack_builder for generate-then-edit path")

        base_steps = [s for s in graph.steps if s.skill == "base_generation"]
        if base_steps:
            base_ids = [s.step_id for s in base_steps]
        else:
            base_id = "base_generation_autofix"
            graph.steps.append(GraphStep(step_id=base_id, skill="base_generation", inputs={}, params={}, outputs=["generated_images"]))
            graph.edges.append((pp_id, base_id))
            base_ids = [base_id]
            issues.append("auto-fix: inserted base_generation before edit for local-edit task")

        for bid in base_ids:
            graph.edges.append((bid, edit_id))

        judge_steps = [s for s in graph.steps if s.skill == "judge"]
        for js in judge_steps:
            graph.edges.append((edit_id, js.step_id))

    def _ensure_base_generation_inputs(self, graph: SkillGraph, issues: List[str]) -> None:
        base_steps = [s for s in graph.steps if s.skill == "base_generation"]
        if not base_steps:
            return
        has_prompt_pack_builder = any(s.skill == "prompt_pack_builder" for s in graph.steps)
        if has_prompt_pack_builder:
            return

        pp_id = "prompt_pack_builder_autofix"
        graph.steps.append(GraphStep(step_id=pp_id, skill="prompt_pack_builder", inputs={}, params={}, outputs=["prompt_pack"]))
        for s in graph.steps:
            if s.step_id == pp_id:
                continue
            if s.skill in {"character_bind", "scene_condition", "pose_plan", "expression_plan", "story_decompose"}:
                graph.edges.append((s.step_id, pp_id))
        for base in base_steps:
            graph.edges.append((pp_id, base.step_id))
        issues.append("auto-fix: inserted prompt_pack_builder for base_generation prerequisites")

    def _ensure_storyboard_generation_path(self, graph: SkillGraph, issues: List[str]) -> None:
        metadata = graph.metadata or {}
        num_frames = int(metadata.get("num_frames", 1) or 1)
        task_type = str(metadata.get("task_type", "")).lower()
        if num_frames <= 1 and task_type != "storyboard":
            return
        if any(s.skill == "base_generation" for s in graph.steps):
            return

        pp_steps = [s for s in graph.steps if s.skill == "prompt_pack_builder"]
        if pp_steps:
            pp_id = pp_steps[0].step_id
        else:
            pp_id = "prompt_pack_builder_autofix"
            graph.steps.append(GraphStep(step_id=pp_id, skill="prompt_pack_builder", inputs={}, params={}, outputs=["prompt_pack"]))
            for s in graph.steps:
                if s.step_id == pp_id:
                    continue
                if s.skill in {"character_bind", "scene_condition", "pose_plan", "expression_plan", "story_decompose"}:
                    graph.edges.append((s.step_id, pp_id))
            issues.append("auto-fix: inserted prompt_pack_builder for storyboard generation path")

        base_id = "base_generation_autofix"
        graph.steps.append(GraphStep(step_id=base_id, skill="base_generation", inputs={}, params={}, outputs=["generated_images"]))
        graph.edges.append((pp_id, base_id))

        edit_steps = [s for s in graph.steps if s.skill in {"edit", "fill_edit"}]
        if edit_steps:
            for s in edit_steps:
                graph.edges.append((base_id, s.step_id))
        else:
            judge_steps = [s for s in graph.steps if s.skill == "judge"]
            for js in judge_steps:
                graph.edges.append((base_id, js.step_id))
        issues.append("auto-fix: inserted base_generation path for storyboard task")

    def _ensure_expression_plan_for_storyboard(self, graph: SkillGraph, issues: List[str]) -> None:
        metadata = graph.metadata or {}
        num_frames = int(metadata.get("num_frames", 1) or 1)
        task_type = str(metadata.get("task_type", "")).lower()
        if num_frames <= 1 and task_type != "storyboard":
            return
        if any(s.skill == "expression_plan" for s in graph.steps):
            return
        if not any(s.skill in {"prompt_pack_builder", "base_generation"} for s in graph.steps):
            return

        expr_id = "expression_plan_autofix"
        graph.steps.append(
            GraphStep(step_id=expr_id, skill="expression_plan", inputs={}, params={}, outputs=["expression_plan_json"])
        )
        step_by_skill = {s.skill: s.step_id for s in graph.steps}
        if "story_decompose" in step_by_skill:
            graph.edges.append((step_by_skill["story_decompose"], expr_id))
        if "prompt_pack_builder" in step_by_skill:
            graph.edges.append((expr_id, step_by_skill["prompt_pack_builder"]))
        issues.append("auto-fix: inserted expression_plan for storyboard-style task")

    def _ensure_story_decompose_as_entry(self, graph: SkillGraph, issues: List[str]) -> None:
        """Ensure storyboard flow starts from story_decompose and fans out to planning nodes."""
        metadata = graph.metadata or {}
        num_frames = int(metadata.get("num_frames", 1) or 1)
        task_type = str(metadata.get("task_type", "")).lower()
        if num_frames <= 1 and task_type != "storyboard":
            return

        story_steps = [s for s in graph.steps if s.skill == "story_decompose"]
        if story_steps:
            story_id = story_steps[0].step_id
        else:
            story_id = "story_decompose_autofix"
            graph.steps.append(
                GraphStep(step_id=story_id, skill="story_decompose", inputs={}, params={}, outputs=["frame_specs"])
            )
            issues.append("auto-fix: inserted story_decompose for storyboard/multi-frame task")

        existing = set(normalized_edges(graph))
        targets = [
            s.step_id
            for s in graph.steps
            if s.skill
            in {
                "character_bind",
                "scene_condition",
                "transition_plan",
                "shot_plan",
                "pose_plan",
                "expression_plan",
                "continuity_state_tracker",
                "visual_style_plan",
                "prompt_pack_builder",
            }
        ]
        added = 0
        for target in targets:
            if target == story_id:
                continue
            edge = (story_id, target)
            if edge not in existing:
                graph.edges.append(edge)
                existing.add(edge)
                added += 1
        if added:
            issues.append(f"auto-fix: connected story_decompose upstream to {added} planning nodes")

    def _ensure_prompt_pack_dependencies(self, graph: SkillGraph, issues: List[str]) -> None:
        """Ensure planning outputs are upstream of prompt_pack_builder in non-strict mode too."""
        pp_steps = [s for s in graph.steps if s.skill == "prompt_pack_builder"]
        if not pp_steps:
            return
        pp_id = pp_steps[0].step_id
        existing = set(normalized_edges(graph))
        planner_skills = {
            "story_decompose",
            "character_bind",
            "scene_condition",
            "transition_plan",
            "shot_plan",
            "pose_plan",
            "expression_plan",
            "continuity_state_tracker",
            "visual_style_plan",
        }
        added = 0
        for s in graph.steps:
            if s.step_id == pp_id:
                continue
            if s.skill not in planner_skills:
                continue
            edge = (s.step_id, pp_id)
            if edge in existing:
                continue
            graph.edges.append(edge)
            existing.add(edge)
            added += 1
        if added:
            issues.append(f"auto-fix: connected {added} planning nodes into prompt_pack_builder")

    def _enforce_strict_mainchain(self, graph: SkillGraph, issues: List[str]) -> None:
        """Force a deterministic main-chain DAG to keep execution order predictable."""
        metadata = graph.metadata or {}
        task_type = str(metadata.get("task_type", "")).lower()
        num_frames = int(metadata.get("num_frames", 1) or 1)
        storyboard_like = task_type == "storyboard" or num_frames > 1

        # Keep chain minimal for single-image tasks; richer for storyboard-like tasks.
        ordered_skills = (
            [
                "story_decompose",
                "character_bind",
                "scene_condition",
                "transition_plan",
                "shot_plan",
                "pose_plan",
                "expression_plan",
                "continuity_state_tracker",
                "visual_style_plan",
                "prompt_pack_builder",
                "base_generation",
                "edit",
                "fill_edit",
                "refine",
                "judge",
            ]
            if storyboard_like
            else [
                "character_bind",
                "scene_condition",
                "prompt_pack_builder",
                "base_generation",
                "edit",
                "fill_edit",
                "refine",
                "judge",
            ]
        )

        ordered_skills = [skill for skill in ordered_skills if skill not in self._disabled_skills()]

        # Pick first step id per skill to avoid duplicated autofix branches.
        first_by_skill: Dict[str, str] = {}
        for step in graph.steps:
            if step.skill in ordered_skills and step.skill not in first_by_skill:
                first_by_skill[step.skill] = step.step_id

        chain_ids: List[str] = []
        for skill in ordered_skills:
            sid = first_by_skill.get(skill)
            if not sid:
                continue
            # edit/fill_edit are alternatives; keep whichever appears first in order.
            if skill == "fill_edit" and first_by_skill.get("edit"):
                continue
            chain_ids.append(sid)

        # Make judge the final sink in chain when present.
        if "judge" in first_by_skill:
            jid = first_by_skill["judge"]
            chain_ids = [x for x in chain_ids if x != jid] + [jid]

        if len(chain_ids) < 2:
            return

        keep_ids = set(chain_ids)
        pruned_steps = [s.step_id for s in graph.steps if s.step_id not in keep_ids]
        if pruned_steps:
            graph.steps = [s for s in graph.steps if s.step_id in keep_ids]
        # Reorder step list to match strict chain order for easier inspection/debugging.
        step_by_id = {s.step_id: s for s in graph.steps}
        graph.steps = [step_by_id[sid] for sid in chain_ids if sid in step_by_id]

        # Strict chain edges only.
        strict_edges = [(chain_ids[i], chain_ids[i + 1]) for i in range(len(chain_ids) - 1)]
        graph.edges = dedup_edges(strict_edges)

        # Ensure last node is judge when available.
        if "judge" in first_by_skill and chain_ids[-1] != first_by_skill["judge"]:
            graph.edges.append((chain_ids[-1], first_by_skill["judge"]))
            graph.edges = dedup_edges(graph.edges)

        msg = "auto-fix: strict mainchain DAG enforced"
        if pruned_steps:
            msg += f"; pruned side branches: {', '.join(pruned_steps)}"
        issues.append(msg)

    def _normalize_edge_node(
        self,
        node: str,
        step_ids: set[str],
        skill_to_ids: Dict[str, List[str]],
        graph: SkillGraph,
        issues: List[str],
    ) -> str:
        if node in step_ids:
            return node
        ids = skill_to_ids.get(node, [])
        if len(ids) == 1:
            mapped = ids[0]
            issues.append(f"auto-fix: edge node '{node}' mapped to step_id '{mapped}'")
            return mapped
        if node in self.KNOWN_SKILLS:
            new_step = GraphStep(step_id=node, skill=node, inputs={}, params={}, outputs=[])
            graph.steps.append(new_step)
            step_ids.add(node)
            skill_to_ids.setdefault(node, []).append(node)
            issues.append(f"auto-fix: inserted missing step '{node}' from edge reference")
            return node
        return node

    def validate_constraints(
        self,
        graph: SkillGraph,
        task_spec: TaskSpec,
        skill_library: SkillLibrary | None = None,
    ) -> List[Violation]:
        """New explicit constraint API: validate(graph, task_spec, skill_library)->violations."""
        lib = skill_library or self.skill_library
        return self.constraint_validator.validate(graph, task_spec, lib)

    def validate(self, graph: SkillGraph, auto_fix: bool = True) -> tuple[SkillGraph, List[str]]:
        """Legacy API used by current tests/pipeline. Returns graph + issue strings."""
        issues: List[str] = []
        step_map: Dict[str, str] = {}

        graph = to_typed_graph(graph)
        if auto_fix:
            self._prune_disabled_steps(graph, issues)

        for step in graph.steps:
            step_map[step.step_id] = step.skill
            missing = [f for f in self.REQUIRED_FIELDS if not hasattr(step, f)]
            if missing:
                issues.append(f"step {step.step_id} missing fields: {missing}")
            if step.skill == "story_decompose" and graph.metadata.get("num_frames", 1) <= 1:
                issues.append("story_decompose only valid for multi-frame")
            if step.skill in {"edit", "fill_edit"} and "image" not in step.inputs and "image" not in step.params:
                issues.append(f"{step.skill} needs image input")

        if "judge" not in step_map.values() and self._flag_enabled("ENABLE_JUDGE", default=False):
            if auto_fix:
                graph.steps.append(GraphStep(step_id="judge_autofix", skill="judge", inputs={}, params={}, outputs=["judge_report"]))
                if graph.steps:
                    prev = graph.steps[-2].step_id if len(graph.steps) > 1 else graph.steps[-1].step_id
                    graph.edges.append((prev, "judge_autofix"))
                issues.append("auto-fix: judge step appended")
            else:
                issues.append("judge step missing")

        has_character_signal = bool(graph.metadata.get("character_names")) or bool(graph.metadata.get("has_character_reference", False))
        has_character_bind = any(step.skill == "character_bind" for step in graph.steps)
        if has_character_signal and not has_character_bind and auto_fix:
            bind_step = GraphStep(step_id="character_bind_autofix", skill="character_bind", inputs={}, params={}, outputs=["char_anchor"])
            graph.steps.insert(0, bind_step)
            if graph.steps and len(graph.steps) > 1:
                first_original = graph.steps[1].step_id
                graph.edges.append(("character_bind_autofix", first_original))
            issues.append("auto-fix: character_bind inserted due to character signals")

        if auto_fix:
            self._prune_steps_for_task(graph, issues)
            self._ensure_storyboard_generation_path(graph, issues)
            self._ensure_base_generation_inputs(graph, issues)
            self._ensure_edit_path_when_needed(graph, issues)
            self._ensure_expression_plan_for_storyboard(graph, issues)
            self._ensure_story_decompose_as_entry(graph, issues)
            self._ensure_prompt_pack_dependencies(graph, issues)

            step_ids = {s.step_id for s in graph.steps}
            skill_to_ids: Dict[str, List[str]] = {}
            for s in graph.steps:
                skill_to_ids.setdefault(s.skill, []).append(s.step_id)
            fixed_edges = []
            for src, dst in normalized_edges(graph):
                src_fixed = self._normalize_edge_node(src, step_ids, skill_to_ids, graph, issues)
                dst_fixed = self._normalize_edge_node(dst, step_ids, skill_to_ids, graph, issues)
                fixed_edges.append((src_fixed, dst_fixed))
            graph.edges = dedup_edges(fixed_edges)

            # Run constraint repair search as a final structural repair pass.
            task_spec = build_task_spec_from_graph_metadata(graph)
            violations = self.validate_constraints(graph, task_spec)
            if violations:
                repaired, remaining, stats = self.repairer.repair(graph, task_spec, self.skill_library, violations)
                graph = repaired
                self._prune_disabled_steps(graph, issues)
                if len(remaining) < len(violations):
                    issues.append(
                        f"auto-fix: repair-search reduced violations {len(violations)}->{len(remaining)}"
                    )
                else:
                    issues.append("auto-fix: repair-search attempted with limited improvement")
                issues.extend([f"constraint:{v.code}:{v.message}" for v in remaining])
                graph.metadata["repair_search"] = stats

            if self._flag_enabled("STRICT_MAINCHAIN_DAG", default=False):
                self._enforce_strict_mainchain(graph, issues)

        g = nx.DiGraph()
        for step in graph.steps:
            g.add_node(step.step_id)
        for src, dst in normalized_edges(graph):
            g.add_edge(src, dst)
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("graph is not a DAG")

        return to_legacy_graph(graph), issues
