"""Candidate-based graph planner with multiple proposal sources."""

from __future__ import annotations

from typing import Any, Dict, List

from anime_pipeline_graph.domain.models import CapabilityPlan, GraphStep, SkillGraph, TaskSpec
from anime_pipeline_graph.planner.case_memory import CaseMemory
from anime_pipeline_graph.planner.planning_models import CandidateGraph, CandidateGraphPlan
from anime_pipeline_graph.planner.skill_library import SkillLibrary
from anime_pipeline_graph.planner.typed_graph import to_typed_graph


class CandidateGraphPlanner:
    """Generate multiple candidate graphs before selection."""

    def __init__(self, qwen_client: Any) -> None:
        self.qwen_client = qwen_client

    def _normalize(self, data: dict, spec: TaskSpec) -> SkillGraph:
        """Normalize nested graph payload shapes from LLM output."""
        source = data.get("skill_graph", data.get("graph", data))
        graph_id = source.get("graph_id", source.get("id", f"graph_{spec.task_id}"))
        steps = source.get("steps", [])
        edges = source.get("edges", [])

        normalized_edges = []
        for edge in edges:
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                normalized_edges.append((edge[0], edge[1]))
            elif isinstance(edge, dict):
                src = edge.get("from") or edge.get("src") or edge.get("source")
                dst = edge.get("to") or edge.get("dst") or edge.get("target")
                if src and dst:
                    normalized_edges.append((src, dst))

        graph = SkillGraph.model_validate(
            {
                "graph_id": graph_id,
                "steps": steps,
                "edges": normalized_edges,
                "metadata": source.get("metadata", {}),
            }
        )
        return self._hydrate_graph_metadata(to_typed_graph(graph), spec)

    def _hydrate_graph_metadata(self, graph: SkillGraph, spec: TaskSpec) -> SkillGraph:
        meta = dict(graph.metadata or {})
        meta.update(
            {
                "dynamic": True,
                "task_id": spec.task_id,
                "task_type": spec.task_type.value,
                "num_frames": spec.num_frames,
                "character_names": spec.character_names,
                "scene_names": spec.scene_names,
                "has_character_reference": spec.has_character_reference,
                "has_scene_reference": spec.has_scene_reference,
                "needs_identity_preservation": spec.needs_identity_preservation,
                "needs_local_editing": spec.needs_local_editing,
                "needs_pose_control": spec.needs_pose_control,
                "needs_story_continuity": spec.needs_story_continuity,
                "needs_multi_character_interaction": spec.needs_multi_character_interaction,
                "source_image": spec.source_image,
            }
        )
        return graph.model_copy(update={"metadata": meta})

    def _build_chain_graph(self, spec: TaskSpec, graph_id: str, step_skills: List[str], source: str) -> SkillGraph:
        steps: List[GraphStep] = []
        for idx, skill in enumerate(step_skills, start=1):
            outputs = []
            if skill == "story_decompose":
                outputs = ["frame_specs"]
            elif skill == "character_bind":
                outputs = ["char_anchor"]
            elif skill == "scene_condition":
                outputs = ["scene_pack"]
            elif skill == "pose_plan":
                outputs = ["pose_plan_json"]
            elif skill == "shot_plan":
                outputs = ["shot_plan_json"]
            elif skill == "transition_plan":
                outputs = ["transition_plan_json"]
            elif skill == "expression_plan":
                outputs = ["expression_plan_json"]
            elif skill == "continuity_state_tracker":
                outputs = ["continuity_state"]
            elif skill == "visual_style_plan":
                outputs = ["visual_style_plan"]
            elif skill == "pose_extract":
                outputs = ["pose_skeleton"]
            elif skill == "prompt_pack_builder":
                outputs = ["prompt_pack"]
            elif skill == "base_generation":
                outputs = ["generated_images"]
            elif skill == "edit":
                outputs = ["edited_images"]
            elif skill == "fill_edit":
                outputs = ["edited_images"]
            elif skill == "judge":
                outputs = ["judge_report"]

            steps.append(
                GraphStep(
                    step_id=f"{skill}_{idx}",
                    skill=skill,
                    skill_name=skill,
                    skill_type="runtime_skill",
                    frame_scope="all" if spec.num_frames > 1 else "single",
                    inputs_required=[],
                    outputs_produced=outputs,
                    optional=False,
                    inputs={},
                    params={},
                    outputs=outputs,
                )
            )

        edges = []
        order = [s.step_id for s in steps]
        for i in range(len(order) - 1):
            edges.append((order[i], order[i + 1]))

        graph = SkillGraph(graph_id=graph_id, steps=steps, edges=edges, metadata={"candidate_source": source})
        return self._hydrate_graph_metadata(to_typed_graph(graph), spec)

    def _build_llm_candidate(
        self,
        spec: TaskSpec,
        cap: CapabilityPlan,
        registry_payload: Dict[str, dict],
    ) -> CandidateGraph:
        payload = {
            "task_spec": spec.model_dump(mode="json"),
            "capability_plan": cap.model_dump(mode="json"),
            "skill_registry": registry_payload,
            "output_schema": {
                "graph": {
                    "graph_id": "string",
                    "steps": [
                        {
                            "step_id": "string",
                            "skill": "string",
                            "skill_type": "string",
                            "frame_scope": "string",
                            "inputs_required": ["string"],
                            "outputs_produced": ["string"],
                            "optional": False,
                            "inputs": {},
                            "params": {},
                            "outputs": ["string"],
                        }
                    ],
                    "edges": [{"source": "string", "target": "string", "edge_type": "data"}],
                    "metadata": {},
                }
            },
        }
        data = self.qwen_client.plan_graph(payload)
        graph = self._normalize(data, spec)
        return CandidateGraph(graph=graph, source="llm_proposal", metadata={"planner": "qwen_structured"})

    def _build_motif_candidate(self, spec: TaskSpec, motifs: List[Dict[str, Any]]) -> CandidateGraph:
        selected = motifs[0] if motifs else None
        if selected:
            step_skills = [str(x) for x in selected.get("steps", []) if str(x)]
        else:
            step_skills = []

        if not step_skills:
            if spec.needs_local_editing:
                step_skills = ["character_bind", "scene_condition", "prompt_pack_builder", "base_generation"]
            elif spec.num_frames > 1:
                step_skills = [
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
                ]
            else:
                step_skills = ["character_bind", "scene_condition", "prompt_pack_builder", "base_generation"]

        graph = self._build_chain_graph(spec, f"graph_motif_{spec.task_id}", step_skills, "motif_assembly")
        return CandidateGraph(
            graph=graph,
            source="motif_assembly",
            metadata={"motif": selected.get("name") if selected else "rule_fallback"},
        )

    def _build_fallback_candidate(self, spec: TaskSpec) -> CandidateGraph:
        if spec.needs_local_editing and spec.source_image:
            step_skills = ["character_bind", "scene_condition", "prompt_pack_builder", "base_generation"]
        elif spec.num_frames > 1:
            step_skills = [
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
            ]
        else:
            step_skills = ["character_bind", "prompt_pack_builder", "base_generation"]

        graph = self._build_chain_graph(spec, f"graph_fallback_{spec.task_id}", step_skills, "fallback")
        return CandidateGraph(graph=graph, source="fallback", metadata={"deterministic": True})

    def _build_memory_replay_candidate(self, spec: TaskSpec, case_memory: CaseMemory | None) -> CandidateGraph | None:
        if not case_memory:
            return None
        similar = case_memory.retrieve_similar_cases(spec, top_k=8)
        chosen = None
        for item in similar:
            if item.outcome.get("success") and item.graph_payload:
                chosen = item
                break
        if chosen is None:
            return None

        graph = self._normalize(chosen.graph_payload or {}, spec)
        graph = graph.model_copy(
            update={
                "graph_id": f"graph_memory_{spec.task_id}_{chosen.run_id}",
                "metadata": {
                    **(graph.metadata or {}),
                    "candidate_source": "memory_replay",
                    "memory_run_id": chosen.run_id,
                    "memory_outcome": chosen.outcome,
                },
            }
        )
        return CandidateGraph(
            graph=graph,
            source="memory_replay",
            metadata={
                "memory_run_id": chosen.run_id,
                "memory_task_type": chosen.task_spec_summary.get("task_type"),
                "memory_num_frames": chosen.task_spec_summary.get("num_frames"),
            },
        )

    def plan(
        self,
        spec: TaskSpec,
        cap: CapabilityPlan,
        registry_payload: Dict[str, dict],
        skill_library: SkillLibrary,
        case_memory: CaseMemory | None = None,
    ) -> CandidateGraphPlan:
        """Generate candidates from llm proposal + motif assembly + deterministic fallback."""
        motifs = skill_library.retrieve_motifs(spec, top_k=3)
        candidates: List[CandidateGraph] = []
        try:
            candidates.append(self._build_llm_candidate(spec, cap, registry_payload))
        except Exception as exc:
            # Keep pipeline robust when remote LLM planning times out.
            print(f"[GraphPlanner] LLM candidate failed, fallback to deterministic candidates: {exc}")

        candidates.append(self._build_motif_candidate(spec, motifs))
        candidates.append(self._build_fallback_candidate(spec))
        memory_candidate = self._build_memory_replay_candidate(spec, case_memory)
        if memory_candidate is not None:
            candidates.append(memory_candidate)
        return CandidateGraphPlan(task_spec=spec, candidates=candidates)
