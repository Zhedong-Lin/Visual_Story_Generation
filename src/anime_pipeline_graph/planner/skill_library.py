"""Structured planning-time skill library with lightweight retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from anime_pipeline_graph.domain.models import SkillSpec, TaskSpec


@dataclass
class SkillCard:
    """Planning-oriented skill specification."""

    name: str
    description: str
    skill_type: str
    applicability_conditions: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    preconditions: List[str]
    effects: List[str]
    allowed_predecessors: List[str]
    allowed_successors: List[str]
    failure_modes: List[str]
    failure_mode_repairs: Dict[str, List[Dict[str, Any]]]
    graph_motifs: List[str]

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "SkillCard":
        """Build a card from yaml/json payload."""
        return cls(
            name=str(payload.get("name", "")).strip(),
            description=str(payload.get("description", "")).strip(),
            skill_type=str(payload.get("skill_type", "generic")).strip(),
            applicability_conditions=list(payload.get("applicability_conditions", []) or []),
            input_schema=dict(payload.get("input_schema", {}) or {}),
            output_schema=dict(payload.get("output_schema", {}) or {}),
            preconditions=list(payload.get("preconditions", []) or []),
            effects=list(payload.get("effects", []) or []),
            allowed_predecessors=list(payload.get("allowed_predecessors", []) or []),
            allowed_successors=list(payload.get("allowed_successors", []) or []),
            failure_modes=list(payload.get("failure_modes", []) or []),
            failure_mode_repairs=dict(payload.get("failure_mode_repairs", {}) or {}),
            graph_motifs=list(payload.get("graph_motifs", []) or []),
        )


class SkillLibrary:
    """Skill cards and motif priors for retrieval-augmented planning."""

    def __init__(self, cards: Iterable[SkillCard], motifs: Dict[str, Dict[str, Any]] | None = None) -> None:
        self._cards: Dict[str, SkillCard] = {c.name: c for c in cards if c.name}
        self._motifs: Dict[str, Dict[str, Any]] = motifs or {}

    @classmethod
    def from_registry(
        cls,
        registry_skills: Dict[str, SkillSpec],
        library_dir: Path | None = None,
    ) -> "SkillLibrary":
        """Create library from existing SkillRegistry plus optional yaml/json overrides."""
        cards = [cls._card_from_registry_spec(spec) for spec in registry_skills.values()]
        motifs: Dict[str, Dict[str, Any]] = {}

        if library_dir and library_dir.exists():
            for path in sorted(library_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml", ".json"}:
                    continue
                payload = cls._read_structured(path)
                if not payload:
                    continue
                if isinstance(payload, dict) and "motifs" in payload:
                    motifs.update(dict(payload.get("motifs", {}) or {}))
                items = payload.get("skills", payload) if isinstance(payload, dict) else payload
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        card = SkillCard.from_payload(item)
                        if card.name:
                            cards = [c for c in cards if c.name != card.name]
                            cards.append(card)

        return cls(cards, motifs=motifs)

    @staticmethod
    def _read_structured(path: Path) -> Dict[str, Any] | List[Any] | None:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return json.loads(text)
        return yaml.safe_load(text)

    @staticmethod
    def _card_from_registry_spec(spec: SkillSpec) -> SkillCard:
        """Build a best-effort planning card from runtime SkillSpec."""
        skill_type = "generator"
        if spec.name in {"character_bind", "scene_condition", "prompt_pack_builder", "asset_normalizer"}:
            skill_type = "builder"
        elif spec.name in {"judge"}:
            skill_type = "judge"
        elif spec.name in {"edit", "fill_edit", "refine"}:
            skill_type = "editor"
        elif spec.name in {
            "story_decompose",
            "pose_plan",
            "shot_plan",
            "transition_plan",
            "expression_plan",
            "continuity_state_tracker",
            "visual_style_plan",
            "pose_extract",
        }:
            skill_type = "planner"

        outputs = {k: {"type": "artifact"} for k in spec.output_types}
        inputs = {k: {"type": "artifact"} for k in spec.input_types}

        return SkillCard(
            name=spec.name,
            description=spec.description,
            skill_type=skill_type,
            applicability_conditions=list(spec.best_for),
            input_schema=inputs,
            output_schema=outputs,
            preconditions=list(spec.preconditions),
            effects=list(spec.output_types),
            allowed_predecessors=[],
            allowed_successors=[],
            failure_modes=[],
            failure_mode_repairs={},
            graph_motifs=["default_generation"],
        )

    def list_skills(self) -> List[SkillCard]:
        """List all skill cards."""
        return list(self._cards.values())

    def get_skill(self, name: str) -> SkillCard | None:
        """Get one skill card by name."""
        return self._cards.get(name)

    def get_failure_mode_repairs(self, skill_name: str, failure_mode: str) -> List[Dict[str, Any]]:
        """Get suggested repair specs for one skill failure mode."""
        card = self.get_skill(skill_name)
        if not card:
            return []
        return list(card.failure_mode_repairs.get(failure_mode, []) or [])

    def get_repairs_for_skill(self, skill_name: str, failure_modes: List[str]) -> List[Dict[str, Any]]:
        """Collect deduplicated repair specs for multiple failure modes."""
        out: List[Dict[str, Any]] = []
        seen = set()
        for mode in failure_modes:
            for rep in self.get_failure_mode_repairs(skill_name, mode):
                key = json.dumps(rep, sort_keys=True, ensure_ascii=False)
                if key in seen:
                    continue
                seen.add(key)
                out.append(rep)
        return out

    def retrieve_candidates(self, task_spec: TaskSpec, top_k: int = 8) -> List[SkillCard]:
        """Retrieve candidate skills with metadata filter + keyword/rule scoring."""
        scored: List[tuple[float, SkillCard]] = []
        for card in self._cards.values():
            score = self._score_card(card, task_spec)
            if score <= 0:
                continue
            scored.append((score, card))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [card for _, card in scored[:top_k]]

    def retrieve_motifs(self, task_spec: TaskSpec, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve graph motifs by lightweight rule/keyword matching."""
        if not self._motifs:
            return []
        scored: List[tuple[float, Dict[str, Any]]] = []
        for name, motif in self._motifs.items():
            score = self._score_motif(name, motif, task_spec)
            if score <= 0:
                continue
            m = dict(motif)
            m.setdefault("name", name)
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def _score_card(self, card: SkillCard, task_spec: TaskSpec) -> float:
        score = 0.1
        text = " ".join(
            [
                card.description.lower(),
                " ".join(card.applicability_conditions).lower(),
                " ".join(card.graph_motifs).lower(),
            ]
        )

        if task_spec.needs_local_editing and card.skill_type == "editor":
            score += 3.0
        if task_spec.num_frames > 1 and any(k in text for k in ["story", "frame", "continuity", "expression", "pose"]):
            score += 2.0
        if task_spec.needs_pose_control and "pose" in text:
            score += 1.6
        if task_spec.needs_identity_preservation and any(k in text for k in ["identity", "character", "anchor"]):
            score += 1.4
        if task_spec.needs_scene_generation and "scene" in text:
            score += 1.1
        if card.name == "judge":
            score += 1.2
        if card.name == "prompt_pack_builder":
            score += 1.0
        if card.name == "base_generation" and not task_spec.needs_local_editing:
            score += 1.8
        if task_spec.needs_local_editing and card.name in {"edit", "fill_edit"}:
            score += 1.8
        return score

    def _score_motif(self, name: str, motif: Dict[str, Any], task_spec: TaskSpec) -> float:
        score = 0.2
        tags = [str(x).lower() for x in motif.get("tags", [])]
        lname = name.lower()
        steps = {str(x).lower() for x in motif.get("steps", [])}
        if task_spec.num_frames > 1 and ("storyboard" in tags or "story" in lname):
            score += 2.5
        if task_spec.needs_local_editing and ("edit" in tags or "edit" in lname):
            score += 2.5
        if not task_spec.needs_local_editing and "base_generation" in steps:
            score += 1.0
        if task_spec.needs_pose_control and ("pose_plan" in steps or "pose" in tags):
            score += 1.2
        if "judge" in steps:
            score += 0.8
        return score
