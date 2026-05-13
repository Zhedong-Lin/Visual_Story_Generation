"""Pydantic models for planner and executor."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from anime_pipeline_graph.domain.enums import BackendType, CostLevel, LatencyLevel, TaskType


class InputBundle(BaseModel):
    """Normalized system input."""

    user_text: str
    character_references: Dict[str, Path] = Field(default_factory=dict)
    scene_references: Dict[str, Path] = Field(default_factory=dict)
    setting_docs: List[str] = Field(default_factory=list)
    history_frames: List[Path] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    action_reference: Optional[Path] = None


class TaskSpec(BaseModel):
    """Structured task spec output by parser."""

    task_id: str
    task_type: TaskType
    num_frames: int = 1
    num_characters: int = 0
    character_names: List[str] = Field(default_factory=list)
    scene_names: List[str] = Field(default_factory=list)
    source_image: Optional[str] = None
    has_character_reference: bool = False
    has_scene_reference: bool = False
    has_setting_doc: bool = False
    needs_identity_preservation: bool = False
    needs_local_editing: bool = False
    needs_pose_control: bool = False
    needs_scene_generation: bool = True
    needs_layout_control: bool = False
    needs_story_continuity: bool = False
    needs_multi_character_interaction: bool = False
    action_intensity: str = "low"
    edit_scope: str = "none"
    scene_strength: str = "medium"
    priority: str = "quality"
    character_constraints: List[str] = Field(default_factory=list)
    scene_constraints: List[str] = Field(default_factory=list)
    story_constraints: List[str] = Field(default_factory=list)
    frame_descriptions: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)


class CapabilityPlan(BaseModel):
    """Capability requirements chosen by planner."""

    identity_preservation: bool
    local_editing: bool
    pose_control: bool
    scene_reference_conditioning: bool
    story_continuity: bool
    multi_character_interaction: bool
    quality_refinement: bool


class SkillSpec(BaseModel):
    """Registry entry for one skill."""

    name: str
    description: str
    backend: BackendType
    model: str
    input_types: List[str]
    output_types: List[str]
    best_for: List[str]
    preconditions: List[str]
    supports_multi_frame: bool
    supports_multi_character: bool
    cost_level: CostLevel
    latency_level: LatencyLevel


class GraphStep(BaseModel):
    """One executable graph step."""

    step_id: str
    skill: str
    # Rich typed-node fields (backward-compatible with existing executor).
    skill_name: Optional[str] = None
    skill_type: str = "generic"
    frame_scope: str = "global"
    inputs_required: List[str] = Field(default_factory=list)
    outputs_produced: List[str] = Field(default_factory=list)
    optional: bool = False
    inputs: Dict[str, Any] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    outputs: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sync_typed_fields(self) -> "GraphStep":
        """Keep typed fields and legacy fields consistent."""
        if not self.skill_name:
            self.skill_name = self.skill
        if not self.outputs_produced and self.outputs:
            self.outputs_produced = list(self.outputs)
        return self


class GraphEdge(BaseModel):
    """Typed edge representation for planning-time constraints/search."""

    source: str
    target: str
    edge_type: str = "data"


class SkillGraph(BaseModel):
    """Dynamic execution graph."""

    graph_id: str
    steps: List[GraphStep]
    edges: List[tuple[str, str] | GraphEdge]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    """Executor step result."""

    step_id: str
    skill: str
    status: str
    started_at: datetime
    ended_at: datetime
    outputs: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[Path] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JudgeSubScores(BaseModel):
    """Judge detailed scores."""

    instruction_match: float
    identity_preservation: float
    costume_accuracy: float
    pose_accuracy: float
    scene_match: float
    story_consistency: float


class JudgeReport(BaseModel):
    """Judge report."""

    final_score: float
    subscores: JudgeSubScores
    failure_tags: List[str] = Field(default_factory=list)
    repair_suggestions: List[str] = Field(default_factory=list)


class GraphPatch(BaseModel):
    """Repair patch to graph."""

    reason: str
    actions: List[Dict[str, Any]] = Field(default_factory=list)


class AssetResolutionReport(BaseModel):
    """Asset resolve report."""

    found_characters: Dict[str, Path] = Field(default_factory=dict)
    missing_characters: List[str] = Field(default_factory=list)
    found_scenes: Dict[str, Path] = Field(default_factory=dict)
    missing_scenes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    """High-level run output."""

    run_id: str
    run_dir: Path
    task_spec: TaskSpec
    capability_plan: CapabilityPlan
    graph: SkillGraph
    judge: Optional[JudgeReport] = None
