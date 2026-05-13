"""Minimal case memory over run records for graph prior scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from anime_pipeline_graph.domain.models import TaskSpec


@dataclass
class CaseMemoryItem:
    """One historical run case summary."""

    run_id: str
    task_spec_summary: Dict[str, Any]
    graph_summary: Dict[str, Any]
    outcome: Dict[str, Any]
    graph_payload: Dict[str, Any] | None = None


class CaseMemory:
    """Load and retrieve similar planning cases from existing run artifacts."""

    def __init__(self, items: Iterable[CaseMemoryItem]) -> None:
        self.items = list(items)

    @classmethod
    def from_run_dirs(cls, roots: List[Path], limit: int = 300) -> "CaseMemory":
        """Build memory from historical run folders."""
        items: List[CaseMemoryItem] = []
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            for run_dir in sorted([p for p in root.iterdir() if p.is_dir()], reverse=True):
                if len(items) >= limit:
                    break
                item = cls._load_case(run_dir)
                if item:
                    items.append(item)
        return cls(items)

    @staticmethod
    def _safe_read_json(path: Path) -> Dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @classmethod
    def _load_case(cls, run_dir: Path) -> CaseMemoryItem | None:
        run_record = cls._safe_read_json(run_dir / "run_record.json")
        task_spec = cls._safe_read_json(run_dir / "task_spec.json")
        graph = cls._safe_read_json(run_dir / "graph.json")
        if not task_spec and run_record:
            task_spec = run_record.get("task_spec")
        if not graph and run_record:
            graph = run_record.get("graph")
        if not task_spec or not graph:
            return None

        skills = []
        for s in graph.get("steps", []):
            if isinstance(s, dict) and s.get("skill"):
                skills.append(s["skill"])

        judge = (run_record or {}).get("judge") or {}
        final_score = judge.get("final_score")
        success = bool(final_score is not None and float(final_score) >= 0.85)

        return CaseMemoryItem(
            run_id=run_dir.name,
            task_spec_summary={
                "task_type": task_spec.get("task_type"),
                "num_frames": task_spec.get("num_frames", 1),
                "character_names": task_spec.get("character_names", []),
                "scene_names": task_spec.get("scene_names", []),
                "needs_local_editing": task_spec.get("needs_local_editing", False),
                "needs_pose_control": task_spec.get("needs_pose_control", False),
            },
            graph_summary={
                "num_steps": len(graph.get("steps", [])),
                "num_edges": len(graph.get("edges", [])),
                "skills": skills,
            },
            outcome={
                "success": success,
                "final_score": final_score,
                "repair_count": 1 if (run_dir / "repair_patch.json").exists() else 0,
                "failure_modes": judge.get("failure_tags", []),
            },
            graph_payload=graph,
        )

    def retrieve_similar_cases(self, task_spec: TaskSpec, top_k: int = 5) -> List[CaseMemoryItem]:
        """Retrieve top-k similar cases using lightweight structured similarity."""
        scored: List[tuple[float, CaseMemoryItem]] = []
        for item in self.items:
            score = self._similarity(task_spec, item)
            if score <= 0:
                continue
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]

    def _similarity(self, task_spec: TaskSpec, item: CaseMemoryItem) -> float:
        s = item.task_spec_summary
        score = 0.0
        if str(s.get("task_type")) == task_spec.task_type.value:
            score += 1.6
        score += max(0.0, 1.0 - abs(int(s.get("num_frames", 1)) - int(task_spec.num_frames)) * 0.3)

        chars = {str(x).lower() for x in s.get("character_names", [])}
        target_chars = {x.lower() for x in task_spec.character_names}
        if chars and target_chars:
            score += len(chars & target_chars) / max(len(target_chars), 1)

        scenes = {str(x).lower() for x in s.get("scene_names", [])}
        target_scenes = {x.lower() for x in task_spec.scene_names}
        if scenes and target_scenes:
            score += 0.8 * len(scenes & target_scenes) / max(len(target_scenes), 1)

        if bool(s.get("needs_local_editing", False)) == bool(task_spec.needs_local_editing):
            score += 0.6
        if bool(s.get("needs_pose_control", False)) == bool(task_spec.needs_pose_control):
            score += 0.4
        return score
