"""Lightweight graph visualization writer for run artifacts."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Tuple

from anime_pipeline_graph.domain.models import GraphEdge, SkillGraph
from anime_pipeline_graph.planner.planning_models import CandidateGraphPlan


def _normalized_edges(graph: SkillGraph) -> Iterable[Tuple[str, str, str]]:
    for edge in graph.edges:
        if isinstance(edge, GraphEdge):
            yield edge.source, edge.target, edge.edge_type or "data"
        elif isinstance(edge, (tuple, list)) and len(edge) >= 2:
            yield str(edge[0]), str(edge[1]), "data"


def _render_dot(graph: SkillGraph, title: str) -> str:
    lines = [
        "digraph G {",
        "  rankdir=LR;",
        '  graph [fontname="Helvetica", labelloc=t, labeljust=l];',
        f'  label="{title}";',
        '  node [shape=box, style="rounded,filled", fillcolor="#eef6ff", color="#4a6fa5", fontname="Helvetica"];',
        '  edge [color="#6c757d", fontname="Helvetica"];',
    ]
    for step in graph.steps:
        label = f"{step.step_id}\\n[{step.skill}]"
        lines.append(f'  "{step.step_id}" [label="{label}"];')
    for src, dst, edge_type in _normalized_edges(graph):
        lines.append(f'  "{src}" -> "{dst}" [label="{edge_type}"];')
    lines.append("}")
    return "\n".join(lines)


def _write_dot_and_png(dot_text: str, dot_path: Path) -> bool:
    dot_path.write_text(dot_text, encoding="utf-8")
    dot_bin = shutil.which("dot")
    if not dot_bin:
        return False
    png_path = dot_path.with_suffix(".png")
    try:
        subprocess.run([dot_bin, "-Tpng", str(dot_path), "-o", str(png_path)], check=True)
        return True
    except Exception:
        return False


def write_run_graph_visualizations(
    run_dir: Path,
    candidate_plan: CandidateGraphPlan,
    selected_graph: SkillGraph,
    validated_graph: SkillGraph | None,
    validation_issues: list[str],
) -> dict:
    """Write candidate/final graph DOT and PNG artifacts under run directory."""
    out_dir = run_dir / "graph_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine selected candidate by highest total score.
    selected_idx = -1
    best_score = float("-inf")
    for i, c in enumerate(candidate_plan.candidates):
        score = c.score.total_score if c.score else float("-inf")
        if score > best_score:
            best_score = score
            selected_idx = i

    wrote_png = False
    for i, cand in enumerate(candidate_plan.candidates, 1):
        score = cand.score.total_score if cand.score else None
        marker = " [SELECTED]" if (i - 1) == selected_idx else ""
        title = f"Candidate {i}: {cand.source} (score={score}){marker}"
        dot = _render_dot(cand.graph, title=title)
        dot_path = out_dir / f"candidate_{i}_{cand.source}.dot"
        wrote_png = _write_dot_and_png(dot, dot_path) or wrote_png

    selected_dot = _render_dot(selected_graph, "Selected Graph (before validate)")
    wrote_png = _write_dot_and_png(selected_dot, out_dir / "selected_before_validate.dot") or wrote_png

    wrote_validated = False
    if validation_issues and validated_graph is not None:
        validated_dot = _render_dot(validated_graph, "Validated Graph (after validate)")
        wrote_validated = _write_dot_and_png(validated_dot, out_dir / "validated_after.dot")

    return {
        "out_dir": str(out_dir),
        "selected_index": selected_idx,
        "selected_source": candidate_plan.candidates[selected_idx].source if selected_idx >= 0 else None,
        "selected_score": best_score if selected_idx >= 0 else None,
        "wrote_png": wrote_png,
        "wrote_validated_png": wrote_validated,
    }

