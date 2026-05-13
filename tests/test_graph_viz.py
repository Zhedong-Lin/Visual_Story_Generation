from pathlib import Path

from anime_pipeline_graph.domain.models import GraphStep, SkillGraph, TaskSpec
from anime_pipeline_graph.planner.planning_models import CandidateGraph, CandidateGraphPlan, GraphScore
from anime_pipeline_graph.utils.graph_viz import write_run_graph_visualizations


def _mk_graph(graph_id: str, steps: list[str]) -> SkillGraph:
    graph_steps = [GraphStep(step_id=f"{s}_1", skill=s, inputs={}, params={}, outputs=[]) for s in steps]
    edges = []
    for i in range(len(graph_steps) - 1):
        edges.append((graph_steps[i].step_id, graph_steps[i + 1].step_id))
    return SkillGraph(graph_id=graph_id, steps=graph_steps, edges=edges, metadata={})


def test_write_run_graph_visualizations_outputs_files(tmp_path: Path):
    c1 = CandidateGraph(
        graph=_mk_graph("g1", ["character_bind", "prompt_pack_builder", "base_generation", "judge"]),
        source="llm_proposal",
        score=GraphScore(
            total_score=0.8,
            coverage_score=0.8,
            validity_score=0.8,
            prior_score=0.8,
            cost_score=0.8,
            diagnostics={},
        ),
    )
    c2 = CandidateGraph(
        graph=_mk_graph("g2", ["story_decompose", "prompt_pack_builder", "base_generation", "judge"]),
        source="motif_assembly",
        score=GraphScore(
            total_score=0.7,
            coverage_score=0.7,
            validity_score=0.7,
            prior_score=0.7,
            cost_score=0.7,
            diagnostics={},
        ),
    )
    c3 = CandidateGraph(
        graph=_mk_graph("g3", ["edit", "judge"]),
        source="fallback",
        score=GraphScore(
            total_score=0.6,
            coverage_score=0.6,
            validity_score=0.6,
            prior_score=0.6,
            cost_score=0.6,
            diagnostics={},
        ),
    )
    plan = CandidateGraphPlan(
        task_spec=TaskSpec(task_id="t1", task_type="storyboard", num_frames=3),
        candidates=[c1, c2, c3],
    )

    out = write_run_graph_visualizations(
        run_dir=tmp_path,
        candidate_plan=plan,
        selected_graph=c1.graph,
        validated_graph=c2.graph,
        validation_issues=["x"],
    )
    out_dir = Path(out["out_dir"])
    assert (out_dir / "candidate_1_llm_proposal.dot").exists()
    assert (out_dir / "candidate_2_motif_assembly.dot").exists()
    assert (out_dir / "candidate_3_fallback.dot").exists()
    assert (out_dir / "selected_before_validate.dot").exists()
    assert (out_dir / "validated_after.dot").exists()

