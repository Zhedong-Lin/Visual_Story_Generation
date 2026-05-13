"""Typer CLI entrypoint."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import typer

from anime_pipeline_graph.config import AppConfig
from anime_pipeline_graph.domain.models import InputBundle, RunRecord
from anime_pipeline_graph.executor.graph_executor import GraphExecutor, RuntimeContext
from anime_pipeline_graph.executor.state_store import StateStore
from anime_pipeline_graph.logging_utils import log_kv, log_step
from anime_pipeline_graph.parser.asset_resolver import AssetResolver
from anime_pipeline_graph.parser.dialogue_preprocessor import preprocess_dialogue_prompt
from anime_pipeline_graph.parser.task_parser import TaskParser
from anime_pipeline_graph.paths import EXAMPLES_DIR, PROMPTS_DIR
from anime_pipeline_graph.planner.capability_planner import CapabilityPlanner
from anime_pipeline_graph.planner.case_memory import CaseMemory
from anime_pipeline_graph.planner.graph_planner import GraphPlanner
from anime_pipeline_graph.planner.skill_library import SkillLibrary
from anime_pipeline_graph.planner.graph_validator import GraphValidator
from anime_pipeline_graph.planner.repair_planner import RepairPlanner
from anime_pipeline_graph.providers.flux_base_provider import FluxBaseProvider, MockFluxBaseProvider
from anime_pipeline_graph.providers.flux_fill_provider import FluxFillProvider, MockFluxFillProvider
from anime_pipeline_graph.providers.flux_kontext_provider import FluxKontextProvider, MockFluxKontextProvider
from anime_pipeline_graph.providers.mock_qwen_client import MockQwenClient
from anime_pipeline_graph.providers.pose_reference_extractor import MockPoseReferenceExtractor, PoseReferenceExtractor
from anime_pipeline_graph.providers.seedream4_api_provider import SeedDream4ApiProvider
from anime_pipeline_graph.skills.registry import SkillRegistry
from anime_pipeline_graph.utils.graph_viz import write_run_graph_visualizations
from anime_pipeline_graph.utils.io import ensure_dir

app = typer.Typer(help="Dynamic anime skill graph planner")


def _list_asset_stems(folder: Path) -> list[str]:
    """List lowercase file stems under a directory."""
    if not folder.exists():
        return []
    return sorted({p.stem.lower() for p in folder.iterdir() if p.is_file()})


def make_bundle(user_text: str, constraints: Optional[Dict[str, Any]] = None) -> InputBundle:
    """Create standard InputBundle."""
    return InputBundle(user_text=user_text, constraints=constraints or {})


def _load_lora_map(project_root: Path) -> Dict[str, str]:
    """Load optional lora map yaml/json when exists."""
    yaml_path = project_root / "examples" / "lora_map.yaml"
    json_path = project_root / "examples" / "lora_map.json"
    if yaml_path.exists():
        import yaml

        return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    return {}


def _load_character_profiles(project_root: Path) -> Dict[str, Dict[str, str]]:
    """Load optional character appearance/outfit profiles."""
    yaml_path = project_root / "examples" / "character_profiles.yaml"
    json_path = project_root / "examples" / "character_profiles.json"
    if yaml_path.exists():
        import yaml

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    return {}


def _load_edit_source_map(project_root: Path) -> Dict[str, str]:
    """Load optional edit source images keyed by lowercase stem."""
    folder = project_root / "examples" / "assets" / "edit_source"
    if not folder.exists():
        return {}
    valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    mapping: Dict[str, str] = {}
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in valid_ext:
            continue
        mapping[p.stem.lower()] = str(p)
    return mapping


def run_pipeline(prompt: str, dry_run: bool, project_root: Path, run_id: Optional[str] = None) -> RunRecord:
    """Main orchestrator used by scripts and CLI."""
    config = AppConfig()
    backend = str(config.image_backend).strip().lower()
    use_seedream_api = backend in {"seedream4_api", "seedream_api", "ark_seedream4", "ark", "doubao_api"}
    run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
    run_dir = ensure_dir(project_root / config.default_output_dir / run_id)
    log_step("Init", f"run_id={run_id} dry_run={dry_run}")

    if dry_run:
        qwen_client = MockQwenClient()
    else:
        from anime_pipeline_graph.providers.qwen_api_client import QwenApiClient

        qwen_client = QwenApiClient(config, PROMPTS_DIR)

    dialogue_preprocess = preprocess_dialogue_prompt(
        prompt,
        qwen_client,
        enabled=os.getenv("PREPROCESS_DIALOGUE", "1").lower() not in {"0", "false", "no", "off"},
    )
    pipeline_prompt = str(dialogue_preprocess["pipeline_prompt"])
    if dialogue_preprocess.get("preprocessed"):
        log_step("Preprocess Dialogue", "Rewrote dialogue-heavy input into visual-only frame text.")

    bundle = make_bundle(pipeline_prompt)
    bundle.constraints["original_user_text"] = prompt
    bundle.constraints["dialogue_preprocess"] = dialogue_preprocess
    char_profiles = _load_character_profiles(project_root)
    if char_profiles:
        bundle.constraints["character_profiles"] = char_profiles
    edit_source_map = _load_edit_source_map(project_root)
    if edit_source_map:
        bundle.constraints["edit_source_map"] = edit_source_map

    log_step("Parse Task", "Calling parser (Qwen or mock)...")
    characters_dir = project_root / "examples" / "assets" / "characters"
    scenes_dir = project_root / "examples" / "assets" / "scenes"
    parser = TaskParser(
        qwen_client,
        known_character_names=_list_asset_stems(characters_dir),
        known_scene_names=_list_asset_stems(scenes_dir),
    )
    task_spec = parser.parse(bundle)

    log_step("Resolve Assets", "Matching character/scene names to local refs...")
    resolver = AssetResolver(
        characters_dir=characters_dir,
        scenes_dir=scenes_dir,
    )
    bundle, resolve_report = resolver.resolve(bundle, task_spec)

    # refresh parser-derived bool fields after resolving assets
    task_spec = task_spec.model_copy(
        update={
            "has_character_reference": len(bundle.character_references) > 0,
            "has_scene_reference": len(bundle.scene_references) > 0,
            "has_setting_doc": len(bundle.setting_docs) > 0,
        }
    )

    log_step("Capability Plan", "Calling capability planner...")
    capability = CapabilityPlanner(qwen_client).plan(task_spec)
    registry = SkillRegistry()
    skill_library = SkillLibrary.from_registry(
        registry.skills,
        library_dir=project_root / "src" / "anime_pipeline_graph" / "skills" / "library",
    )
    retrieved_skills = [c.name for c in skill_library.retrieve_candidates(task_spec, top_k=10)]
    retrieved_motifs = skill_library.retrieve_motifs(task_spec, top_k=5)
    case_memory = CaseMemory.from_run_dirs(
        [project_root / config.default_output_dir, project_root / "runs_baseline"],
        limit=250,
    )

    planner = GraphPlanner(qwen_client, skill_library=skill_library, case_memory=case_memory)
    candidate_plan = planner.plan_candidates(task_spec, capability, registry.as_dict())
    log_step("Graph Plan", "Generating candidate graphs + selecting best...")
    graph = planner.select_best(candidate_plan, task_spec)
    selected_graph_before_validate = graph.model_copy(deep=True)
    graph.metadata["num_frames"] = task_spec.num_frames
    graph.metadata["task_type"] = task_spec.task_type.value
    graph.metadata["character_names"] = task_spec.character_names
    graph.metadata["has_character_reference"] = task_spec.has_character_reference
    graph.metadata["has_scene_reference"] = task_spec.has_scene_reference
    graph.metadata["needs_pose_control"] = task_spec.needs_pose_control
    graph.metadata["needs_story_continuity"] = task_spec.needs_story_continuity
    graph.metadata["needs_multi_character_interaction"] = task_spec.needs_multi_character_interaction
    graph.metadata["needs_local_editing"] = task_spec.needs_local_editing
    graph.metadata["source_image"] = task_spec.source_image
    log_step("Graph Validate", "Validating graph...")
    graph, issues = GraphValidator().validate(graph, auto_fix=True)

    if dry_run:
        base_provider = MockFluxBaseProvider()
        kontext_provider = MockFluxKontextProvider()
        fill_provider = MockFluxFillProvider()
        pose_extractor = MockPoseReferenceExtractor()
    else:
        if use_seedream_api:
            # API mode currently supports text2img; disable Kontext path by default.
            os.environ["FORCE_NO_KONTEXT"] = "1"
            os.environ["FORCE_BASE_ONLY"] = "1"
            seedream_provider = SeedDream4ApiProvider(
                model_name=config.ark_seedream_model,
                api_key=config.ark_api_key,
                base_url=config.ark_base_url,
                endpoint=config.ark_images_endpoint,
                timeout_seconds=config.ark_timeout_seconds,
                response_format=config.ark_response_format,
                default_size=config.ark_default_size,
                min_pixels=config.ark_min_pixels,
            )
            # Reuse the same API provider for base/edit/fill skill interfaces.
            base_provider = seedream_provider
            kontext_provider = seedream_provider
            fill_provider = seedream_provider
            pose_extractor = PoseReferenceExtractor()
        else:
            base_provider = FluxBaseProvider(config.flux_base_model, config.hf_token)
            kontext_provider = FluxKontextProvider(config.flux_kontext_model, config.hf_token)
            fill_provider = FluxFillProvider(config.flux_fill_model, config.hf_token)
            pose_extractor = PoseReferenceExtractor()

    lora_map = _load_lora_map(project_root)
    context = RuntimeContext(
        qwen_client=qwen_client,
        base_provider=base_provider,
        kontext_provider=kontext_provider,
        fill_provider=fill_provider,
        pose_extractor=pose_extractor,
        lora_map=lora_map,
        dry_run=dry_run,
    )

    store = StateStore(run_dir)
    store.save_named_json("dialogue_preprocess", dialogue_preprocess)
    store.save_named_json("task_spec", task_spec.model_dump(mode="json"))
    store.save_named_json("asset_resolution", resolve_report.model_dump(mode="json"))
    store.save_named_json("capability_plan", capability.model_dump(mode="json"))
    store.save_named_json(
        "skill_retrieval",
        {
            "skills": retrieved_skills,
            "motifs": retrieved_motifs,
            "case_memory_size": len(case_memory.items),
        },
    )
    viz_meta = write_run_graph_visualizations(
        run_dir=run_dir,
        candidate_plan=candidate_plan,
        selected_graph=selected_graph_before_validate,
        validated_graph=graph,
        validation_issues=issues,
    )
    selected_idx = int(viz_meta["selected_index"]) if viz_meta["selected_index"] is not None else -1
    selected_ids = [
        c.graph.graph_id
        for c in candidate_plan.candidates
    ]
    selected_graph_id = selected_ids[selected_idx] if 0 <= selected_idx < len(selected_ids) else None
    candidate_payload = candidate_plan.model_dump(mode="json")
    candidate_payload["selected_index"] = selected_idx
    candidate_payload["selected_source"] = viz_meta["selected_source"]
    candidate_payload["selected_graph_id"] = selected_graph_id
    candidate_payload["graph_viz"] = viz_meta
    store.save_named_json("graph_candidates", candidate_payload)
    store.save_named_json("graph_validation", {"issues": issues})
    selected_source = str(viz_meta.get("selected_source") or "unknown")
    log_step("Graph Select", f"Selected candidate source={selected_source} (index={selected_idx})")
    if viz_meta.get("wrote_png"):
        log_step("Graph Viz", f"Candidate/selected graph PNGs saved to {viz_meta['out_dir']}")
    else:
        log_step("Graph Viz", f"DOT files saved to {viz_meta['out_dir']} (graphviz 'dot' not available)")
    if issues:
        log_step("Graph Viz", f"Validate issues present ({len(issues)}), wrote post-validate graph view")

    log_step("Execute Graph", "Running planned steps...")
    executor = GraphExecutor(store, context)
    executor.execute(graph, bundle, task_spec)
    judge = store.get("judge_report")

    record = RunRecord(
        run_id=run_id,
        run_dir=run_dir,
        task_spec=task_spec,
        capability_plan=capability,
        graph=graph,
        judge=judge,
    )
    store.save_named_json("run_record", record.model_dump(mode="json"))

    if judge and judge.final_score < 0.85:
        repair = RepairPlanner(qwen_client).plan(task_spec, judge)
        store.save_named_json("repair_patch", repair.model_dump(mode="json"))

    log_step("Run Completed", f"run_id={run_id} dry_run={dry_run}")
    log_kv(
        "Summary",
        {
            "task_type": task_spec.task_type,
            "num_frames": task_spec.num_frames,
            "characters": task_spec.character_names,
            "scenes": task_spec.scene_names,
            "judge_score": judge.final_score if judge else "n/a",
            "run_dir": str(run_dir),
        },
    )
    return record


@app.command()
def parse(prompt: str, dry_run: bool = True) -> None:
    """Only run parser."""
    if dry_run:
        client = MockQwenClient()
    else:
        from anime_pipeline_graph.providers.qwen_api_client import QwenApiClient

        client = QwenApiClient(AppConfig(), PROMPTS_DIR)
    spec = TaskParser(client).parse(make_bundle(prompt))
    typer.echo(spec.model_dump_json(indent=2))


@app.command("resolve-assets")
def resolve_assets(prompt: str) -> None:
    """Run parser + resolver."""
    spec = TaskParser(MockQwenClient()).parse(make_bundle(prompt))
    resolver = AssetResolver(EXAMPLES_DIR / "assets" / "characters", EXAMPLES_DIR / "assets" / "scenes")
    _, report = resolver.resolve(make_bundle(prompt), spec)
    typer.echo(report.model_dump_json(indent=2))


@app.command()
def plan(prompt: str, dry_run: bool = True) -> None:
    """Run parser/capability/graph planning."""
    if dry_run:
        client = MockQwenClient()
    else:
        from anime_pipeline_graph.providers.qwen_api_client import QwenApiClient

        client = QwenApiClient(AppConfig(), PROMPTS_DIR)
    spec = TaskParser(client).parse(make_bundle(prompt))
    cap = CapabilityPlanner(client).plan(spec)
    graph = GraphPlanner(client).plan(spec, cap, SkillRegistry().as_dict())
    typer.echo(graph.model_dump_json(indent=2))


@app.command()
def run(prompt: str, dry_run: bool = True) -> None:
    """Execute full pipeline."""
    root = Path(__file__).resolve().parents[3]
    record = run_pipeline(prompt, dry_run=dry_run, project_root=root)
    typer.echo(record.model_dump_json(indent=2))


@app.command()
def inspect(run_id: str) -> None:
    """Inspect run folder quickly."""
    root = Path(__file__).resolve().parents[3]
    run_dir = root / AppConfig().default_output_dir / run_id
    if not run_dir.exists():
        raise typer.BadParameter(f"run not found: {run_dir}")
    files = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file())
    typer.echo("\n".join(files))


@app.command()
def repair(run_id: str, dry_run: bool = True) -> None:
    """Plan repair from existing run's judge report."""
    root = Path(__file__).resolve().parents[3]
    run_dir = root / AppConfig().default_output_dir / run_id
    payload = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    if not payload.get("judge"):
        typer.echo("No judge report found")
        return
    if dry_run:
        client = MockQwenClient()
    else:
        from anime_pipeline_graph.providers.qwen_api_client import QwenApiClient

        client = QwenApiClient(AppConfig(), PROMPTS_DIR)
    from anime_pipeline_graph.domain.models import JudgeReport, TaskSpec

    patch = RepairPlanner(client).plan(TaskSpec.model_validate(payload["task_spec"]), JudgeReport.model_validate(payload["judge"]))
    typer.echo(patch.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
