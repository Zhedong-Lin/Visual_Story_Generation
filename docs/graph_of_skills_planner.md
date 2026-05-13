# Graph-of-Skills Planning Framework (Incremental Upgrade)

## New Planning Flow

`run_pipeline` now follows this planning sequence while keeping executor compatibility:

1. Parse task (`TaskParser`)
2. Resolve assets (`AssetResolver`)
3. Build `SkillLibrary` (registry + yaml/json overrides)
4. Retrieve candidate skills and motifs
5. Generate candidate graphs (`CandidateGraphPlanner`):
   - `llm_proposal`
   - `motif_assembly`
   - `fallback`
6. Validate candidates with explicit constraints
7. Score candidates (`GraphScorer`) and select best
8. Repair invalid best candidate via graph edit search (`BestFirstGraphRepair`)
9. Execute final graph with existing `GraphExecutor`
10. Judge and optional post-run repair planner (legacy path preserved)

## Skill Library Schema

`SkillCard` fields:

- `name`
- `description`
- `skill_type`
- `applicability_conditions`
- `input_schema`
- `output_schema`
- `preconditions`
- `effects`
- `allowed_predecessors`
- `allowed_successors`
- `failure_modes`
- `graph_motifs`

Skill cards can be loaded from `yaml/json` and override registry-derived defaults.

Current default data file:

- `src/anime_pipeline_graph/skills/library/skill_cards.yaml`

## Constraint System

Constraint base class: `GraphConstraint`

Violation model: `Violation`

Implemented constraints:

- `MustHaveJudgeConstraint`
- `MultiFrameNeedsGenerationConstraint`
- `PromptBuilderBeforeGenerationConstraint`
- `TypeCompatibleEdgesConstraint`
- `PreconditionsSatisfiedConstraint`
- `GraphMustBeDAGConstraint`
- `NoOrphanCriticalNodesConstraint`

Primary API:

- `ConstraintValidator.validate(graph, task_spec, skill_library) -> list[Violation]`

Compatibility API remains:

- `GraphValidator.validate(graph, auto_fix=True) -> (graph, issues)`

## Candidate Graph Generation

Module: `planner/candidate_graph_planner.py`

Output model:

- `CandidateGraphPlan`
- `CandidateGraph`

Sources:

- LLM structured proposal (`plan_graph` JSON)
- motif/template assembly
- deterministic fallback

## Typed Graph Schema

Typed node fields are added to `GraphStep` (backward compatible):

- `step_id`
- `skill` (legacy)
- `skill_name`
- `skill_type`
- `frame_scope`
- `inputs_required`
- `outputs_produced`
- `optional`

Typed edge model:

- `GraphEdge(source, target, edge_type)` where `edge_type in {"data", "control"}`

Adapters in `planner/typed_graph.py` provide legacy tuple edge conversion for executor compatibility.

## Graph Scoring

Module: `planner/graph_scorer.py`

Sub-scores:

- `coverage_score`
- `validity_score`
- `prior_score`
- `cost_score`

Final score:

- weighted sum used for candidate selection

Diagnostics include violation codes and graph complexity stats.

## Repair as Search

Module: `planner/graph_repair_search.py`

Edit operators:

- `AddNode`
- `RemoveNode`
- `AddEdge`
- `RemoveEdge`
- `ReplaceNode`
- `InsertMotif`

Search strategy:

- best-first with beam limiting
- objective combines remaining violations, edit cost, and graph complexity

## Case Memory / Graph Prior (Minimal)

Module: `planner/case_memory.py`

Reads historical runs from `runs/` and `runs_baseline/` and stores:

- task summary
- graph summary
- outcome (`success`, `final_score`, `repair_count`, `failure_modes`)

Retrieval:

- `retrieve_similar_cases(task_spec, top_k)`

Used by `GraphScorer.prior_score` as a lightweight graph prior.

## New Run Artifacts

`run_pipeline` now saves:

- `skill_retrieval.json`
- `graph_candidates.json`

Existing outputs are unchanged.

## Minimal Demo

Existing scripts still work:

- `python scripts/run_baseline_prompt_fixed_real.py`
- `python scripts/run_lulu_single_lora_real.py`

For dry run:

- `python scripts/run_demo_dry.py`

