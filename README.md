# Storyboard Generation

A research prototype for dynamic skill graph planning in anime-style visual story generation. This system is a **dynamic graph planner**, not a fixed pipeline template selector.

The current version implements:
- All Qwen calls through API access using the DashScope OpenAI-compatible interface.
- Switchable image generation backends:
  - Local Hugging Face / diffusers backend with FLUX.
  - Doubao SeedDream4 API backend through Ark.
- Default base generation model: `black-forest-labs/FLUX.1-dev`.
- Character skills using `Kontext + optional LoRA + character reference images`.
- Scene skills using scene reference images for conditioning.
- Editing skills:
  - Without a mask: use Kontext.
  - With a mask: use Fill.
  - If Fill is requested without a mask, the system automatically falls back to edit mode.

## 1. Project Overview

### System Inputs

The system supports the following inputs:

- User text prompt.
- Optional character reference images.
- Optional scene reference images.
- Optional world-building text.
- Optional historical storyboard frames.

### System Outputs

The system can generate:

- A single anime-style image, or
- A short anime storyboard with 2 to 15 frames.

### Core Workflow

1. The parser converts the input into a `TaskSpec`.
2. The capability planner produces a required capability combination.
3. The skill library retrieves candidate skills and motifs.
4. The candidate graph planner generates multiple candidate graphs from LLM, motif, memory, or fallback sources.
5. Constraints and scoring select the best graph.
6. Repair-as-search fixes invalid graphs using a minimal best-first strategy.
7. The graph validator performs compatibility checks and patches.
8. The executor runs the graph and saves all intermediate outputs.
9. The judge evaluates the result.
10. An optional repair planner outputs a patch when needed.

## 2. End-to-End Input-to-Output Workflow

1. The `PROMPT` variable is written inside the script rather than entered interactively.
2. The parser, powered by Qwen, outputs a `TaskSpec`, including character names, scene names, and the number of storyboard frames.
3. The asset resolver automatically matches reference images by name.
4. The capability planner, powered by Qwen, identifies required capabilities.
5. The candidate-based graph planner outputs candidate graphs.
6. Constraints and the scorer select the best graph.
7. Repair search fixes the graph structure when necessary.
8. The graph validator checks the graph and can auto-fix compatible issues.
9. Precondition-building skills are executed:
   - `character_bind`
   - `scene_condition`
   - `pose_plan` / `pose_extract`
10. `base_generation` produces the first formal image output.
11. Optional `edit` / `fill_edit` steps refine the result.
12. The Qwen-based `judge` evaluates the output.
13. If the score is low, the `repair_planner` proposes a patch.

## 3. Execution Logic

The following design choices are important:

- `character_bind` and `scene_condition` are **precondition builders**, not default post-processing modules.
- `base_generation` is the first formal image generation step.
- `edit` and `fill_edit` are optional correction steps, not a default iterative patch chain.

## 4. Test Input Rules

Character and scene names recognized in the prompt are automatically matched with images of the same name:

- `lulu -> lulu.png`
- `jiddo -> jiddo.png`
- `cyber_city -> cyber_city.png`

Reference image directories:

- `examples/assets/characters/`
- `examples/assets/scenes/`

Supported image extensions:

- `.png`
- `.jpg`
- `.jpeg`
- `.webp`

## 5. Installation

```bash
conda create -n anime-pipeline-graph python=3.10 -y
conda activate anime-pipeline-graph
pip install -e .
```

Notes:

- Before running local FLUX models, install the version of `torch` that matches your CUDA environment.
- Kontext and Fill are gated models. You need to accept the model agreements on Hugging Face before using them.
- `HF_TOKEN` is required to download gated models.

## 6. Dependencies

- `pydantic`: schema definitions.
- `typer`: command-line interface.
- `rich`: readable logging.
- `python-dotenv`: loading `.env` files.
- `pillow`: mock image generation and basic image processing.
- `httpx` / `openai`: Qwen API client.
- `networkx`: DAG validation and topological execution.
- `pyyaml`: LoRA map configuration.
- `jinja2`: reserved for prompt template expansion.
- `diffusers` / `transformers` / `accelerate`: FLUX inference.
- `torch` / `torchvision` / `safetensors`: inference backend.
- `opencv-python`: image processing compatibility.
- `controlnet-aux`: OpenPose skeleton extraction.
- `numpy`: numerical processing.
- `pytest`: testing.

## 7. Environment Variables

```env
DASHSCOPE_API_KEY=
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL_NAME=qwen-vl-max-latest
HF_TOKEN=
IMAGE_BACKEND=local
FLUX_BASE_MODEL=black-forest-labs/FLUX.1-dev
FLUX_KONTEXT_MODEL=black-forest-labs/FLUX.1-Kontext-dev
FLUX_FILL_MODEL=black-forest-labs/FLUX.1-Fill-dev
ARK_API_KEY=
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_IMAGES_ENDPOINT=/images/generations
ARK_SEEDREAM_MODEL=ep-xxxxxx
ARK_TIMEOUT_SECONDS=120
ARK_RESPONSE_FORMAT=url
ARK_DEFAULT_SIZE=1024x1024
DEFAULT_OUTPUT_DIR=runs
```

### `IMAGE_BACKEND` Options

- `local`: use local FLUX + Kontext + optional LoRA.
- `seedream4_api`: use Doubao SeedDream4 API. The current path runs text-to-image generation, and LoRA is not enabled.

## 8. Dry Run

```bash
python scripts/run_demo_dry.py
```

Features:

- Does not require API keys.
- Uses `MockQwenClient`.
- Uses mock image providers based on Pillow, generating images with step labels.
- Saves complete intermediate JSON files and images.

## 9. Real Mode

```bash
python scripts/run_demo_real.py
```

Requirements:

- Qwen API key.
- A local environment capable of loading FLUX, Kontext, and Fill models.
- Hugging Face access permission and `HF_TOKEN`.

Example using Doubao SeedDream4 API:

```bash
export IMAGE_BACKEND=seedream4_api
export ARK_API_KEY=your_key
export ARK_SEEDREAM_MODEL=ep-xxxxxx
python scripts/run_baseline_prompt_fixed_real.py --backend seedream4_api
python scripts/run_lulu_single_lora_real.py --backend seedream4_api
```

## 10. Inspecting Intermediate Run Results

```bash
python scripts/inspect_run.py <run_id>
```

Or through the CLI:

```bash
anime-pipeline inspect <run_id>
```

## 11. Known Limitations

- FLUX requires a large amount of GPU memory.
- Kontext and Fill require gated model access.
- The current scene skill only performs reference conditioning; it is not an independent background generation model.
- The pose skill currently performs planning and auxiliary skeleton extraction; it is not a full motion-controlled generation module.

## 12. Character LoRA Notes

- Reserved LoRA directory: `examples/lora/`
- Optional configuration file:
  - `examples/lora_map.yaml`
  - `examples/lora_map.json`

Example configuration:

```yaml
lulu: examples/lora/lulu.safetensors
jiddo: examples/lora/jiddo.safetensors
```

Current identity consistency sources:

- With LoRA: `LoRA + character reference image`.
- Without LoRA: `character reference image only`.

LoRA is an enhancement, not a required component.

## 13. CLI Usage

```bash
anime-pipeline parse --prompt "lulu is chased by jiddo in cyber_city"
anime-pipeline resolve-assets --prompt "lulu is chased by jiddo in cyber_city"
anime-pipeline plan --prompt "lulu is chased by jiddo in cyber_city"
anime-pipeline run --prompt "lulu is chased by jiddo in cyber_city" --dry-run
anime-pipeline inspect <run_id>
anime-pipeline repair <run_id> --dry-run
```

## 14. Testing

```bash
pytest -q
```

## 15. Directory Structure

See the repository file tree. All key intermediate artifacts are saved under:

```bash
runs/<run_id>/
```

## 16. Development Notes

Graph-of-Skills incremental refactoring documentation:

- `docs/graph_of_skills_planner.md`
