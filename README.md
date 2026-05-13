# anime_pipeline_graph

动漫图动态 skill graph planner 研究原型。这个系统是**动态图规划器**，不是固定 pipeline 模板选择器。

当前版本严格实现：
- Qwen 全部走 API（DashScope OpenAI 兼容）
- 图像生成支持两种后端可切换：
  - 本地 Hugging Face/diffusers（FLUX）
  - 豆包 SeedDream4 API（Ark）
- base generation 默认 `black-forest-labs/FLUX.1-dev`
- 角色 skill 使用 `Kontext + (optional LoRA) + 角色参考图`
- 场景 skill 使用场景参考图做 conditioning
- 编辑 skill：无 mask 用 Kontext，有 mask 用 Fill，无 mask 时 fill 自动回退 edit

## 1. 项目简介

系统输入：
- 用户文本 prompt
- 可选角色参考图
- 可选场景参考图
- 可选世界观文本
- 可选历史分镜图

系统输出：
- 单张动漫图，或
- 2~15 张短分镜动漫图

核心流程：
1. parser 把输入转 TaskSpec
2. capability planner 产出能力组合
3. skill library 检索候选 skills/motifs
4. candidate graph planner 生成多候选图（LLM/motif/fallback）
5. constraints + scoring 选最优图
6. repair-as-search 修复非法图（最小 best-first）
7. graph validator 兼容校验与补丁
8. executor 执行图并落盘所有中间结果
9. judge 评分
10. 可选 repair planner 输出 patch

## 2. 输入到输出完整流程

1. `PROMPT` 变量写在脚本里（不是交互输入）
2. parser（Qwen）输出 TaskSpec（含角色名/场景名/分镜数）
3. asset resolver 按名字自动匹配参考图
4. capability planner（Qwen）
5. graph planner（candidate-based）输出候选 graphs
6. constraints + scorer 选择最优 graph
7. repair search（必要时）修复结构
8. graph validator 校验并可 auto-fix
9. 前置条件构建：
   - `character_bind`
   - `scene_condition`
   - `pose_plan` / `pose_extract`
10. `base_generation` 第一次正式出图
11. 可选 `edit` / `fill_edit`
12. `judge`（Qwen）
13. 分数低则 `repair_planner`

## 3. 调用逻辑声明（关键）

- `character_bind` 与 `scene_condition` 是**前置条件构建器**，不是默认后处理器。
- `base_generation` 是第一次正式出图。
- `edit/fill_edit` 是可选修正步骤，不是默认循环补丁链。

## 4. 测试输入规则

Prompt 中识别到的角色/场景名会自动匹配同名图片：
- `lulu -> lulu.png`
- `jiddo -> jiddo.png`
- `cyber_city -> cyber_city.png`

目录：
- `examples/assets/characters/`
- `examples/assets/scenes/`

支持后缀：`.png .jpg .jpeg .webp`

## 5. 环境安装

```bash
conda create -n anime-pipeline-graph python=3.10 -y
conda activate anime-pipeline-graph
pip install -e .
```

说明：
- 跑本地 FLUX 前，请按你的 CUDA 版本安装匹配的 `torch`。
- Kontext/Fill 为 gated 模型，需要先在 Hugging Face 接受协议。
- 需要 `HF_TOKEN` 才能下载 gated 模型。

## 6. 依赖说明

- `pydantic`: 全部 schema 定义
- `typer`: CLI
- `rich`: 可读日志
- `python-dotenv`: 加载 `.env`
- `pillow`: mock 图片与基础图像处理
- `httpx`/`openai`: Qwen API 客户端
- `networkx`: DAG 校验/拓扑执行
- `pyyaml`: LoRA map 配置
- `jinja2`: 预留 prompt 模板扩展
- `diffusers`/`transformers`/`accelerate`: FLUX 推理
- `torch`/`torchvision`/`safetensors`: 推理底层
- `opencv-python`: 图像工具链兼容
- `controlnet-aux`: OpenPose skeleton 提取
- `numpy`: 数值处理
- `pytest`: 测试

## 7. 环境变量

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

`IMAGE_BACKEND` 说明：
- `local`：走本地 FLUX + Kontext + (optional LoRA)
- `seedream4_api`：走豆包 SeedDream4 API（当前按 text2img 路径运行；LoRA 不启用）

## 8. Dry-run

```bash
python scripts/run_demo_dry.py
```

特点：
- 不需要 API key
- 使用 `MockQwenClient`
- 使用 mock image providers（Pillow 生成带 step 标识图）
- 保存完整中间 JSON/图片

## 9. Real mode

```bash
python scripts/run_demo_real.py
```

要求：
- 配置 Qwen API key
- 本地可加载 FLUX/Kontext/Fill 模型
- 具备 Hugging Face 权限和 `HF_TOKEN`

使用豆包 SeedDream4 API（示例）：

```bash
export IMAGE_BACKEND=seedream4_api
export ARK_API_KEY=你的key
export ARK_SEEDREAM_MODEL=ep-xxxxxx
python scripts/run_baseline_prompt_fixed_real.py --backend seedream4_api
python scripts/run_lulu_single_lora_real.py --backend seedream4_api
```

## 10. 查看 runs 中间结果

```bash
python scripts/inspect_run.py <run_id>
```

或 CLI：

```bash
anime-pipeline inspect <run_id>
```

## 11. 已知限制

- FLUX 显存占用高
- Kontext/Fill 需要 gated access
- 场景 skill 当前仅做 reference conditioning，不是独立背景模型
- Pose skill 当前是“规划 + 辅助骨架”，不是完整动作控制生成器

## 12. 角色 LoRA 说明

- LoRA 目录预留：`examples/lora/`
- 可选配置：`examples/lora_map.yaml` 或 `examples/lora_map.json`
- 配置格式示例：

```yaml
lulu: examples/lora/lulu.safetensors
jiddo: examples/lora/jiddo.safetensors
```

当前版本身份一致性来源：
- 有 LoRA 时：`LoRA + 角色参考图`
- 无 LoRA 时：`仅角色参考图`

LoRA 是增强项，不是必须项。

## 13. CLI

```bash
anime-pipeline parse --prompt "lulu在cyber_city被jiddo追"
anime-pipeline resolve-assets --prompt "lulu在cyber_city被jiddo追"
anime-pipeline plan --prompt "lulu在cyber_city被jiddo追"
anime-pipeline run --prompt "lulu在cyber_city被jiddo追" --dry-run
anime-pipeline inspect <run_id>
anime-pipeline repair <run_id> --dry-run
```

## 14. 测试

```bash
pytest -q
```

## 15. 目录结构

见本文开头与仓库文件树；所有关键中间产物会进入 `runs/<run_id>/`。

## 16. 开发说明

Graph-of-Skills 增量改造说明文档：
- `docs/graph_of_skills_planner.md`
