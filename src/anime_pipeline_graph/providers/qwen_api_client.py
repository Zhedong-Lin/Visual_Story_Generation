"""Qwen API client over OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

from anime_pipeline_graph.config import AppConfig
from anime_pipeline_graph.utils.json_utils import safe_json_loads
from anime_pipeline_graph.utils.prompt_utils import load_prompt


class QwenApiClient:
    """Unified Qwen API client for all planning and judging tasks."""

    def __init__(self, config: AppConfig, prompts_dir: Path) -> None:
        self.config = config
        if not config.dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY is empty. Please set it in your environment or .env.")
        timeout_seconds = float(max(1, int(config.qwen_timeout_seconds)))
        max_retries = max(0, int(config.qwen_max_retries))
        self.client = OpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.qwen_base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self.model = config.qwen_model_name
        self.prompts_dir = prompts_dir

    def _ask_json(self, system_prompt_path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send prompt to Qwen and parse json response."""
        system_prompt = load_prompt(self.prompts_dir / system_prompt_path)
        t0 = time.monotonic()
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.2,
            )
        finally:
            elapsed = time.monotonic() - t0
            print(f"[Qwen] {system_prompt_path} model={self.model} elapsed={elapsed:.1f}s")
        text = completion.choices[0].message.content or "{}"
        try:
            return safe_json_loads(text)
        except Exception as exc:
            # Keep the raw LLM text available to callers that can recover
            # useful content from malformed JSON, especially storyboard frames.
            setattr(exc, "raw_text", text)
            raise

    def _ask_text(self, system_prompt_path: str, payload: Dict[str, Any]) -> str:
        """Send prompt to Qwen and return plain text."""
        system_prompt = load_prompt(self.prompts_dir / system_prompt_path)
        t0 = time.monotonic()
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.2,
            )
        finally:
            elapsed = time.monotonic() - t0
            print(f"[Qwen] {system_prompt_path} model={self.model} elapsed={elapsed:.1f}s")
        return (completion.choices[0].message.content or "").strip()

    def parse_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse user input into task spec dict."""
        return self._ask_json("parser_prompt.txt", payload)

    def plan_capabilities(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Plan capability toggles."""
        return self._ask_json("capability_prompt.txt", payload)

    def plan_graph(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Plan dynamic skill graph."""
        return self._ask_json("graph_planner_prompt.txt", payload)

    def decompose_story(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose story into frame specs."""
        return self._ask_json("story_decompose_prompt.txt", payload)

    def judge_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Judge generated result."""
        return self._ask_json("judge_prompt.txt", payload)

    def plan_repair(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create repair patch actions."""
        return self._ask_json("repair_prompt.txt", payload)

    def pose_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured pose plan."""
        return self._ask_json("pose_plan_prompt.txt", payload)

    def expression_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate per-frame expression plan."""
        return self._ask_json("expression_plan_prompt.txt", payload)

    def infer_storyboard_intent(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Infer whether ambiguous prompt should become multi-frame storyboard."""
        return self._ask_json("storyboard_intent_prompt.txt", payload)

    def shot_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate per-frame shot/camera plan."""
        return self._ask_json("shot_plan_prompt.txt", payload)

    def transition_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate frame-to-frame transition plan."""
        return self._ask_json("transition_plan_prompt.txt", payload)

    def continuity_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cross-frame continuity ledger."""
        return self._ask_json("continuity_state_prompt.txt", payload)

    def visual_style_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate style bible / visual style plan."""
        return self._ask_json("visual_style_plan_prompt.txt", payload)

    def dialogue_to_storyboard_prompt(self, payload: Dict[str, Any]) -> str:
        """Rewrite dialogue-heavy input into visual-only story text."""
        return self._ask_text("dialogue_rewrite_prompt.txt", payload)
