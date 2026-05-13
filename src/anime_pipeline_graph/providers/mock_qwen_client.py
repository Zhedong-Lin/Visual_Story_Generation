"""Dry-run mock Qwen client."""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List

from anime_pipeline_graph.constants import MAX_STORYBOARD_FRAMES


class MockQwenClient:
    """Mock implementation with deterministic structured outputs."""

    def dialogue_to_storyboard_prompt(self, payload: Dict[str, Any]) -> str:
        """Rule-based dialogue cleanup for dry-run tests."""
        text = str(payload.get("user_text", ""))
        cleaned = re.sub(r"[\"“”‘’][^\"“”‘’]{1,160}[\"“”‘’]", "", text)
        cleaned = re.sub(r"(?m)^\s*[-*]?\s*[\w\u4e00-\u9fff][\w\s\u4e00-\u9fff]{0,32}(?:\s*\([^)]{1,80}\))?\s*[:：]\s*", "", cleaned)
        cleaned = re.sub(r"\b(?!Frame\b|Panel\b)[A-Za-z_][A-Za-z0-9_]{0,23}\s*[:：]\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .。")
        if not cleaned:
            cleaned = "Characters act out the story visually with expressive poses and reactions, without speech text."
        target_frames = 3 if len(cleaned) < 500 else 6
        return "\n".join(f"Frame {i + 1}: {cleaned}" for i in range(target_frames))

    def _extract_names(self, text: str) -> tuple[List[str], List[str]]:
        candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
        scene_hits = [c for c in candidates if any(k in c.lower() for k in ["city", "street", "room", "forest", "school"]) ]
        chars = [c for c in candidates if c not in scene_hits and c.lower() not in {"in", "with", "and"}]
        return sorted(set(chars))[:4], sorted(set(scene_hits))[:2]

    def _infer_frames(self, text: str) -> int:
        lowered = text.lower()
        frame_label_hits = [
            int(m.group(1))
            for pat in [r"\bframe\s*(\d+)\b", r"\bpanel\s*(\d+)\b", r"第\s*(\d+)\s*[帧格张]"]
            for m in re.finditer(pat, lowered)
        ]
        if frame_label_hits:
            return max(1, min(max(frame_label_hits), MAX_STORYBOARD_FRAMES))
        chinese_numbers = [
            ("十五", 15),
            ("十四", 14),
            ("十三", 13),
            ("十二", 12),
            ("十一", 11),
            ("十", 10),
            ("九", 9),
            ("八", 8),
            ("七", 7),
            ("六", 6),
            ("五", 5),
            ("四", 4),
            ("三", 3),
            ("两", 2),
            ("二", 2),
            ("一", 1),
        ]
        for pat in [
            r"\b(\d+)\s*-\s*panel\b",
            r"\b(\d+)\s*panels?\b",
            r"\b(\d+)\s*-\s*frame\b",
            r"\b(\d+)\s*frames?\b",
            r"(\d+)\s*[帧格张]",
            r"(\d+)\s*个分镜",
        ]:
            m = re.search(pat, lowered)
            if m:
                return max(1, min(int(m.group(1)), MAX_STORYBOARD_FRAMES))
        for zh, value in chinese_numbers:
            if any(token in lowered for token in [f"{zh}帧", f"{zh}格", f"{zh}张", f"{zh}个分镜"]):
                return value
        if any(token in lowered for token in ["three"]):
            return 3
        if any(token in lowered for token in ["追", "打", "fight", "chase"]):
            return 2
        return 1

    def parse_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload["user_text"]
        chars, scenes = self._extract_names(text)
        num_frames = self._infer_frames(text)
        lowered = text.lower()
        edit_intent = any(k in lowered for k in ["change", "replace", "swap", "edit", "only change"])
        return {
            "task_id": f"task_{uuid.uuid4().hex[:8]}",
            "task_type": "single_image" if edit_intent else ("storyboard" if num_frames > 1 else "single_image"),
            "num_frames": 1 if edit_intent else num_frames,
            "num_characters": len(chars),
            "character_names": chars,
            "scene_names": scenes,
            "has_character_reference": bool(payload.get("character_references")),
            "has_scene_reference": bool(payload.get("scene_references")),
            "has_setting_doc": bool(payload.get("setting_docs")),
            "needs_identity_preservation": len(chars) > 0,
            "needs_local_editing": edit_intent,
            "needs_pose_control": (num_frames > 1) and not edit_intent,
            "needs_scene_generation": True,
            "needs_layout_control": (num_frames > 1) and not edit_intent,
            "needs_story_continuity": (num_frames > 1) and not edit_intent,
            "needs_multi_character_interaction": len(chars) > 1,
            "action_intensity": "high" if num_frames > 1 else "low",
            "edit_scope": "local" if edit_intent else "none",
            "scene_strength": "high" if scenes else "medium",
            "priority": "story" if num_frames > 1 else "quality",
            "character_constraints": ["keep identity consistent"],
            "scene_constraints": ["match reference scene when provided"],
            "story_constraints": ["frame-to-frame continuity"] if num_frames > 1 else [],
            "frame_descriptions": [f"Frame {i+1}: {text}" for i in range(1 if edit_intent else num_frames)],
            "risk_flags": ["missing_assets_possible"],
        }

    def plan_capabilities(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = payload["task_spec"]
        return {
            "identity_preservation": spec["needs_identity_preservation"],
            "local_editing": spec["needs_local_editing"],
            "pose_control": spec["needs_pose_control"],
            "scene_reference_conditioning": True,
            "story_continuity": spec["needs_story_continuity"],
            "multi_character_interaction": spec["needs_multi_character_interaction"],
            "quality_refinement": True,
        }

    def plan_graph(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task_spec = payload["task_spec"]
        cap = payload["capability_plan"]
        steps = []
        edges = []

        def add(step_id: str, skill: str, outputs: list[str]) -> None:
            steps.append({"step_id": step_id, "skill": skill, "inputs": {}, "params": {}, "outputs": outputs})

        if task_spec["num_frames"] > 1:
            add("story_1", "story_decompose", ["frame_specs"])
        if cap["identity_preservation"]:
            add("char_1", "character_bind", ["char_anchor"])
        add("scene_1", "scene_condition", ["scene_pack"])
        if cap["pose_control"]:
            add("pose_1", "pose_plan", ["pose_plan_json"])
        if task_spec.get("num_frames", 1) > 1:
            add("expr_1", "expression_plan", ["expression_plan_json"])
        if cap["pose_control"]:
            add("pose_2", "pose_extract", ["pose_skeleton"])
        add("prompt_1", "prompt_pack_builder", ["prompt_pack"])
        add("gen_1", "base_generation", ["generated_images"])
        add("judge_1", "judge", ["judge_report"])

        order = [s["step_id"] for s in steps]
        for i in range(len(order) - 1):
            edges.append((order[i], order[i + 1]))

        return {
            "graph_id": f"graph_{task_spec['task_id']}",
            "steps": steps,
            "edges": edges,
            "metadata": {"dynamic": True},
        }

    def decompose_story(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        frames = payload.get("num_frames", 2)
        text = payload.get("user_text", "")
        return {"frames": [{"frame_id": i + 1, "description": f"{text} - beat {i+1}"} for i in range(frames)]}

    def judge_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "final_score": 0.88,
            "subscores": {
                "instruction_match": 0.9,
                "identity_preservation": 0.87,
                "costume_accuracy": 0.85,
                "pose_accuracy": 0.86,
                "scene_match": 0.9,
                "story_consistency": 0.88,
            },
            "failure_tags": [],
            "repair_suggestions": [],
        }

    def plan_repair(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        tags = payload.get("failure_tags", [])
        actions = []
        if "identity" in tags:
            actions.append({"type": "rerun", "steps": ["character_bind", "base_generation"]})
        if "pose" in tags:
            actions.append({"type": "rerun", "steps": ["pose_plan", "base_generation"]})
        if "details" in tags:
            actions.append({"type": "rerun", "steps": ["edit"]})
        return {"reason": "rule_based_mock", "actions": actions}

    def pose_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "poses": [
                {
                    "subject": name,
                    "pose": "running" if i == 0 else "chasing",
                    "camera": "medium shot",
                    "notes": "dynamic anime motion",
                }
                for i, name in enumerate(payload.get("character_names", []))
            ]
        }

    def expression_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a simple expression sequence aligned to frames."""
        frame_descriptions = payload.get("frame_descriptions", []) or []
        num_frames = int(payload.get("num_frames", max(len(frame_descriptions), 1)) or 1)
        chars = payload.get("character_names", []) or []
        primary = chars[0] if chars else "character"
        items = []
        for i in range(num_frames):
            text = frame_descriptions[i].lower() if i < len(frame_descriptions) else ""
            expression = "neutral"
            if any(k in text for k in ["fright", "fear", "panic", "害怕", "惊慌"]):
                expression = "frightened, wide eyes"
            elif any(k in text for k in ["angry", "rage", "愤怒"]):
                expression = "angry, brows furrowed"
            elif any(k in text for k in ["sad", "cry", "悲伤"]):
                expression = "sad, teary eyes"
            elif any(k in text for k in ["happy", "smile", "开心"]):
                expression = "happy, slight smile"
            items.append(
                {
                    "frame_index": i + 1,
                    "subject": primary,
                    "expression": expression,
                    "intensity": "medium",
                    "notes": "keep expression readable on face",
                }
            )
        return {"expressions": items}

    def infer_storyboard_intent(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Mock semantic storyboard inference for ambiguous prompts."""
        text = str(payload.get("user_text", "")).lower()
        if "multi-beat storyboard narrative" in text:
            return {
                "should_storyboard": True,
                "suggested_num_frames": 3,
                "confidence": 0.76,
                "reason": "mock dialogue-cleanup storyboard marker",
            }
        seq_markers = [
            "then",
            "suddenly",
            "after",
            "before",
            "while",
            "最后",
            "然后",
            "接着",
            "突然",
            "先",
            "后",
            "之后",
        ]
        action_markers = [
            "walk",
            "run",
            "chase",
            "turn",
            "look",
            "escape",
            "react",
            "跑",
            "追",
            "走",
            "看",
            "逃",
        ]
        seq_hits = sum(1 for k in seq_markers if k in text)
        action_hits = sum(1 for k in action_markers if k in text)
        should_storyboard = (seq_hits >= 1 and action_hits >= 2) or seq_hits >= 2
        suggested = 1
        if should_storyboard:
            suggested = 3 if action_hits >= 3 else 2
        return {
            "should_storyboard": should_storyboard,
            "suggested_num_frames": suggested,
            "confidence": 0.72 if should_storyboard else 0.58,
            "reason": "mock semantic inference",
        }

    def shot_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a deterministic cinematic shot plan."""
        num_frames = int(payload.get("num_frames", 1) or 1)
        scales = ["LS", "MS", "MS", "CU", "MS", "CU"]
        angles = ["eye-level", "eye-level", "slight-low", "eye-level", "eye-level", "eye-level"]
        items = []
        for i in range(num_frames):
            items.append(
                {
                    "frame_index": i + 1,
                    "shot_scale": scales[i % len(scales)],
                    "camera_angle": angles[i % len(angles)],
                    "framing_intent": "single subject, rule-of-thirds bias",
                    "viewpoint_continuity": "maintain axis consistency",
                    "notes": "keep cinematic readability",
                }
            )
        return {"shots": items}

    def transition_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return deterministic transition metadata."""
        num_frames = int(payload.get("num_frames", 1) or 1)
        items = []
        for i in range(num_frames):
            if i == 0:
                t = "establish"
                c = {"action": "start", "space": "establish", "time": "continuous"}
            else:
                t = "same_action_cut"
                c = {"action": "continuous", "space": "continuous", "time": "continuous"}
            items.append(
                {
                    "frame_index": i + 1,
                    "transition_type": t,
                    "continuity_level": c,
                    "expected_deltas": "pose/camera can change, identity/outfit stay unless explicitly changed",
                }
            )
        return {"transitions": items}

    def continuity_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return lightweight continuity ledger."""
        chars = payload.get("character_names", []) or []
        num_frames = int(payload.get("num_frames", 1) or 1)
        ledger = []
        for i in range(num_frames):
            char_state = {
                c: {
                    "outfit": "unchanged",
                    "hair": "unchanged",
                    "expression_baseline": "track expression_plan",
                }
                for c in chars
            }
            ledger.append(
                {
                    "frame_index": i + 1,
                    "character_state": char_state,
                    "scene_state": {"time_of_day": "night", "weather": "rain/storm", "lighting": "consistent"},
                    "object_state": {},
                    "narrative_state": {"event_progress": i + 1, "continuity_anchor": "same story world"},
                }
            )
        return {"continuity_ledger": ledger}

    def visual_style_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return deterministic style bible."""
        return {
            "style_bible": {
                "art_style": "anime storyboard cinematic",
                "medium": "digital painting",
                "color_palette": "cool neon + warm indoor contrast",
                "lighting_tone": "stormy exterior, warm interior",
                "line_detail_density": "medium-high line clarity",
                "renderer_hints": {"consistency_priority": "high", "stylization_strength": "medium"},
            }
        }
