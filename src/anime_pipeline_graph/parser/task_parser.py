"""Task parser adapter."""

from __future__ import annotations

import uuid
from typing import Any
import re

from anime_pipeline_graph.constants import MAX_STORYBOARD_FRAMES
from anime_pipeline_graph.domain.enums import TaskType
from anime_pipeline_graph.domain.models import InputBundle, TaskSpec
from anime_pipeline_graph.parser.dialogue_preprocessor import MAX_AUTO_DIALOGUE_FRAMES, SPEAKER_LINE_RE


class TaskParser:
    """Parse InputBundle into TaskSpec via Qwen client."""

    def __init__(
        self,
        qwen_client: Any,
        known_character_names: list[str] | None = None,
        known_scene_names: list[str] | None = None,
    ) -> None:
        self.qwen_client = qwen_client
        self.known_character_names = [n.lower() for n in (known_character_names or [])]
        self.known_scene_names = [n.lower() for n in (known_scene_names or [])]

    def _fallback_extract_names(self, text: str) -> tuple[list[str], list[str]]:
        """Fallback extraction for character/scene names."""
        lowered = text.lower()
        chars = [n for n in self.known_character_names if n in lowered]
        scenes = [n for n in self.known_scene_names if n in lowered]

        if not chars:
            tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
            stop = {
                "frame",
                "generate",
                "scene",
                "storyboard",
                "panel",
                "with",
                "and",
                "the",
                "a",
                "an",
            }
            seen = set()
            extracted = []
            for tok in tokens:
                lt = tok.lower()
                if lt in stop or lt in seen:
                    continue
                seen.add(lt)
                extracted.append(lt)
            chars = extracted[:4]
        return chars, scenes

    def _sanitize_entities(self, normalized: dict, bundle: InputBundle) -> dict:
        """Sanitize parser-produced entity lists to reduce false-positive character names."""
        raw_chars = [str(x).strip().lower() for x in (normalized.get("character_names") or []) if str(x).strip()]
        raw_scenes = [str(x).strip().lower() for x in (normalized.get("scene_names") or []) if str(x).strip()]
        raw_chars = list(dict.fromkeys(raw_chars))
        raw_scenes = list(dict.fromkeys(raw_scenes))

        char_set = set(self.known_character_names)
        scene_set = set(self.known_scene_names)

        # Conservative non-character vocabulary for common parser mistakes in narrative text.
        generic_stop = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "with",
            "without",
            "to",
            "in",
            "on",
            "at",
            "from",
            "into",
            "through",
            "then",
        }
        action_words = {
            "cross",
            "crosses",
            "walk",
            "walks",
            "run",
            "runs",
            "look",
            "looks",
            "see",
            "sees",
            "mistake",
            "mistakes",
            "snap",
            "snaps",
            "drop",
            "drops",
            "lose",
            "loses",
        }
        object_scene_words = {
            "bridge",
            "bone",
            "water",
            "river",
            "reflection",
            "mouth",
            "street",
            "city",
            "room",
            "forest",
            "school",
            "home",
        }
        deny_words = generic_stop | action_words | object_scene_words

        def _valid_character(name: str) -> bool:
            if not name:
                return False
            if name in char_set:
                return True
            if name in scene_set:
                return False
            if name in deny_words:
                return False
            # Basic guard for obvious English verb forms when not in known characters.
            if name.endswith("ing") and len(name) > 4:
                return False
            return True

        clean_chars = [c for c in raw_chars if _valid_character(c)]
        clean_scenes = [s for s in raw_scenes if s and s not in char_set and s not in generic_stop]

        # If all characters were filtered out, keep one likely protagonist token as fallback.
        if not clean_chars:
            lowered = bundle.user_text.lower()
            m = re.search(r"\b(?:a|an|the)\s+([a-z_][a-z0-9_]*)\b", lowered)
            if m:
                candidate = m.group(1).strip().lower()
                if _valid_character(candidate):
                    clean_chars = [candidate]

        normalized["character_names"] = clean_chars
        normalized["scene_names"] = clean_scenes
        normalized["num_characters"] = len(clean_chars)
        normalized["needs_identity_preservation"] = len(clean_chars) > 0
        normalized["needs_multi_character_interaction"] = len(clean_chars) > 1
        return normalized

    def _detect_edit_intent(self, text: str) -> bool:
        """Detect likely local-edit intent from natural language."""
        lowered = text.lower()
        keywords = [
            "edit",
            "existing image",
            "keep pose",
            "keep background",
            "only change",
            "change ",
            "replace ",
            "swap ",
            "modify ",
            "换",
            "替换",
            "改成",
            "保持不变",
            "只改",
            "wearing",
            "wears",
            "dressed in",
            "change outfit",
            "outfit to",
            "put on",
            "换装",
            "换衣服",
            "换上",
            "穿上",
            "穿着",
            "身穿",
            "abaya",
            "hijab",
        ]
        return any(k in lowered for k in keywords)

    def _detect_pose_intent(self, text: str) -> bool:
        """Detect whether user explicitly asks for pose/action changes."""
        lowered = text.lower()
        pose_keywords = [
            "pose",
            "posture",
            "sitting",
            "standing",
            "running",
            "walking",
            "jumping",
            "kneeling",
            "raise",
            "turn",
            "arms",
            "legs",
            "坐",
            "站",
            "跑",
            "走",
            "抬手",
            "姿势",
            "动作",
        ]
        return any(k in lowered for k in pose_keywords)

    def _infer_explicit_frame_count(self, text: str) -> int | None:
        """Infer frame count from labels such as "Frame 10:" or "第10帧"."""
        lowered = text.lower()
        hits: list[int] = []
        for pat in [
            r"\bframe\s*(\d+)\b",
            r"\bpanel\s*(\d+)\b",
            r"第\s*(\d+)\s*[帧格张]",
        ]:
            hits.extend(int(m.group(1)) for m in re.finditer(pat, lowered))
        if hits:
            return max(1, min(max(hits), MAX_STORYBOARD_FRAMES))
        return None

    def _resolve_source_image(self, normalized: dict, bundle: InputBundle) -> str | None:
        """Resolve source image path from constraints map and recognized names."""
        constraints = bundle.constraints or {}
        edit_map = constraints.get("edit_source_map", {})
        if not isinstance(edit_map, dict) or not edit_map:
            return None
        names = [str(x).lower() for x in (normalized.get("character_names") or [])]
        for name in names:
            if name in edit_map:
                return str(edit_map[name])
        # Fallback: search by token mention in user text.
        lowered = bundle.user_text.lower()
        for name, path in edit_map.items():
            if str(name).lower() in lowered:
                return str(path)
        return None

    def _coerce_qwen_mapping(self, data: Any, bundle: InputBundle) -> dict:
        """Coerce loose LLM output into a mapping when possible."""
        if isinstance(data, dict):
            candidate = data.get("task_spec")
            return candidate if isinstance(candidate, dict) else data

        if isinstance(data, list):
            # Accept list-of-pairs forms such as [["task_type", "storyboard"], ...].
            if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in data):
                try:
                    return dict(data)
                except Exception:
                    pass
            # Accept single wrapped object forms such as [ {...} ].
            if len(data) == 1 and isinstance(data[0], dict):
                candidate = data[0].get("task_spec")
                return candidate if isinstance(candidate, dict) else data[0]

        # Fall back to the parser's own heuristics instead of crashing the run.
        fb_chars, fb_scenes = self._fallback_extract_names(bundle.user_text)
        explicit_frames = self._infer_explicit_frame_count(bundle.user_text) or 1
        is_storyboard = explicit_frames > 1
        return {
            "task_id": f"task_{uuid.uuid4().hex[:8]}",
            "task_type": TaskType.STORYBOARD.value if is_storyboard else TaskType.SINGLE_IMAGE.value,
            "num_frames": explicit_frames,
            "num_characters": len(fb_chars),
            "character_names": fb_chars,
            "scene_names": fb_scenes,
            "frame_descriptions": [bundle.user_text] if explicit_frames == 1 else [],
            "risk_flags": ["qwen_parse_shape_fallback"],
        }

    def _normalize_qwen_output(self, data: Any, bundle: InputBundle) -> dict:
        """Normalize non-strict Qwen payload into TaskSpec-compatible dict."""
        normalized = self._coerce_qwen_mapping(data, bundle)

        if "task_id" not in normalized:
            normalized["task_id"] = f"task_{uuid.uuid4().hex[:8]}"

        num_frames = int(normalized.get("num_frames", 1) or 1)
        if num_frames < 1:
            num_frames = 1
        num_frames = min(num_frames, MAX_STORYBOARD_FRAMES)
        explicit_count = self._infer_explicit_frame_count(bundle.user_text)
        if explicit_count and explicit_count > num_frames:
            num_frames = explicit_count
        frame_desc = normalized.get("frame_descriptions", [])
        if isinstance(frame_desc, list) and num_frames == 1 and len(frame_desc) > 1:
            num_frames = min(len(frame_desc), MAX_STORYBOARD_FRAMES)
        normalized["num_frames"] = num_frames

        task_type_raw = str(normalized.get("task_type", "")).strip().lower()
        if task_type_raw in {"multi_image", "multi-image", "multiimage", "sequence", "comic"}:
            normalized["task_type"] = TaskType.STORYBOARD.value
        elif task_type_raw in {"single", "single_frame", "single-frame", "one_image", "one-image"}:
            normalized["task_type"] = TaskType.SINGLE_IMAGE.value

        if "task_type" not in normalized:
            role = str(normalized.get("role", "")).lower()
            if num_frames > 1 or "storyboard" in role:
                normalized["task_type"] = TaskType.STORYBOARD.value
            else:
                normalized["task_type"] = TaskType.SINGLE_IMAGE.value

        frame_desc = normalized.get("frame_descriptions", [])
        if isinstance(frame_desc, list):
            converted: list[str] = []
            for item in frame_desc:
                if isinstance(item, str):
                    converted.append(item)
                elif isinstance(item, dict):
                    desc = item.get("description") or item.get("desc") or item.get("text")
                    if desc:
                        converted.append(str(desc))
            normalized["frame_descriptions"] = converted
            if len(converted) > int(normalized.get("num_frames", 1) or 1):
                normalized["num_frames"] = min(len(converted), MAX_STORYBOARD_FRAMES)

        if not normalized.get("frame_descriptions"):
            normalized["frame_descriptions"] = [bundle.user_text]

        chars = normalized.get("character_names") or []
        scenes = normalized.get("scene_names") or []
        if not chars and not scenes:
            fb_chars, fb_scenes = self._fallback_extract_names(bundle.user_text)
            normalized["character_names"] = fb_chars
            normalized["scene_names"] = fb_scenes
        else:
            normalized["character_names"] = [str(c).lower() for c in chars]
            normalized["scene_names"] = [str(s).lower() for s in scenes]

        normalized = self._sanitize_entities(normalized, bundle)

        normalized.setdefault("num_characters", len(normalized.get("character_names", [])))
        normalized.setdefault("source_image", None)
        normalized.setdefault("has_character_reference", False)
        normalized.setdefault("has_scene_reference", False)
        normalized.setdefault("has_setting_doc", bool(bundle.setting_docs))
        normalized.setdefault("needs_identity_preservation", bool(normalized.get("character_names")))
        normalized.setdefault("needs_local_editing", False)
        normalized.setdefault("needs_pose_control", normalized["num_frames"] > 1)
        normalized.setdefault("needs_scene_generation", True)
        normalized.setdefault("needs_layout_control", normalized["num_frames"] > 1)
        normalized.setdefault("needs_story_continuity", normalized["num_frames"] > 1)
        normalized.setdefault("needs_multi_character_interaction", normalized.get("num_characters", 0) > 1)
        normalized.setdefault("action_intensity", "medium")
        normalized.setdefault("edit_scope", "none")
        normalized.setdefault("scene_strength", "medium")
        normalized.setdefault("priority", "quality")
        normalized.setdefault("character_constraints", [])
        normalized.setdefault("scene_constraints", [])
        normalized.setdefault("story_constraints", [])
        normalized.setdefault("risk_flags", [])

        # Soft fallback: if parser misses clear edit intent, infer with lightweight semantics.
        source_image = normalized.get("source_image") or self._resolve_source_image(normalized, bundle)
        if source_image:
            normalized["source_image"] = source_image
        inferred_edit = self._detect_edit_intent(bundle.user_text)
        if source_image and inferred_edit:
            normalized["needs_local_editing"] = True
            edit_scope = str(normalized.get("edit_scope", "none")).lower()
            normalized["edit_scope"] = "local" if edit_scope == "none" else edit_scope
            # Keep storyboard tasks as storyboard when multi-frame intent already exists.
            # Only collapse to single-image for true local single-frame edits.
            task_type_now = str(normalized.get("task_type", "")).lower()
            num_frames_now = int(normalized.get("num_frames", 1) or 1)
            explicit_frame = self._has_explicit_frame_signal(bundle.user_text)
            keep_storyboard = (
                task_type_now == TaskType.STORYBOARD.value
                or num_frames_now > 1
                or explicit_frame
            )
            if not keep_storyboard:
                normalized["task_type"] = TaskType.SINGLE_IMAGE.value
                normalized["num_frames"] = 1
                pose_intent = self._detect_pose_intent(bundle.user_text)
                normalized["needs_pose_control"] = pose_intent
                normalized["needs_layout_control"] = False
                normalized["needs_story_continuity"] = False
                normalized["needs_multi_character_interaction"] = False
            else:
                normalized["task_type"] = TaskType.STORYBOARD.value
                normalized["needs_pose_control"] = True
                normalized["needs_layout_control"] = True
                normalized["needs_story_continuity"] = True
        elif source_image and not inferred_edit:
            # Source image alone should not force local editing.
            normalized["needs_local_editing"] = False
            normalized["edit_scope"] = "none"

        # Parser models sometimes return single_image while still emitting multiple
        # frame descriptions. Keep the structured count and task type coherent.
        if int(normalized.get("num_frames", 1) or 1) > 1 and not normalized.get("needs_local_editing", False):
            normalized["task_type"] = TaskType.STORYBOARD.value
            normalized["needs_pose_control"] = True
            normalized["needs_layout_control"] = True
            normalized["needs_story_continuity"] = True

        return normalized

    def _has_explicit_frame_signal(self, text: str) -> bool:
        """Whether user explicitly requests frame/panel count."""
        lowered = text.lower()
        patterns = [
            r"\b\d+\s*-\s*panel\b",
            r"\b\d+\s*panels?\b",
            r"\b\d+\s*-\s*frame\b",
            r"\b\d+\s*frames?\b",
            r"\bframe\s*\d+\b",
            r"\bpanel\s*\d+\b",
            r"\d+\s*帧",
            r"\d+\s*格",
            r"[一二两三四五六七八九十]+\s*[帧格张]",
            r"[一二两三四五六七八九十]+\s*个分镜",
            r"分镜",
            r"四格",
            r"三格",
            r"多帧",
        ]
        return any(re.search(p, lowered) for p in patterns)

    def _semantic_storyboard_refine(self, normalized: dict, bundle: InputBundle) -> dict:
        """Use LLM semantic judgment for ambiguous 1-frame non-edit prompts."""
        if normalized.get("needs_local_editing"):
            return normalized
        if self._has_explicit_frame_signal(bundle.user_text):
            return normalized
        if int(normalized.get("num_frames", 1) or 1) > 1:
            return normalized

        infer_fn = getattr(self.qwen_client, "infer_storyboard_intent", None)
        if not callable(infer_fn):
            return normalized

        payload = {
            "user_text": bundle.user_text,
            "current_parse": {
                "task_type": normalized.get("task_type"),
                "num_frames": normalized.get("num_frames", 1),
                "frame_descriptions": normalized.get("frame_descriptions", []),
                "needs_local_editing": normalized.get("needs_local_editing", False),
            },
        }
        try:
            out = infer_fn(payload) or {}
        except Exception:
            return normalized

        should_storyboard = bool(out.get("should_storyboard", False))
        if not should_storyboard:
            return normalized
        try:
            suggested = int(out.get("suggested_num_frames", 2))
        except Exception:
            suggested = 2
        suggested = max(2, min(suggested, MAX_STORYBOARD_FRAMES))
        normalized["num_frames"] = suggested
        normalized["task_type"] = TaskType.STORYBOARD.value
        normalized["needs_pose_control"] = True
        normalized["needs_layout_control"] = True
        normalized["needs_story_continuity"] = True

        frame_desc = normalized.get("frame_descriptions", [])
        if not isinstance(frame_desc, list):
            frame_desc = [bundle.user_text]
        frame_desc = [str(x) for x in frame_desc if str(x).strip()]
        if len(frame_desc) <= 1:
            frame_desc = [bundle.user_text for _ in range(suggested)]
        elif len(frame_desc) < suggested:
            frame_desc.extend([frame_desc[-1]] * (suggested - len(frame_desc)))
        normalized["frame_descriptions"] = frame_desc[:suggested]
        return normalized

    def _dialogue_storyboard_frame_refine(self, normalized: dict, bundle: InputBundle) -> dict:
        """Prevent dialogue-cleaned long stories from collapsing into too few frames."""
        constraints = bundle.constraints or {}
        preprocess = constraints.get("dialogue_preprocess") or {}
        if not isinstance(preprocess, dict) or not preprocess.get("preprocessed"):
            return normalized
        if self._has_explicit_frame_signal(str(preprocess.get("original_prompt") or "")):
            return normalized

        original = str(preprocess.get("original_prompt") or constraints.get("original_user_text") or "")
        if not original:
            return normalized

        lines = [line.strip() for line in original.splitlines() if line.strip()]
        speaker_count = sum(1 for line in lines if SPEAKER_LINE_RE.search(line))
        action_count = sum(
            1
            for line in lines
            if not SPEAKER_LINE_RE.search(line)
            and re.search(
                r"\b(scene|everyone|laughs?|sitting|standing|hangs up|turns|boat|trip|looks?|walks?|go(?:ing)?)\b",
                line,
                re.I,
            )
        )
        if speaker_count + action_count < 6 and len(original) < 900:
            return normalized

        current = int(normalized.get("num_frames", 1) or 1)
        recommended = max(5, min(MAX_AUTO_DIALOGUE_FRAMES, round((speaker_count + action_count) / 2)))
        if len(original) > 900:
            recommended = max(recommended, min(6, MAX_AUTO_DIALOGUE_FRAMES))
        if current >= recommended:
            return normalized

        normalized["num_frames"] = recommended
        normalized["task_type"] = TaskType.STORYBOARD.value
        normalized["needs_pose_control"] = True
        normalized["needs_layout_control"] = True
        normalized["needs_story_continuity"] = True
        normalized["needs_multi_character_interaction"] = len(normalized.get("character_names") or []) > 1
        risks = list(normalized.get("risk_flags") or [])
        if "dialogue_story_expanded_frame_count" not in risks:
            risks.append("dialogue_story_expanded_frame_count")
        normalized["risk_flags"] = risks
        return normalized

    def parse(self, bundle: InputBundle) -> TaskSpec:
        """Call Qwen parser and return validated TaskSpec."""
        payload = {
            "user_text": bundle.user_text,
            "character_references": {k: str(v) for k, v in bundle.character_references.items()},
            "scene_references": {k: str(v) for k, v in bundle.scene_references.items()},
            "setting_docs": bundle.setting_docs,
            "history_frames": [str(p) for p in bundle.history_frames],
            "constraints": bundle.constraints,
        }
        data = self.qwen_client.parse_task(payload)
        normalized = self._normalize_qwen_output(data, bundle)
        normalized = self._semantic_storyboard_refine(normalized, bundle)
        normalized = self._dialogue_storyboard_frame_refine(normalized, bundle)
        return TaskSpec.model_validate(normalized)
