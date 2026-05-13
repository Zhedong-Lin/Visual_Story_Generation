"""Prompt pack builder skill."""

from __future__ import annotations
import re


def _compress_text(text: str, max_words: int = 24) -> str:
    """Compress text by words while preserving order."""
    words = text.replace("\n", " ").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _add_prompt_field(parts: list[str], label: str, text: str, max_words: int) -> None:
    """Append a complete prompt field without cutting the final prompt mid-field."""
    compact = _compress_text(str(text or "").strip(), max_words)
    if compact:
        parts.append(f"{label}: {compact}.")


def _token_set(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-zA-Z][a-zA-Z']{2,}", str(text or "").lower())
        if tok
        not in {
            "the",
            "and",
            "with",
            "from",
            "into",
            "onto",
            "beside",
            "while",
            "frame",
            "shot",
            "camera",
            "expression",
            "pose",
        }
    }


def _pick_planner_item(items: list, frame_idx: int, beat: str, used: set[int]) -> dict:
    """Pick a planner item by semantic overlap, tolerating extra/missing LLM rows."""
    if not items:
        return {}
    if len(items) == 1:
        return items[0] if isinstance(items[0], dict) else {}
    beat_tokens = _token_set(beat)
    candidates: list[tuple[float, int, dict]] = []
    for idx, item in enumerate(items):
        if idx in used or not isinstance(item, dict):
            continue
        item_text = " ".join(str(v) for v in item.values() if isinstance(v, (str, int, float)))
        item_tokens = _token_set(item_text)
        overlap = len(beat_tokens & item_tokens)
        distance = abs(idx - frame_idx)
        score = overlap - distance * 0.35
        candidates.append((score, idx, item))
    if not candidates:
        item = items[min(frame_idx, len(items) - 1)]
        return item if isinstance(item, dict) else {}
    candidates.sort(key=lambda row: (-row[0], abs(row[1] - frame_idx), row[1]))
    _, idx, item = candidates[0]
    used.add(idx)
    return item


def _role_hint(name: str) -> str:
    """Simple role hints to keep character depiction stable."""
    key = name.lower()
    if key == "lulu":
        return "young woman"
    if key == "jiddo":
        return "elderly man"
    return "character"


def _detect_key_character(text: str, character_names: list[str], pose_subject: str | None = None) -> str | None:
    """Pick the most relevant character for one frame."""
    if pose_subject:
        for name in character_names:
            if name.lower() == pose_subject.lower():
                return name
    lowered = text.lower()
    hits = []
    for name in character_names:
        match = _find_character_name_match(lowered, name)
        if match:
            hits.append((match.start(), name))
    if hits:
        hits.sort(key=lambda x: x[0])
        return hits[0][1]
    return None


def _detect_present_characters(
    text: str,
    character_names: list[str],
    pose_subject: str | None = None,
    expression_subject: str | None = None,
) -> list[str]:
    """Detect which named characters are explicitly present in this frame."""
    del pose_subject, expression_subject
    lowered = (text or "").lower()
    present: list[str] = []
    for name in character_names:
        if _find_character_name_match(lowered, name):
            present.append(name)
    return present


def _find_character_name_match(text: str, name: str) -> re.Match[str] | None:
    for pattern in _character_name_patterns(name):
        match = pattern.search(text)
        if match:
            return match
    return None


def _character_name_patterns(name: str) -> list[re.Pattern[str]]:
    """Build complete-word patterns plus safe aliases for descriptive names."""

    raw = str(name).strip().lower()
    if not raw:
        return []
    patterns = [_character_name_pattern(raw)]
    parts = raw.split()
    if len(parts) > 1 and parts[0] in {"little", "young", "old", "elderly", "small", "big"}:
        patterns.append(_character_name_pattern(" ".join(parts[1:])))
    return patterns


def _character_name_pattern(name: str) -> re.Pattern[str]:
    """Match character names as complete words/phrases, not substrings."""

    escaped = re.escape(str(name).strip().lower())
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])", flags=re.IGNORECASE)


def _build_prompt_anchor(
    frame_idx: int,
    num_frames: int,
    scene_text: str,
    key_char: str | None,
    supporting: list[str],
    action_text: str,
    pose_text: str,
    require_both_characters: bool,
) -> str:
    """Build short anchor prompt with action/object terms prioritized up front."""
    main_label = key_char if key_char else "none"
    main_role = _role_hint(key_char or "character")
    support_text = ", ".join(supporting) if supporting else "none"
    anchor = (
        f"anime storyboard frame {frame_idx}/{num_frames}. "
        f"frame goal {action_text}. "
        f"pose {pose_text}. "
        f"scene {scene_text}. "
        f"main {main_label} {main_role}; support {support_text}. "
        f"{'must show both characters. ' if require_both_characters else ''}"
        f"allow visible action change from previous frame."
    )
    # Keep this short to avoid CLIP truncation while preserving action words.
    return _compress_text(anchor, 52)


def _extract_outfit_overrides_from_text(text: str, character_names: list[str]) -> dict[str, str]:
    """Extract explicit outfit instructions from one text chunk."""
    text = text or ""
    lowered = text.lower()
    out: dict[str, str] = {}
    for name in character_names:
        lname = name.lower()
        # English patterns.
        en_patterns = [
            rf"{re.escape(lname)}\s+(?:is\s+)?wearing\s+([^.,;\n]+)",
            rf"{re.escape(lname)}\s+dressed\s+in\s+([^.,;\n]+)",
            rf"{re.escape(lname)}\s+wears\s+([^.,;\n]+)",
            rf"{re.escape(lname)}.*?(?:change|switch)\s+(?:her|his|their)?\s*outfit\s+to\s+([^.,;\n]+)",
        ]
        hit = None
        for pat in en_patterns:
            m = re.search(pat, lowered, flags=re.IGNORECASE)
            if m:
                hit = m.group(1).strip()
                break
        # Chinese patterns.
        if not hit:
            cn_patterns = [
                rf"{re.escape(lname)}.*?换上([^，。；\n]+)",
                rf"{re.escape(lname)}.*?穿上([^，。；\n]+)",
                rf"{re.escape(lname)}.*?穿着([^，。；\n]+)",
                rf"{re.escape(lname)}.*?身穿([^，。；\n]+)",
            ]
            for pat in cn_patterns:
                m = re.search(pat, text, flags=re.IGNORECASE)
                if m:
                    hit = m.group(1).strip()
                    break
        # Global fallback for single-character prompts.
        if not hit and len(character_names) == 1:
            m = re.search(r"(?:wearing|dressed in|wears)\s+([^.,;\n]+)", lowered, flags=re.IGNORECASE)
            if m:
                hit = m.group(1).strip()
        if hit:
            out[name] = hit
    return out


def _extract_outfit_overrides(user_text: str, character_names: list[str]) -> dict[str, str]:
    """Backward-compatible wrapper for full prompt extraction."""
    return _extract_outfit_overrides_from_text(user_text, character_names)


def _extract_key_props(text: str, character_names: list[str], scene_names: list[str]) -> list[str]:
    """Extract likely object props from narrative text (non-character, non-scene)."""
    lowered = (text or "").lower()
    props: list[str] = []
    stop = {
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
        "of",
        "his",
        "her",
        "its",
        "their",
    }
    verbs = {
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
    modifiers = {
        "gentle",
        "slight",
        "slightly",
        "subtle",
        "soft",
        "warm",
        "calm",
        "contemplative",
        "dramatic",
        "intense",
        "wide",
        "faint",
        "small",
        "sudden",
        "ripe",
        "red",
        "slow",
        "slowly",
        "mid",
        "single",
        "slight",
        "slight",
    }
    pronouns = {"it", "he", "she", "they", "them", "his", "her", "their"}
    banned = set([x.lower() for x in character_names]) | set([x.lower() for x in scene_names]) | stop | verbs | modifiers | pronouns

    def _is_valid_prop_token(tok: str) -> bool:
        tok = tok.strip().lower()
        if not tok or len(tok) <= 1:
            return False
        if tok in banned:
            return False
        if tok.endswith("ly"):  # likely adverb
            return False
        return True

    def _push_phrase_head(phrase: str) -> None:
        # Prefer phrase head noun (last token), fallback to first valid token.
        tokens = [t.strip().lower() for t in phrase.split() if t.strip()]
        if not tokens:
            return
        head = tokens[-1]
        if _is_valid_prop_token(head):
            props.append(head)
            return
        for t in tokens:
            if _is_valid_prop_token(t):
                props.append(t)
                return

    patterns = [
        r"\bwith\s+(?:a|an|the)\s+([a-z_][a-z0-9_]*(?:\s+[a-z_][a-z0-9_]*){0,2})\b",
        r"\bholding\s+(?:a|an|the)\s+([a-z_][a-z0-9_]*(?:\s+[a-z_][a-z0-9_]*){0,2})\b",
        r"\bdrops?\s+(?:his|her|its|their|the|a|an)?\s*([a-z_][a-z0-9_]*)\b",
        r"\bloses?\s+(?:his|her|its|their|the|a|an)?\s*([a-z_][a-z0-9_]*)\b",
        # Motion/event patterns to catch key objects in action beats.
        r"\b([a-z_][a-z0-9_]*)\s+(?:begins?|starts?)\s+to\s+fall\b",
        r"\b([a-z_][a-z0-9_]*)\s+falls?\b",
        r"\b([a-z_][a-z0-9_]*)\s+hits?\b",
        r"\b([a-z_][a-z0-9_]*)\s+strikes?\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, lowered):
            tok = (m.group(1) or "").strip().lower()
            if not tok:
                continue
            if " " in tok:
                _push_phrase_head(tok)
            elif _is_valid_prop_token(tok):
                props.append(tok)

    # Keep stable order and cap size.
    dedup = list(dict.fromkeys(props))
    return dedup[:5]


def _extract_subjects(lowered_text: str, pattern: str) -> list[str]:
    """Extract deduplicated event subjects from a regex pattern with one capture group."""
    out: list[str] = []
    for m in re.finditer(pattern, lowered_text):
        tok = (m.group(1) or "").strip().lower()
        if tok:
            out.append(tok)
    return list(dict.fromkeys(out))


def _build_event_constraints(beat_text: str) -> list[str]:
    """Build hard event constraints to prevent action/object regressions across frames."""
    lowered = (beat_text or "").lower()
    constraints: list[str] = []
    pronouns = {"it", "he", "she", "they", "them", "his", "her", "their"}

    fall_subjects = _extract_subjects(lowered, r"\b([a-z_][a-z0-9_]*)\s+(?:begins?|starts?)\s+to\s+fall\b")
    fall_subjects += _extract_subjects(lowered, r"\b([a-z_][a-z0-9_]*)\s+falls?\b")
    hit_subjects = _extract_subjects(lowered, r"\b([a-z_][a-z0-9_]*)\s+hits?\b")
    hit_subjects += _extract_subjects(lowered, r"\b([a-z_][a-z0-9_]*)\s+strikes?\b")
    fall_subjects = [s for s in fall_subjects if s not in pronouns]
    clean_hit_subjects = []
    for s in hit_subjects:
        if s in pronouns:
            if fall_subjects:
                clean_hit_subjects.append(fall_subjects[0])
            continue
        clean_hit_subjects.append(s)
    hit_subjects = list(dict.fromkeys(clean_hit_subjects))

    # Only add constraints when event has a concrete object subject.
    if fall_subjects:
        subj = fall_subjects[0]
        constraints.append(f"{subj} must be visibly off the branch and in mid-air")
        constraints.append(f"do not keep the {subj} hanging on the tree in this frame")
    if hit_subjects:
        subj = hit_subjects[0]
        constraints.append(f"{subj} must visibly contact the character head or be just-after impact")
        constraints.append(f"do not keep the impact {subj} hanging on the branch")

    return list(dict.fromkeys(constraints))


def _infer_scene_tokens_from_frames(frame_texts: list[str], character_names: list[str]) -> list[str]:
    """Infer lightweight scene tokens from frame descriptions when parser scene_names is empty."""
    vocab = {
        "bridge",
        "river",
        "water",
        "street",
        "road",
        "forest",
        "city",
        "room",
        "house",
        "beach",
        "park",
        "school",
    }
    char_set = {c.lower() for c in character_names}
    counts: dict[str, int] = {}
    for text in frame_texts or []:
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower()):
            if tok in char_set:
                continue
            if tok not in vocab:
                continue
            counts[tok] = counts.get(tok, 0) + 1
    if not counts:
        return []
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:3]]


def _build_character_bible(
    character_names: list[str],
    profiles: dict,
    outfit_overrides: dict[str, str],
) -> str:
    """Build compact character appearance/outfit text for the current frame."""
    chunks = []
    appearance_words = 16 if len(character_names) >= 4 else 24 if len(character_names) == 3 else 32
    outfit_words = 10 if len(character_names) >= 4 else 14 if len(character_names) == 3 else 18
    for name in character_names:
        data = {}
        if isinstance(profiles, dict):
            data = profiles.get(name, {}) or profiles.get(str(name).lower(), {}) or {}
        if not isinstance(data, dict):
            data = {}
        appearance = str(data.get("appearance", "")).strip()
        override = str(outfit_overrides.get(name, "")).strip()
        outfit = override if override else str(data.get("outfit", "")).strip()
        fields = []
        if appearance:
            fields.append(_compress_text(appearance, appearance_words))
        if outfit:
            fields.append(f"outfit: {_compress_text(outfit, outfit_words)}")
        text = f"{name}: {'; '.join(fields)}" if fields else f"{name}:"
        chunks.append(text)
    return " | ".join(chunks)


def execute(step, store, ctx):
    """Merge spec + conditions into final prompt pack."""
    spec = store.get("task_spec")
    bundle = store.get("input_bundle")
    scene_pack = store.get("scene_pack", {})
    char_anchor = store.get("char_anchor", {})
    pose_plan = store.get("pose_plan_json", {})
    shot_plan = store.get("shot_plan_json", {})
    transition_plan = store.get("transition_plan_json", {})
    expression_plan = store.get("expression_plan_json", {})
    continuity_state = store.get("continuity_state", {})
    visual_style_plan = store.get("visual_style_plan", {})
    frame_specs = store.get("frame_specs", [])
    bundle_profiles = bundle.constraints.get("character_profiles", {}) if bundle and bundle.constraints else {}
    anchor_profiles = char_anchor.get("character_profiles", {}) if isinstance(char_anchor, dict) else {}
    profiles = {}
    if isinstance(bundle_profiles, dict):
        profiles.update({str(k).lower(): v for k, v in bundle_profiles.items() if isinstance(v, dict)})
    # character_bind output has higher priority because it is task-specific per prompt text.
    if isinstance(anchor_profiles, dict):
        for k, v in anchor_profiles.items():
            if isinstance(v, dict):
                profiles[str(k).lower()] = dict(v)

    style_core = "anime storyboard frame, cinematic shot, coherent identity"
    char_hint = ", ".join(spec.character_names) if spec.character_names else "characters from prompt"
    scene_hint = ", ".join(spec.scene_names) if spec.scene_names else "scene from prompt"
    global_outfit_overrides = _extract_outfit_overrides(bundle.user_text if bundle else "", spec.character_names)

    frame_texts = []
    if frame_specs:
        for frame in frame_specs:
            if isinstance(frame, dict):
                frame_texts.append(frame.get("description") or frame.get("desc") or "")
            elif isinstance(frame, str):
                frame_texts.append(frame)
    if not frame_texts:
        frame_texts = list(spec.frame_descriptions) if spec.frame_descriptions else [bundle.user_text]
    if len(frame_texts) < spec.num_frames:
        frame_texts.extend([frame_texts[-1]] * (spec.num_frames - len(frame_texts)))
    frame_texts = frame_texts[: spec.num_frames]
    inferred_scenes = _infer_scene_tokens_from_frames(frame_texts, spec.character_names)
    resolved_scene_names = list(spec.scene_names) if spec.scene_names else inferred_scenes

    # Per-frame outfit state: only switch when a specific frame explicitly describes outfit change.
    outfit_state: dict[str, str] = {}
    for name in spec.character_names:
        data = profiles.get(str(name).lower(), {}) if isinstance(profiles, dict) else {}
        if isinstance(data, dict):
            base_outfit = str(data.get("outfit", "")).strip()
            if base_outfit:
                outfit_state[name] = base_outfit
    # For single-image tasks, keep the old global override behavior.
    if spec.num_frames <= 1:
        outfit_state.update({k: v for k, v in global_outfit_overrides.items() if v})

    frame_outfit_states: list[dict[str, str]] = []
    for i in range(spec.num_frames):
        frame_override = _extract_outfit_overrides_from_text(frame_texts[i], spec.character_names)
        if frame_override:
            outfit_state.update(frame_override)
        frame_outfit_states.append(dict(outfit_state))

    frame_prompts = []
    frame_prompt_2 = []
    frame_prompt_payloads = []
    frame_key_characters = []
    frame_present_characters = []
    continuity_blocks = []
    global_props = _extract_key_props(bundle.user_text if bundle else "", spec.character_names, spec.scene_names)
    pose_items = pose_plan.get("poses", []) if isinstance(pose_plan, dict) else []
    expression_items = expression_plan.get("expressions", []) if isinstance(expression_plan, dict) else []
    shot_items = shot_plan.get("shots", []) if isinstance(shot_plan, dict) else []
    transition_items = transition_plan.get("transitions", []) if isinstance(transition_plan, dict) else []
    style_bible = visual_style_plan.get("style_bible", {}) if isinstance(visual_style_plan, dict) else {}
    used_pose_indices: set[int] = set()
    used_expression_indices: set[int] = set()
    used_shot_indices: set[int] = set()
    used_transition_indices: set[int] = set()
    for i in range(spec.num_frames):
        beat = frame_texts[i] if i < len(frame_texts) else frame_texts[-1]
        pose_item = _pick_planner_item(pose_items, i, beat, used_pose_indices)
        expression_item = _pick_planner_item(expression_items, i, beat, used_expression_indices)
        shot_item = _pick_planner_item(shot_items, i, beat, used_shot_indices)
        transition_item = _pick_planner_item(transition_items, i, beat, used_transition_indices)
        frame_outfits = frame_outfit_states[i] if i < len(frame_outfit_states) else {}
        pose_subject = None
        if pose_item:
            pose_subject = pose_item.get("subject")
        expression_subject = None
        if expression_item:
            expression_subject = expression_item.get("subject")
        present_chars = _detect_present_characters(
            beat,
            spec.character_names,
            pose_subject=pose_subject,
            expression_subject=expression_subject,
        )
        key_char = _detect_key_character(beat, present_chars, pose_subject=pose_subject) if present_chars else None
        if key_char is None and pose_subject:
            for c in present_chars:
                if c.lower() == str(pose_subject).lower():
                    key_char = c
                    break
        if key_char is None and present_chars:
            key_char = present_chars[0]
        frame_key_characters.append(key_char)
        pose_line = ""
        if pose_item:
            pose_line = pose_item.get("pose", "")
        camera_line = ""
        notes_line = ""
        if pose_item:
            camera_line = pose_item.get("camera", "")
            notes_line = pose_item.get("notes", "")
        expression_line = ""
        expression_intensity = ""
        expression_notes = ""
        if expression_item:
            expression_line = expression_item.get("expression", "")
            expression_intensity = expression_item.get("intensity", "")
            expression_notes = expression_item.get("notes", "")
        supporting = [n for n in present_chars if n != key_char]
        frame_present_characters.append(present_chars)
        exact_actor_rule = (
            f"Show exactly {len(present_chars)} named visible actor"
            f"{'' if len(present_chars) == 1 else 's'} once: {', '.join(present_chars)}. "
            "Do not duplicate any named actor."
            if present_chars
            else "Show no named character in this frame."
        )
        scene_text = ", ".join(resolved_scene_names) if resolved_scene_names else "scene from prompt"
        action_text = _compress_text(beat, 24)
        pose_text = _compress_text(pose_line, 12) if pose_line else "dynamic motion"
        expression_text = _compress_text(expression_line, 10) if expression_line else "natural expression"
        frame_props = _extract_key_props(beat, spec.character_names, spec.scene_names)
        must_props = frame_props or global_props
        event_constraints = _build_event_constraints(beat)
        props_hint = f"key props: {', '.join(must_props)}" if must_props else "key props: none"
        event_hint_short = _compress_text(" ".join(event_constraints), 20) if event_constraints else ""
        frame_prompt = _build_prompt_anchor(
            frame_idx=i + 1,
            num_frames=spec.num_frames,
            scene_text=scene_text,
            key_char=key_char,
            supporting=supporting,
            action_text=f"{action_text}, {event_hint_short}, expression {expression_text}, {props_hint}",
            pose_text=pose_text,
            require_both_characters=len(spec.character_names) > 1 and len(present_chars) > 1,
        )
        frame_prompt = f"{frame_prompt} {exact_actor_rule}"
        prev_summary = frame_texts[i - 1] if i > 0 and i - 1 < len(frame_texts) else ""
        transition_type = str(transition_item.get("transition_type", "")).lower() if transition_item else ""
        is_scene_change = "scene_change" in transition_type
        continuity = {
            "previous_frame_summary": prev_summary,
            "must_keep_scene_layout": i > 0 and not is_scene_change,
            "allowed_delta": "new scene layout; keep only character identity/style from previous reference" if is_scene_change else "character pose and camera shift only",
        }
        continuity_blocks.append(continuity)
        if prev_summary and is_scene_change:
            continuity_hint = (
                "Continuity: use previous frame only for character identity and drawing style; "
                f"do not copy previous scene layout or extra characters; previous frame: {_compress_text(prev_summary, 28)}."
            )
        elif prev_summary:
            continuity_hint = f"Continuity: keep scene layout/light; previous frame: {_compress_text(prev_summary, 40)}."
        else:
            continuity_hint = "Continuity: establish scene layout and identities for next frames."
        shot_hint = ""
        if shot_item:
            ss = shot_item.get("shot_scale", "")
            ca = shot_item.get("camera_angle", "")
            fi = shot_item.get("framing_intent", "")
            shot_hint = _compress_text(f"Shot: {ss}; Camera: {ca}; Framing: {fi}.", 20)
        transition_hint = ""
        if transition_item:
            tt = transition_item.get("transition_type", "")
            ed = transition_item.get("expected_deltas", "")
            transition_hint = _compress_text(f"Transition: {tt}; Delta rule: {ed}.", 22)
        style_hint = ""
        if isinstance(style_bible, dict) and style_bible:
            style_hint = _compress_text(
                f"Style: {style_bible.get('art_style', '')}; palette {style_bible.get('color_palette', '')}; lighting {style_bible.get('lighting_tone', '')}.",
                24,
            )
        frame_character_bible = _build_character_bible(
            present_chars,
            profiles,
            frame_outfits,
        ) if present_chars else ""
        long_parts = [f"Frame {i + 1}/{spec.num_frames}."]
        _add_prompt_field(long_parts, "Character bible", frame_character_bible, 130)
        _add_prompt_field(long_parts, "Frame goal", beat, 55)
        _add_prompt_field(long_parts, "Required props", ", ".join(must_props) if must_props else "none", 18)
        _add_prompt_field(long_parts, "Event constraints", "; ".join(event_constraints) if event_constraints else "none", 32)
        _add_prompt_field(long_parts, "Pose", pose_line, 18)
        expression_full = f"{expression_line}; intensity: {expression_intensity}".strip(" ;")
        _add_prompt_field(long_parts, "Expression", expression_full, 15)
        _add_prompt_field(long_parts, "Camera", camera_line, 16)
        _add_prompt_field(long_parts, "Scene", ", ".join(resolved_scene_names) if resolved_scene_names else scene_hint, 12)
        _add_prompt_field(long_parts, "Characters in frame", ", ".join(present_chars) if present_chars else "none", 16)
        _add_prompt_field(long_parts, "Visible actor rule", exact_actor_rule, 28)
        _add_prompt_field(long_parts, "Key/supporting", f"{key_char or 'none'}; supporting {', '.join(supporting) if supporting else 'none'}", 16)
        if len(spec.character_names) > 1 and len(present_chars) > 1:
            long_parts.append("Both characters must be visible.")
        _add_prompt_field(long_parts, "Notes", notes_line, 16)
        _add_prompt_field(long_parts, "Shot", shot_hint, 16)
        _add_prompt_field(long_parts, "Transition", transition_hint, 18)
        _add_prompt_field(long_parts, "Style", style_hint, 18)
        _add_prompt_field(long_parts, "Continuity", continuity_hint.replace("Continuity:", "").strip(), 42)
        p2 = " ".join(part for part in long_parts if part.strip())
        frame_prompts.append(frame_prompt)
        frame_prompt_2.append(p2)
        frame_prompt_payloads.append({"prompt": frame_prompt, "prompt_2": p2})

    prompt_pack = {
        "prompt": _compress_text(
            f"{style_core}; characters {char_hint}; scene {scene_hint}; {_compress_text(bundle.user_text, 12)}",
            34,
        ),
        "prompt_2": _compress_text(
            f"{bundle.user_text}",
            150,
        ),
        "positive_prompt": _compress_text(
            f"{style_core}; characters {char_hint}; scene {scene_hint}",
            34,
        ),  # backward compatibility
        "frame_prompts": frame_prompts,
        "frame_prompt_2": frame_prompt_2,
        "frame_prompt_payloads": frame_prompt_payloads,
        "frame_key_characters": frame_key_characters,
        "frame_present_characters": frame_present_characters,
        "continuity_blocks": continuity_blocks,
        "scene_pack": scene_pack,
        "char_anchor": char_anchor,
        "character_bible": _build_character_bible(
            spec.character_names,
            profiles,
            frame_outfit_states[-1] if frame_outfit_states else outfit_state,
        ),
        "frame_outfit_states": frame_outfit_states,
        "pose_plan": pose_plan,
        "shot_plan": shot_plan,
        "transition_plan": transition_plan,
        "expression_plan": expression_plan,
        "continuity_state": continuity_state,
        "visual_style_plan": visual_style_plan,
        "num_frames": spec.num_frames,
        "num_inference_steps": 12,
        "max_sequence_length": 256,
    }
    store.set("prompt_pack", prompt_pack)
    return {"outputs": {"prompt_pack": prompt_pack}, "artifacts": []}
