"""Rewrite dialogue-heavy user input into visual-only story text."""

from __future__ import annotations

import re
from typing import Any

from anime_pipeline_graph.constants import MAX_STORYBOARD_FRAMES


MAX_AUTO_DIALOGUE_FRAMES = min(10, MAX_STORYBOARD_FRAMES)


SPEAKER_LINE_RE = re.compile(
    r"^\s*[-*]?\s*(?!Frame\b|Panel\b)(?!第\s*\d+\s*[帧格张])"
    r"[\w\u4e00-\u9fff][\w\s\u4e00-\u9fff]{0,32}(?:\s*\([^)]{1,80}\))?\s*[:：]\s*\S.{2,}",
    flags=re.IGNORECASE,
)

SPEAKER_CAPTURE_RE = re.compile(
    r"^\s*[-*]?\s*(?!Frame\b|Panel\b)(?!第\s*\d+\s*[帧格张])"
    r"(?P<speaker>[\w\u4e00-\u9fff][\w\s\u4e00-\u9fff]{0,32})(?:\s*\([^)]{1,80}\))?\s*[:：]\s*(?P<body>\S.{2,})",
    flags=re.IGNORECASE,
)

CAT_SOUND_RE = re.compile(r"\b(?:miaow|meow|miao|miaou)\b|喵", flags=re.IGNORECASE)

VAGUE_FRAME_ACTOR_RE = re.compile(
    r"\b(?:the\s+children|the\s+kids|the\s+family|everyone|they|them|their|each\s+child|child['’]s|kids|children)\b|孩子们|他们|她们|大家",
    flags=re.IGNORECASE,
)

ADULT_ROLE_NAMES = {
    "dad",
    "father",
    "mom",
    "mother",
    "mama",
    "mum",
    "grandpa",
    "grandma",
    "jiddo",
    "bibi",
    "narrator",
    "scene",
}


def contains_dialogue(text: str) -> bool:
    """Detect quoted dialogue or screenplay-style speaker lines."""

    if re.search(r"[\"“”‘’「」『』][^\"“”‘’「」『』]{2,160}[\"“”‘’「」『』]", text):
        return True
    if any(SPEAKER_LINE_RE.search(line) for line in text.splitlines()):
        return True
    dialogue_words = [" says ", " said ", " asks ", " asked ", " replies ", " shouted ", " whispers "]
    lowered = f" {text.lower()} "
    return any(word in lowered for word in dialogue_words)


def infer_target_frames(text: str) -> int | None:
    hits = [int(m.group(1)) for m in re.finditer(r"\b(?:frame|panel)\s*(\d+)\b", text, flags=re.IGNORECASE)]
    hits.extend(int(m.group(1)) for m in re.finditer(r"第\s*(\d+)\s*[帧格张]", text))
    if hits:
        return max(1, min(max(hits), MAX_STORYBOARD_FRAMES))

    m = re.search(r"\b(\d+)\s*(?:frames?|panels?)\b", text, flags=re.IGNORECASE)
    if m:
        return max(1, min(int(m.group(1)), MAX_STORYBOARD_FRAMES))

    m = re.search(r"(\d+)\s*(?:帧|格|张|个分镜)", text)
    if m:
        return max(1, min(int(m.group(1)), MAX_STORYBOARD_FRAMES))
    return None


def estimate_target_frames(text: str) -> int:
    explicit = infer_target_frames(text)
    if explicit:
        return explicit

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    speaker_count = sum(1 for line in lines if SPEAKER_LINE_RE.search(line))
    action_count = sum(
        1
        for line in lines
        if not SPEAKER_LINE_RE.search(line)
        and re.search(r"\b(scene|everyone|laughs?|sitting|standing|hangs up|turns|go(?:ing)?|boat|trip)\b", line, re.I)
    )
    estimated = max(3, min(MAX_AUTO_DIALOGUE_FRAMES, round((speaker_count + action_count) / 2)))
    if len(text) > 900:
        estimated = max(estimated, min(6, MAX_AUTO_DIALOGUE_FRAMES))
    return estimated


def infer_species_hints(text: str) -> dict[str, str]:
    """Infer simple species hints from dialogue/action cues in the raw prompt."""

    hints: dict[str, str] = {}
    for line in text.splitlines():
        m = SPEAKER_CAPTURE_RE.search(line)
        if not m:
            continue
        speaker = re.sub(r"\s+", " ", m.group("speaker").strip())
        body = m.group("body")
        if CAT_SOUND_RE.search(body) and speaker:
            hints[speaker.lower()] = "cat"
    return hints


def infer_speaker_names(text: str) -> list[str]:
    """Collect named speakers from screenplay-style lines."""

    names: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        m = SPEAKER_CAPTURE_RE.search(line)
        if not m:
            continue
        speaker = re.sub(r"\s+", " ", m.group("speaker").strip())
        if " and " in speaker.lower():
            parts = re.split(r"\s+and\s+|,|、|和", speaker, flags=re.IGNORECASE)
        else:
            parts = [speaker]
        for part in parts:
            name = part.strip()
            key = name.lower()
            if not name or key in seen or key in {"narrator", "scene"}:
                continue
            seen.add(key)
            names.append(name)
    return names


def infer_child_actor_names(text: str, species_hints: dict[str, str]) -> list[str]:
    """Infer child actor names for replacing vague 'children/kids' wording."""

    child_names: list[str] = []
    for name in infer_speaker_names(text):
        key = name.lower()
        if key in ADULT_ROLE_NAMES:
            continue
        if key in species_hints:
            continue
        child_names.append(name)
    return child_names


def apply_explicit_actor_names(frame_text: str, child_names: list[str]) -> str:
    """Replace common vague child group wording with concrete names."""

    if not child_names:
        return frame_text
    child_text = " and ".join(child_names)
    text = frame_text
    replacements = [
        (r"\beach\s+child['’]s\s+shoulder\b", f"{child_text}'s shoulders"),
        (r"\bthe\s+children\b", child_text),
        (r"\bthe\s+kids\b", child_text),
        (r"\bchildren\b", child_text),
        (r"\bkids\b", child_text),
        (r"孩子们", child_text),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def apply_species_hints(frame_text: str, species_hints: dict[str, str]) -> str:
    """Patch common LLM rewrite mistakes before downstream parsing."""

    if not species_hints:
        return frame_text

    text = frame_text
    for raw_name, species in species_hints.items():
        if species != "cat":
            continue
        name = re.escape(raw_name)

        # Fix common over-generalization in dialogue rewrites.
        if raw_name == "kishmish":
            text = re.sub(
                r"\bDad and three children,\s*Lulu,\s*Zane,\s*and\s*Kishmish(?:\s+the\s+cat)?\b",
                "Dad, Lulu, Zane, and Kishmish the cat",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bthree children,\s*Lulu,\s*Zane,\s*and\s*Kishmish(?:\s+the\s+cat)?\b",
                "two children, Lulu and Zane, and Kishmish the cat",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bLulu,\s*Zane,\s*and\s*Kishmish(?!\s+the\s+cat)\b",
                "Lulu, Zane, and Kishmish the cat",
                text,
                flags=re.IGNORECASE,
            )

        text = re.sub(
            rf"\b({name})(?!\s+the\s+cat)\b",
            lambda m: f"{m.group(1)} the cat",
            text,
            flags=re.IGNORECASE,
        )

    fixed_lines: list[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(name in lowered and species == "cat" for name, species in species_hints.items()):
            line = re.sub(r"\barms crossed tightly\b", "front paws tucked close", line, flags=re.IGNORECASE)
            line = re.sub(r"\btheir lips\b", "its small mouth", line, flags=re.IGNORECASE)
            line = re.sub(r"\bchild's\b", "cat's", line, flags=re.IGNORECASE)
            line = re.sub(r"\bchildish despair\b", "pet-like disappointment", line, flags=re.IGNORECASE)
            line = re.sub(r"\bmimic(?:s|king)? a cat[’']s disappointed meow\b", "makes a disappointed meow", line, flags=re.IGNORECASE)
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


def find_vague_actor_frames(frame_text: str) -> list[str]:
    """Return frame labels whose text still contains vague actor references."""

    warnings: list[str] = []
    for line in frame_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not re.match(r"(?i)^Frame\s+\d+\s*:", stripped):
            continue
        if VAGUE_FRAME_ACTOR_RE.search(stripped):
            label = stripped.split(":", 1)[0].strip()
            warnings.append(f"{label} contains vague actor wording")
    return warnings


def fallback_remove_dialogue(text: str) -> str:
    """Last-resort local cleanup if LLM preprocessing is unavailable."""

    cleaned = re.sub(r"[\"“”‘’「」『』][^\"“”‘’「」『』]{1,200}[\"“”‘’「」『』]", "", text)
    cleaned_lines = []
    for line in cleaned.splitlines():
        if SPEAKER_LINE_RE.search(line):
            line = re.sub(r"^\s*[-*]?\s*[^:：]{1,80}\s*[:：]\s*", "", line)
        if line.strip():
            cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\b(?!Frame\b|Panel\b)[A-Za-z_][A-Za-z0-9_]{0,23}\s*[:：]\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .。")
    if not cleaned:
        cleaned = "Characters act out the story visually with expressive poses and reactions, without speech text."
    target = estimate_target_frames(text)
    return "\n".join(f"Frame {i + 1}: {cleaned}" for i in range(target))


def normalize_frame_labels(text: str) -> str:
    """Remove accidental duplicated frame labels from rewrites."""

    text = re.sub(
        r"(?im)^\s*(Frame\s+\d+\s*:\s*)+(?=Frame\s+\d+\s*:)",
        "",
        text,
    )
    text = re.sub(
        r"(?im)^\s*(Frame\s+(\d+)\s*:\s*)Frame\s+\2\s*:\s*",
        r"Frame \2: ",
        text,
    )
    return text.strip()


def extract_frame_prompt_text(text: str) -> str:
    """Recover Frame N prompts from malformed JSON or noisy LLM text."""

    cleaned = text.replace("\\n", "\n")
    cleaned = cleaned.replace('\\"', '"')
    pattern = re.compile(
        r"(?ims)(Frame\s*(\d+)\s*:\s*.*?)(?=\n\s*\"?Frame\s*\d+\s*:|\n\s*\]|\n\s*\"?(?:selected_frame_count|removed_dialogue|notes)\"?\s*:|$)"
    )
    frames: list[tuple[int, str]] = []
    for match in pattern.finditer(cleaned):
        idx = int(match.group(2))
        frame = match.group(1).strip()
        frame = re.sub(r'^[",\s]+', "", frame)
        frame = re.sub(r'[",\s]+$', "", frame)
        frame = normalize_frame_labels(frame)
        if frame:
            frames.append((idx, frame))

    if not frames:
        return ""

    seen: set[int] = set()
    ordered: list[str] = []
    for idx, frame in sorted(frames, key=lambda item: item[0]):
        if idx in seen:
            continue
        seen.add(idx)
        ordered.append(frame)
    return "\n".join(ordered)


def preprocess_dialogue_prompt(prompt: str, qwen_client: Any, enabled: bool = True) -> dict[str, Any]:
    """Normalize raw user input before the parser sees it.

    The returned ``pipeline_prompt`` is the only prompt downstream parser,
    planner, and generator modules should consume.
    """

    needs_preprocess = enabled and contains_dialogue(prompt)
    if not needs_preprocess:
        return {
            "original_prompt": prompt,
            "pipeline_prompt": prompt,
            "preprocessed": False,
            "reason": "no_dialogue_detected" if enabled else "disabled",
        }

    payload = {
        "user_text": prompt,
        "instruction": (
            "Remove dialogue and convert to plain-text Frame N visual prompts. "
            "Do not return JSON. Use 3 to 10 frames unless the user explicitly requested a count. "
            "Every frame must explicitly name the visible actors; do not use vague words like children, kids, they, them, or everyone."
        ),
    }
    species_hints = infer_species_hints(prompt)
    child_actor_names = infer_child_actor_names(prompt, species_hints)
    try:
        data = qwen_client.dialogue_to_storyboard_prompt(payload)
        if isinstance(data, dict):
            rewritten = str(data.get("story") or data.get("narrative") or data.get("rewritten_prompt") or "").strip()
        else:
            rewritten = str(data or "").strip()
        rewritten = re.sub(r"^```(?:text)?\s*|\s*```$", "", rewritten, flags=re.IGNORECASE).strip()
        rewritten = normalize_frame_labels(rewritten)
        rewritten = apply_species_hints(rewritten, species_hints)
        rewritten = apply_explicit_actor_names(rewritten, child_actor_names)
        if not rewritten:
            raise ValueError("LLM returned empty rewritten frame text")
        if not re.search(r"(?im)^\s*Frame\s+\d+\s*:", rewritten):
            rewritten = f"Frame 1: {rewritten}"
        vague_actor_warnings = find_vague_actor_frames(rewritten)
        return {
            "original_prompt": prompt,
            "pipeline_prompt": rewritten,
            "preprocessed": True,
            "method": qwen_client.__class__.__name__,
            "llm_output": data,
            "species_hints": species_hints,
            "child_actor_names": child_actor_names,
            "warnings": vague_actor_warnings,
        }
    except Exception as exc:
        raw_text = str(getattr(exc, "raw_text", "") or "")
        recovered = extract_frame_prompt_text(raw_text)
        rewritten = recovered or (normalize_frame_labels(raw_text.strip()) if raw_text.strip() else fallback_remove_dialogue(prompt))
        if rewritten and not re.search(r"(?im)^\s*Frame\s+\d+\s*:", rewritten):
            rewritten = f"Frame 1: {rewritten}"
        rewritten = apply_species_hints(rewritten, species_hints)
        rewritten = apply_explicit_actor_names(rewritten, child_actor_names)
        vague_actor_warnings = find_vague_actor_frames(rewritten)
        return {
            "original_prompt": prompt,
            "pipeline_prompt": rewritten,
            "preprocessed": True,
            "method": "raw_frame_recovery" if recovered else ("raw_text_recovery" if raw_text.strip() else "fallback"),
            "error": str(exc),
            "raw_llm_text": raw_text[:8000] if raw_text else "",
            "species_hints": species_hints,
            "child_actor_names": child_actor_names,
            "warnings": vague_actor_warnings,
        }
