"""Character bind skill."""

from __future__ import annotations

import re


def _compress_text(text: str, max_words: int = 28) -> str:
    words = str(text or "").replace("\n", " ").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _extract_named_character_chunks(user_text: str, character_names: list[str]) -> dict[str, str]:
    """Extract per-character description chunks from free-form prompt text."""
    text = user_text or ""
    if not text or not character_names:
        return {}
    name_map = {name.lower(): name for name in character_names}

    # Match patterns like "zhuge liang: ...", "诸葛亮：..."
    pattern = re.compile(r"([A-Za-z][A-Za-z0-9_ \-]{1,40}|[\u4e00-\u9fff]{1,12})\s*[:：]\s*")
    hits = list(pattern.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(hits):
        raw_name = (m.group(1) or "").strip().lower()
        if raw_name not in name_map:
            continue
        start = m.end()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        chunk = text[start:end].strip(" \n\t,.;；。")
        # Cut before frame instructions to keep role profile clean.
        chunk = re.split(r"\bframe\s*\d+\b|分镜|镜头", chunk, maxsplit=1, flags=re.IGNORECASE)[0].strip(" \n\t,.;；。")
        if chunk:
            out[name_map[raw_name]] = chunk
    return out


def _extract_outfit_from_chunk(chunk: str) -> str:
    lowered = (chunk or "").lower()
    patterns = [
        r"(?:wearing|dressed in|wears)\s+([^.,;\n]+)",
        r"(?:in)\s+([^.,;\n]*(?:robe|armor|armour|uniform|dress|kimono|hanfu|clothes|outfit)[^.,;\n]*)",
        r"(?:身穿|穿着|穿上|着)\s*([^，。；\n]+)",
    ]
    for pat in patterns:
        m = re.search(pat, lowered, flags=re.IGNORECASE)
        if m:
            return _compress_text(m.group(1).strip(" ,.;"))
    # Fallback: lightweight keyword extraction in original chunk.
    kw = re.search(r"([^.,;\n]*(?:robe|armor|armour|uniform|dress|hanfu|outfit)[^.,;\n]*)", chunk, flags=re.IGNORECASE)
    if kw:
        return _compress_text(kw.group(1).strip(" ,.;"))
    return ""


def _build_text_character_profiles(bundle, spec) -> dict:
    """Build structured character profiles from text when no refs are available."""
    base = {}
    if bundle and getattr(bundle, "constraints", None):
        existing = bundle.constraints.get("character_profiles", {})
        if isinstance(existing, dict):
            for k, v in existing.items():
                if isinstance(v, dict):
                    base[str(k).lower()] = dict(v)

    by_name_desc = _extract_named_character_chunks(getattr(bundle, "user_text", ""), list(spec.character_names))
    for name in spec.character_names:
        key = str(name).lower()
        cur = base.get(key, {}) if isinstance(base.get(key), dict) else {}
        chunk = by_name_desc.get(name, "")
        if chunk:
            appearance = _compress_text(chunk, 36)
            outfit = _extract_outfit_from_chunk(chunk)
            cur["appearance"] = cur.get("appearance") or appearance
            if outfit:
                cur["outfit"] = cur.get("outfit") or outfit
        cur["must_keep"] = cur.get("must_keep") or "keep same face/hair and signature outfit across frames"
        base[key] = cur
    return base


def execute(step, store, ctx):
    """Bind character identity with refs and optional LoRA."""
    bundle = store.get("input_bundle")
    spec = store.get("task_spec")
    lora_map = ctx.lora_map
    lora_hits = {name: lora_map[name] for name in spec.character_names if name in lora_map}
    first_lora = None
    for name in spec.character_names:
        if name in lora_map:
            first_lora = lora_map[name]
            break
    brief = ", ".join(spec.character_constraints) if spec.character_constraints else "anime characters"
    text_profiles = _build_text_character_profiles(bundle, spec)
    has_refs = bool(getattr(bundle, "character_references", {}))
    # Keep bind lightweight: this step prepares metadata only and avoids loading heavy models.
    anchor = {
        "brief": brief,
        "refs": {k: str(v) for k, v in bundle.character_references.items()},
        "lora_path": first_lora,
        "lora_map": lora_hits,
        "character_profiles": text_profiles,
        "anchor_type": "char_anchor",
        "bind_mode": "ref_anchor" if has_refs else "textual_anchor",
    }
    store.set("char_anchor", anchor)
    return {"outputs": {"char_anchor": anchor}, "artifacts": []}
