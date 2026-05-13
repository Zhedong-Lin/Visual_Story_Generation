"""JSON parsing helpers."""

import json
import re
from typing import Any


def _strip_code_fence(text: str) -> str:
    """Remove optional markdown code fences."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_json_span(text: str) -> str:
    """Extract most likely JSON object/array span from noisy text."""
    t = text.strip()
    # Prefer object span.
    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and r > l:
        return t[l : r + 1]
    # Fallback to array span.
    l = t.find("[")
    r = t.rfind("]")
    if l != -1 and r != -1 and r > l:
        return t[l : r + 1]
    return t


def _light_json_repair(text: str) -> str:
    """Apply lightweight repairs for common LLM JSON errors."""
    t = text.strip()
    # Remove trailing commas before closing braces/brackets.
    t = re.sub(r",\s*([}\]])", r"\1", t)
    # Remove a single dangling comma at end-of-text: ["a", "b"], -> ["a", "b"]
    t = re.sub(r",\s*$", "", t)
    # Normalize smart quotes.
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # Quote bare object keys: {foo: 1, bar_baz: 2} -> {"foo": 1, "bar_baz": 2}
    t = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:)', r'\1"\2"\3', t)
    t = _insert_missing_line_commas(t)
    return t


def _insert_missing_line_commas(text: str) -> str:
    """Add likely-missing commas between multiline JSON values.

    LLMs often emit arrays/objects with one item per line but forget commas.
    This is a conservative line-based repair that only adds commas when the
    previous line already looks like a complete JSON value and the next line
    begins another value or field.
    """

    lines = text.splitlines()
    if len(lines) < 2:
        return text

    def _prev_ends_value(line: str) -> bool:
        stripped = line.rstrip()
        return bool(
            stripped
            and not stripped.endswith((",", "[", "{", ":"))
            and (
                stripped.endswith(('"', "]", "}", "true", "false", "null"))
                or stripped[-1].isdigit()
            )
        )

    def _next_starts_value(line: str) -> bool:
        stripped = line.lstrip()
        return bool(
            stripped
            and not stripped.startswith(("]", "}", ","))
            and (
                stripped.startswith(('"', "{", "["))
                or re.match(r"^[A-Za-z_][A-Za-z0-9_\-]*\s*:", stripped)
                or re.match(r"^-?\d", stripped)
                or stripped.startswith(("true", "false", "null"))
            )
        )

    repaired: list[str] = []
    for idx, line in enumerate(lines):
        repaired.append(line)
        if idx == len(lines) - 1:
            continue
        if _prev_ends_value(line) and _next_starts_value(lines[idx + 1]):
            repaired[-1] = line.rstrip() + ","
    return "\n".join(repaired)


def _yaml_fallback(text: str) -> Any:
    """Parse JSON-ish LLM output with YAML as a last resort."""
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - depends on optional import state
        raise json.JSONDecodeError(str(exc), text, 0) from exc

    data = yaml.safe_load(text)
    if isinstance(data, (dict, list)):
        return data
    raise json.JSONDecodeError("YAML fallback did not return object/array", text, 0)


def safe_json_loads(text: str) -> Any:
    """Attempt to parse JSON with tolerant fallback steps."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = _strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    span = _extract_json_span(cleaned)
    try:
        return json.loads(span)
    except json.JSONDecodeError:
        pass

    repaired = _light_json_repair(span)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    try:
        return _yaml_fallback(span)
    except Exception:
        pass

    try:
        return _yaml_fallback(repaired)
    except Exception as exc:
        raise json.JSONDecodeError(str(exc), repaired, 0) from exc
