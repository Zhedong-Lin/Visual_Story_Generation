"""Prompt loading helpers."""

from pathlib import Path


def load_prompt(path: Path) -> str:
    """Read a prompt template file."""
    return path.read_text(encoding="utf-8")
