"""State store for graph execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from anime_pipeline_graph.utils.io import dump_json, ensure_dir


class StateStore:
    """Persist runtime state and artifacts."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = ensure_dir(run_dir)
        self.state: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Set value in in-memory state."""
        self.state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        return self.state.get(key, default)

    def save_step_json(self, step_id: str, phase: str, payload: Any) -> None:
        """Save pre/post step payload json."""
        dump_json(self.run_dir / "steps" / step_id / f"{phase}.json", payload)

    def save_named_json(self, name: str, payload: Any) -> None:
        """Save root-level json payload."""
        dump_json(self.run_dir / f"{name}.json", payload)

    def images_dir(self) -> Path:
        """Return run images directory."""
        return ensure_dir(self.run_dir / "images")
