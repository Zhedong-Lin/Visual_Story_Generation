"""Asset resolver for auto-matching names to local images."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from anime_pipeline_graph.constants import IMAGE_EXTENSIONS
from anime_pipeline_graph.domain.models import AssetResolutionReport, InputBundle, TaskSpec


class AssetResolver:
    """Resolve character and scene assets by parser-discovered names."""

    def __init__(self, characters_dir: Path, scenes_dir: Path) -> None:
        self.characters_dir = characters_dir
        self.scenes_dir = scenes_dir

    def _find_image(self, base_dir: Path, name: str) -> Path | None:
        # Fast exact-match path first.
        for ext in IMAGE_EXTENSIONS:
            p = base_dir / f"{name}{ext}"
            if p.exists():
                return p
        # Case-insensitive fallback for Linux filesystems.
        target = name.lower()
        for p in base_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if p.stem.lower() == target:
                return p
        return None

    def _resolve_names(self, names: Iterable[str], base_dir: Path) -> tuple[dict[str, Path], list[str]]:
        found: dict[str, Path] = {}
        missing: list[str] = []
        for name in names:
            hit = self._find_image(base_dir, name)
            if hit is None:
                missing.append(name)
            else:
                found[name] = hit
        return found, missing

    def resolve(self, bundle: InputBundle, spec: TaskSpec) -> tuple[InputBundle, AssetResolutionReport]:
        """Populate InputBundle references and return report."""
        found_chars, missing_chars = self._resolve_names(spec.character_names, self.characters_dir)
        found_scenes, missing_scenes = self._resolve_names(spec.scene_names, self.scenes_dir)

        merged_chars = {**bundle.character_references, **found_chars}
        merged_scenes = {**bundle.scene_references, **found_scenes}
        new_bundle = bundle.model_copy(
            update={
                "character_references": merged_chars,
                "scene_references": merged_scenes,
            }
        )
        warnings = [
            *[f"character ref missing: {name}" for name in missing_chars],
            *[f"scene ref missing: {name}" for name in missing_scenes],
        ]
        report = AssetResolutionReport(
            found_characters=found_chars,
            missing_characters=missing_chars,
            found_scenes=found_scenes,
            missing_scenes=missing_scenes,
            warnings=warnings,
        )
        return new_bundle, report
