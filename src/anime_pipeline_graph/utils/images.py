"""Image helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw


def make_mock_image(path: Path, title: str, subtitle: str, size: Tuple[int, int] = (832, 1216)) -> Path:
    """Create a visible mock image for dry-run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(28, 35, 48))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, size[0] - 20, size[1] - 20), outline=(110, 195, 255), width=4)
    draw.text((40, 60), title, fill=(240, 240, 240))
    draw.text((40, 110), subtitle[:160], fill=(210, 210, 210))
    img.save(path)
    return path


def normalize_to_png(src: Path, out: Path) -> Path:
    """Convert image to png."""
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    img.save(out, format="PNG")
    return out
