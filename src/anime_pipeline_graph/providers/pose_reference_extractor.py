"""OpenPose reference extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

from anime_pipeline_graph.utils.images import make_mock_image


class PoseReferenceExtractor:
    """Extract skeleton map from pose reference image."""

    def extract(self, image_path: Optional[Path], output_path: Path) -> Optional[Path]:
        """Run OpenPose and save skeleton image. Return None when no reference."""
        if image_path is None:
            return None
        from controlnet_aux import OpenposeDetector

        output_path.parent.mkdir(parents=True, exist_ok=True)
        detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        img = Image.open(image_path).convert("RGB")
        skeleton = detector(img)
        skeleton.save(output_path)
        return output_path


class MockPoseReferenceExtractor:
    """Mock pose extractor for dry-run."""

    def extract(self, image_path: Optional[Path], output_path: Path) -> Optional[Path]:
        """Return placeholder skeleton when input exists."""
        if image_path is None:
            return None
        return make_mock_image(output_path, "pose_extract", f"from {image_path.name}")
