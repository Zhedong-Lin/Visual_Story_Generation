"""FLUX Fill provider."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - fallback for dry-run environments
    class _TorchStub:
        bfloat16 = "bfloat16"
        float32 = "float32"
        OutOfMemoryError = RuntimeError

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def empty_cache() -> None:
                return None

    torch = _TorchStub()  # type: ignore[assignment]

from anime_pipeline_graph.utils.images import make_mock_image


class MissingMaskError(ValueError):
    """Raised when mask path is required but missing."""


class FluxFillProvider:
    """Local HF diffusers provider for FLUX Fill."""

    def __init__(self, model_name: str, hf_token: str = "") -> None:
        self.model_name = model_name
        self.hf_token = hf_token
        self.pipe = None

    def load_pipeline(self) -> None:
        """Load fill pipeline lazily."""
        if self.pipe is not None:
            return
        from diffusers import FluxFillPipeline

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.pipe = FluxFillPipeline.from_pretrained(self.model_name, token=self.hf_token, torch_dtype=dtype)
        if torch.cuda.is_available():
            try:
                self.pipe = self.pipe.to("cuda")
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if hasattr(self.pipe, "enable_model_cpu_offload"):
                    self.pipe.enable_model_cpu_offload()
                elif hasattr(self.pipe, "enable_sequential_cpu_offload"):
                    self.pipe.enable_sequential_cpu_offload()
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing("max")

    def fill_edit(self, image_path: Path, mask_path: Optional[Path], instruction: str) -> Image.Image:
        """Run masked local edit."""
        if mask_path is None:
            raise MissingMaskError("fill_edit requires mask_path; executor should fallback to edit")
        self.load_pipeline()
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        result = self.pipe(prompt=instruction, image=image, mask_image=mask)
        return result.images[0]


class MockFluxFillProvider:
    """Dry-run mock fill provider."""

    def load_pipeline(self) -> None:
        """No-op for mock."""

    def fill_edit(self, image_path: Path, mask_path: Optional[Path], instruction: str) -> Image.Image:
        """Create mock fill edit image."""
        if mask_path is None:
            raise MissingMaskError("fill_edit requires mask_path; executor should fallback to edit")
        temp = Path("/tmp/mock_fill.png")
        make_mock_image(temp, "fill_edit", instruction)
        return Image.open(temp).copy()
