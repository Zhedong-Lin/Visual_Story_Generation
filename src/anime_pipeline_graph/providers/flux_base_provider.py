"""FLUX base generation provider."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

        class Generator:
            def __init__(self, device: str = "cpu") -> None:
                self.device = device

            def manual_seed(self, seed: int) -> "_TorchStub.Generator":
                return self

    torch = _TorchStub()  # type: ignore[assignment]

from anime_pipeline_graph.utils.images import make_mock_image


class FluxBaseProvider:
    """Local HF diffusers provider for FLUX.1-dev."""

    def __init__(self, model_name: str, hf_token: str = "") -> None:
        self.model_name = model_name
        self.hf_token = hf_token
        self.pipe = None
        self._active_lora_path: Optional[str] = None
        self._active_lora_paths: tuple[str, ...] = ()

    def load_pipeline(self) -> None:
        """Load diffusers pipeline lazily."""
        if self.pipe is not None:
            return
        from diffusers import FluxPipeline

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        force_cpu_offload = os.getenv("FORCE_CPU_OFFLOAD", "1").strip() == "1"
        self.pipe = FluxPipeline.from_pretrained(self.model_name, token=self.hf_token, torch_dtype=dtype)
        if torch.cuda.is_available():
            if force_cpu_offload and hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
            else:
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

    def apply_optional_lora(self, lora_path: Optional[str] = None) -> None:
        """Apply optional LoRA weights."""
        if not lora_path:
            return
        self.apply_optional_loras([lora_path])

    def apply_optional_loras(self, lora_paths: Optional[list[str]] = None) -> None:
        """Apply one or more LoRA adapters for the current frame."""
        if not lora_paths:
            return
        clean_paths = tuple(dict.fromkeys([p for p in lora_paths if p]))
        if not clean_paths:
            return
        self.load_pipeline()
        if self._active_lora_paths == clean_paths:
            return

        # Reset previously loaded adapters when supported.
        if self._active_lora_paths and hasattr(self.pipe, "unload_lora_weights"):
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass
        # Try multi-adapter loading when supported.
        loaded_names = []
        try:
            for idx, p in enumerate(clean_paths):
                adapter_name = f"char_{idx}"
                self.pipe.load_lora_weights(p, adapter_name=adapter_name)
                loaded_names.append(adapter_name)
            if loaded_names and hasattr(self.pipe, "set_adapters"):
                self.pipe.set_adapters(loaded_names, adapter_weights=[1.0] * len(loaded_names))
            self._active_lora_paths = clean_paths
            self._active_lora_path = clean_paths[0]
            return
        except TypeError:
            # Adapter name may be unsupported in some versions.
            pass
        except Exception:
            pass

        # Fallback: load only first LoRA for compatibility.
        self.pipe.load_lora_weights(clean_paths[0])
        self._active_lora_paths = (clean_paths[0],)
        self._active_lora_path = clean_paths[0]

    def generate(
        self,
        prompt_pack: Dict[str, Any],
        refs: Optional[Dict[str, Path]] = None,
        seed: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """Generate one image from prompt pack."""
        self.load_pipeline()
        width, height = size or (832, 1216)
        prompt = prompt_pack.get("prompt", prompt_pack.get("positive_prompt", "anime illustration"))
        prompt_2 = prompt_pack.get("prompt_2", "")
        num_inference_steps = int(prompt_pack.get("num_inference_steps", 20))
        max_sequence_length = int(prompt_pack.get("max_sequence_length", 256))
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        call_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }
        if prompt_2:
            call_kwargs["prompt_2"] = prompt_2
        # FLUX supports up to 512 here; we default to 256 for stability.
        call_kwargs["max_sequence_length"] = max_sequence_length
        try:
            result = self.pipe(**call_kwargs)
            return result.images[0]
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
            # last-resort retry with lower settings
            retry_kwargs = dict(call_kwargs)
            retry_kwargs["width"] = min(width, 512)
            retry_kwargs["height"] = min(height, 768)
            retry_kwargs["num_inference_steps"] = min(num_inference_steps, 4)
            result = self.pipe(**retry_kwargs)
            return result.images[0]


class MockFluxBaseProvider:
    """Dry-run mock base provider."""

    def load_pipeline(self) -> None:
        """No-op for mock provider."""

    def apply_optional_lora(self, lora_path: Optional[str] = None) -> None:
        """No-op for mock provider."""

    def apply_optional_loras(self, lora_paths: Optional[list[str]] = None) -> None:
        """No-op for mock provider."""

    def generate(
        self,
        prompt_pack: Dict[str, Any],
        refs: Optional[Dict[str, Path]] = None,
        seed: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """Generate mock image with text tags."""
        temp = Path("/tmp/mock_base.png")
        make_mock_image(temp, "base_generation", prompt_pack.get("positive_prompt", ""), size or (832, 1216))
        return Image.open(temp).copy()
