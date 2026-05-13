"""FLUX Kontext provider for bind and edit."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class FluxKontextProvider:
    """Local HF diffusers provider for FLUX Kontext."""

    def __init__(self, model_name: str, hf_token: str = "") -> None:
        self.model_name = model_name
        self.hf_token = hf_token
        self.pipe = None
        self.pipe_family = "unknown"
        self._active_lora_path: Optional[str] = None
        self._active_lora_paths: tuple[str, ...] = ()

    def load_pipeline(self) -> None:
        """Load Kontext pipeline lazily."""
        if self.pipe is not None:
            return
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        force_cpu_offload = os.getenv("FORCE_CPU_OFFLOAD", "1").strip() == "1"

        # Prefer image-conditioned pipeline families for Kontext.
        pipe = None
        family = "unknown"
        try:
            from diffusers import FluxKontextPipeline  # type: ignore

            pipe = FluxKontextPipeline.from_pretrained(self.model_name, token=self.hf_token, torch_dtype=dtype)
            family = "flux_kontext"
        except Exception:
            try:
                from diffusers import AutoPipelineForImage2Image

                pipe = AutoPipelineForImage2Image.from_pretrained(
                    self.model_name, token=self.hf_token, torch_dtype=dtype
                )
                family = "img2img_auto"
            except Exception:
                from diffusers import FluxPipeline

                pipe = FluxPipeline.from_pretrained(self.model_name, token=self.hf_token, torch_dtype=dtype)
                family = "flux_text2img_fallback"

        self.pipe = pipe
        self.pipe_family = family
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
        """Apply LoRA if available."""
        if not lora_path:
            return
        self.apply_optional_loras([lora_path])

    def apply_optional_loras(self, lora_paths: Optional[list[str]] = None) -> None:
        """Apply one or more LoRA adapters."""
        if not lora_paths:
            return
        clean_paths = tuple(dict.fromkeys([p for p in lora_paths if p]))
        if not clean_paths:
            return
        self.load_pipeline()
        if self._active_lora_paths == clean_paths:
            return
        if self._active_lora_paths and hasattr(self.pipe, "unload_lora_weights"):
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass
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
            pass
        except Exception:
            pass
        self.pipe.load_lora_weights(clean_paths[0])
        self._active_lora_paths = (clean_paths[0],)
        self._active_lora_path = clean_paths[0]

    def unload_lora(self) -> None:
        """Unload current LoRA adapter when supported."""
        if self.pipe is None:
            self._active_lora_path = None
            self._active_lora_paths = ()
            return
        if hasattr(self.pipe, "unload_lora_weights"):
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass
        self._active_lora_path = None
        self._active_lora_paths = ()

    @staticmethod
    def _resize_for_kontext(image: Image.Image, max_side: int = 768) -> Image.Image:
        """Resize reference image to a safer inference size and keep aspect ratio."""
        w, h = image.size
        scale = min(max_side / max(w, 1), max_side / max(h, 1), 1.0)
        nw = int((w * scale) // 16 * 16)
        nh = int((h * scale) // 16 * 16)
        nw = max(nw, 384)
        nh = max(nh, 384)
        if (nw, nh) != (w, h):
            return image.resize((nw, nh), Image.LANCZOS)
        return image

    @staticmethod
    def _load_ip_adapter_refs(refs: Optional[List[Path]], max_side: int = 384) -> list[Image.Image]:
        """Load extra conditioning refs for ip-adapter style conditioning."""
        if not refs:
            return []
        images: list[Image.Image] = []
        for ref in refs[:3]:
            try:
                img = Image.open(ref).convert("RGB")
                w, h = img.size
                scale = min(max_side / max(w, 1), max_side / max(h, 1), 1.0)
                nw = max(256, int((w * scale) // 16 * 16))
                nh = max(256, int((h * scale) // 16 * 16))
                if (nw, nh) != (w, h):
                    img = img.resize((nw, nh), Image.LANCZOS)
                images.append(img)
            except Exception:
                continue
        return images

    def character_bind(
        self,
        character_refs: Dict[str, Path],
        character_brief: str,
        lora_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build character anchor object."""
        self.apply_optional_lora(lora_path)
        return {
            "brief": character_brief,
            "refs": {k: str(v) for k, v in character_refs.items()},
            "lora_path": lora_path,
            "anchor_type": "char_anchor",
        }

    def edit_image(
        self,
        image_path: Path,
        instruction: str,
        refs: Optional[List[Path]] = None,
        lora_path: Optional[str] = None,
        lora_paths: Optional[List[str]] = None,
        prompt_2: Optional[str] = None,
        max_sequence_length: int = 256,
        size: tuple[int, int] = (576, 832),
        num_inference_steps: int = 8,
        strength: float = 0.65,
    ) -> Image.Image:
        """Edit image without mask using Kontext model."""
        if lora_paths:
            self.apply_optional_loras(lora_paths)
        else:
            self.apply_optional_lora(lora_path)
        self.load_pipeline()
        input_image = Image.open(image_path).convert("RGB")
        input_image = self._resize_for_kontext(input_image)
        target_w, target_h = size
        target_w = max(384, (target_w // 16) * 16)
        target_h = max(384, (target_h // 16) * 16)
        input_image = input_image.resize((target_w, target_h), Image.LANCZOS)

        call_sig = inspect.signature(self.pipe.__call__)
        params = call_sig.parameters
        kwargs: Dict[str, Any] = {"prompt": instruction}
        if prompt_2 and "prompt_2" in params:
            kwargs["prompt_2"] = prompt_2
        if "max_sequence_length" in params:
            kwargs["max_sequence_length"] = max_sequence_length
        if "num_inference_steps" in params:
            kwargs["num_inference_steps"] = num_inference_steps
        if "guidance_scale" in params:
            kwargs["guidance_scale"] = 3.5
        if "strength" in params:
            kwargs["strength"] = min(max(float(strength), 0.35), 0.9)
        if "width" in params:
            kwargs["width"] = input_image.width
        if "height" in params:
            kwargs["height"] = input_image.height
        if "max_area" in params:
            kwargs["max_area"] = input_image.width * input_image.height
        if "_auto_resize" in params:
            kwargs["_auto_resize"] = False
        if "image" in params:
            kwargs["image"] = input_image
        elif "init_image" in params:
            kwargs["init_image"] = input_image
        elif "input_image" in params:
            kwargs["input_image"] = input_image
        elif "images" in params:
            kwargs["images"] = [input_image]
        # Optional extra references for every frame (only when pipeline actually supports it).
        ip_refs = self._load_ip_adapter_refs(refs)
        supports_ip_adapter = False
        if "ip_adapter_image" in params:
            transformer = getattr(self.pipe, "transformer", None)
            if transformer is not None and hasattr(transformer, "encoder_hid_proj"):
                supports_ip_adapter = True
        if ip_refs and supports_ip_adapter:
            kwargs["ip_adapter_image"] = ip_refs
        # If no image-like argument is present, this is a text2img fallback.
        try:
            result = self.pipe(**kwargs)
        except AttributeError as exc:
            # Some Flux-Kontext builds expose `ip_adapter_image` in signature but have no encoder_hid_proj.
            if "encoder_hid_proj" in str(exc) and "ip_adapter_image" in kwargs:
                kwargs.pop("ip_adapter_image", None)
                kwargs.pop("negative_ip_adapter_image", None)
                result = self.pipe(**kwargs)
                return result.images[0]
            raise
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            # First retry path: drop LoRA for this frame but keep reference conditioning.
            if self._active_lora_path:
                self.unload_lora()
                try:
                    if "num_inference_steps" in kwargs:
                        kwargs["num_inference_steps"] = min(int(kwargs["num_inference_steps"]), 6)
                    result = self.pipe(**kwargs)
                    return result.images[0]
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
            if "num_inference_steps" in kwargs:
                kwargs["num_inference_steps"] = min(int(kwargs["num_inference_steps"]), 4)
            if "width" in kwargs and "height" in kwargs:
                kwargs["width"] = min(int(kwargs["width"]), 512)
                kwargs["height"] = min(int(kwargs["height"]), 768)
                if "max_area" in kwargs:
                    kwargs["max_area"] = int(kwargs["width"]) * int(kwargs["height"])
            result = self.pipe(**kwargs)
        return result.images[0]

    def render_from_reference(
        self,
        reference_image_path: Path,
        instruction: str,
        refs: Optional[List[Path]] = None,
        lora_path: Optional[str] = None,
        lora_paths: Optional[List[str]] = None,
        prompt_2: Optional[str] = None,
        max_sequence_length: int = 256,
        size: tuple[int, int] = (576, 832),
        num_inference_steps: int = 8,
        strength: float = 0.65,
    ) -> Image.Image:
        """Render a new frame using reference image conditioning via Kontext."""
        return self.edit_image(
            reference_image_path,
            instruction,
            refs=([reference_image_path] + (refs or [])),
            lora_path=lora_path,
            lora_paths=lora_paths,
            prompt_2=prompt_2,
            max_sequence_length=max_sequence_length,
            size=size,
            num_inference_steps=num_inference_steps,
            strength=strength,
        )


class MockFluxKontextProvider:
    """Dry-run mock kontext provider."""

    def load_pipeline(self) -> None:
        """No-op for mock."""

    def apply_optional_lora(self, lora_path: Optional[str] = None) -> None:
        """No-op for mock."""

    def character_bind(
        self,
        character_refs: Dict[str, Path],
        character_brief: str,
        lora_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return mock char anchor."""
        return {
            "brief": character_brief,
            "refs": {k: str(v) for k, v in character_refs.items()},
            "lora_path": lora_path,
            "anchor_type": "char_anchor",
            "mock": True,
        }

    def edit_image(
        self,
        image_path: Path,
        instruction: str,
        refs: Optional[List[Path]] = None,
        lora_path: Optional[str] = None,
        lora_paths: Optional[List[str]] = None,
        prompt_2: Optional[str] = None,
        max_sequence_length: int = 256,
        size: tuple[int, int] = (576, 832),
        num_inference_steps: int = 8,
        strength: float = 0.65,
    ) -> Image.Image:
        """Create mock edited image."""
        temp = Path("/tmp/mock_edit.png")
        extra = f" | p2={prompt_2[:80] if prompt_2 else ''}"
        make_mock_image(temp, "edit", instruction + extra, size)
        return Image.open(temp).copy()

    def render_from_reference(
        self,
        reference_image_path: Path,
        instruction: str,
        refs: Optional[List[Path]] = None,
        lora_path: Optional[str] = None,
        lora_paths: Optional[List[str]] = None,
        prompt_2: Optional[str] = None,
        max_sequence_length: int = 256,
        size: tuple[int, int] = (576, 832),
        num_inference_steps: int = 8,
        strength: float = 0.65,
    ) -> Image.Image:
        """Create mock render from reference."""
        temp = Path("/tmp/mock_kontext_ref.png")
        extra = f" | p2={prompt_2[:80] if prompt_2 else ''}"
        make_mock_image(temp, "kontext_ref", f"{reference_image_path.name} | {instruction}{extra}", size)
        return Image.open(temp).copy()
