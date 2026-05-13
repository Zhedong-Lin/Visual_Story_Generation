"""Doubao SeedDream4 API provider (text-to-image)."""

from __future__ import annotations

import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx
from PIL import Image


class SeedDream4ApiProvider:
    """Remote API provider compatible with base provider interface."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        endpoint: str = "/images/generations",
        timeout_seconds: int = 120,
        response_format: str = "url",
        default_size: str = "1024x1024",
        min_pixels: int = 921600,
        ref_max_side: int = 768,
        ref_jpeg_quality: int = 85,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.timeout_seconds = int(timeout_seconds)
        self.response_format = response_format
        self.default_size = default_size
        self.min_pixels = int(min_pixels)
        self.ref_max_side = int(ref_max_side)
        self.ref_jpeg_quality = int(ref_jpeg_quality)

    def load_pipeline(self) -> None:
        """No-op for remote API provider."""

    def apply_optional_lora(self, lora_path: Optional[str] = None) -> None:
        """No-op: API mode does not support local LoRA injection."""

    def apply_optional_loras(self, lora_paths: Optional[list[str]] = None) -> None:
        """No-op: API mode does not support local LoRA injection."""

    def _to_size_text(self, size: Optional[Tuple[int, int]], fallback: str) -> str:
        if not size:
            return fallback
        w, h = int(size[0]), int(size[1])
        area = max(w, 1) * max(h, 1)
        if area < self.min_pixels:
            # Keep aspect ratio while meeting API minimum pixel requirement.
            scale = (self.min_pixels / float(area)) ** 0.5
            w = int(round(w * scale))
            h = int(round(h * scale))
            # Guard rounding drift.
            while w * h < self.min_pixels:
                if w <= h:
                    w += 1
                else:
                    h += 1
        return f"{w}x{h}"

    @staticmethod
    def _extract_image_payload(data: Dict[str, Any]) -> Dict[str, Any]:
        # OpenAI-compatible shape: {"data":[{"url":"..."}]}
        items = data.get("data")
        if isinstance(items, list) and items:
            first = items[0]
            if isinstance(first, dict):
                return first
        # Fallback for provider-specific direct response.
        if isinstance(data, dict):
            return data
        return {}

    def _auth_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise RuntimeError(
                "Missing ARK_API_KEY for SeedDream4 API mode. "
                "Please set ARK_API_KEY in your environment."
            )
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _validate_model_name(self) -> None:
        model = (self.model_name or "").strip()
        if not model:
            raise RuntimeError("Missing ARK_SEEDREAM_MODEL. Please set your SeedDream model or endpoint id.")
        if model in {"ep-xxxxxx", "ep-xxxxx", "your-endpoint-id"}:
            raise RuntimeError(
                "ARK_SEEDREAM_MODEL is still a placeholder. "
                "Please replace it with a real value, e.g. an endpoint id like `ep-...` "
                "or a supported model name like `doubao-seedream-4-0-250828`."
            )

    def _build_prompt(self, prompt_pack: Dict[str, Any]) -> str:
        prompt = str(prompt_pack.get("prompt", prompt_pack.get("positive_prompt", "anime illustration"))).strip()
        prompt_2 = str(prompt_pack.get("prompt_2", "")).strip()
        if prompt_2:
            return f"{prompt}\n{prompt_2}"
        return prompt

    def _image_file_to_data_url(self, path: Path) -> Optional[str]:
        if not path.exists() or not path.is_file():
            return None
        # Compress and resize reference to reduce request body and avoid upload write timeout.
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            max_side = max(w, h, 1)
            if max_side > self.ref_max_side:
                scale = self.ref_max_side / float(max_side)
                nw = max(256, int(round(w * scale)))
                nh = max(256, int(round(h * scale)))
                img = img.resize((nw, nh), Image.LANCZOS)
            buf = BytesIO()
            img.save(
                buf,
                format="JPEG",
                quality=max(40, min(self.ref_jpeg_quality, 95)),
                optimize=True,
            )
            raw = buf.getvalue()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _prepare_reference_images(self, refs: Optional[Dict[str, Path]]) -> list[str]:
        if not refs:
            return []
        urls: list[str] = []
        seen = set()
        for _, p in refs.items():
            try:
                data_url = self._image_file_to_data_url(Path(p))
            except Exception:
                data_url = None
            if not data_url:
                continue
            if data_url in seen:
                continue
            seen.add(data_url)
            urls.append(data_url)
            if len(urls) >= 3:
                break
        return urls

    def _refs_from_paths(self, image_path: Path, refs: Optional[list[Path]] = None, mask_path: Optional[Path] = None) -> Dict[str, Path]:
        merged: Dict[str, Path] = {"primary": image_path}
        idx = 1
        for p in refs or []:
            if str(p) == str(image_path):
                continue
            merged[f"extra_{idx}"] = p
            idx += 1
        if mask_path is not None:
            merged[f"mask_{idx}"] = mask_path
        return merged

    @staticmethod
    def _request_payload_candidates(base_payload: Dict[str, Any], refs: list[str]) -> list[Dict[str, Any]]:
        if not refs:
            return [base_payload]
        first = refs[0]
        candidates: list[Dict[str, Any]] = []
        # Candidate 1: single-image field used by many image-edit/image-reference APIs.
        p1 = dict(base_payload)
        p1["image"] = first
        candidates.append(p1)
        # Candidate 2: multi-image field.
        p2 = dict(base_payload)
        p2["images"] = refs
        candidates.append(p2)
        # Candidate 3: explicit url-list style.
        p3 = dict(base_payload)
        p3["image_urls"] = refs
        candidates.append(p3)
        return candidates

    def generate(
        self,
        prompt_pack: Dict[str, Any],
        refs: Optional[Dict[str, Path]] = None,
        seed: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """Generate one image from SeedDream4 API and return a PIL image."""
        self._validate_model_name()
        endpoint = f"{self.base_url}/{self.endpoint.lstrip('/')}"
        base_payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": self._build_prompt(prompt_pack),
            "size": self._to_size_text(size, self.default_size),
            "response_format": self.response_format,
        }
        if seed is not None:
            base_payload["seed"] = int(seed)
        ref_images = self._prepare_reference_images(refs)
        payload_candidates = self._request_payload_candidates(base_payload, ref_images)

        timeout = httpx.Timeout(
            connect=min(float(self.timeout_seconds), 30.0),
            read=max(float(self.timeout_seconds), 180.0),
            write=max(float(self.timeout_seconds), 180.0),
            pool=30.0,
        )
        with httpx.Client(timeout=timeout) as client:
            last_error = ""
            resp = None
            for idx, payload in enumerate(payload_candidates):
                # Retry transient network failures before switching payload shape.
                max_retries = 3
                retryable_exc = (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError)
                for retry_idx in range(max_retries):
                    try:
                        resp = client.post(endpoint, headers=self._auth_headers(), json=payload)
                        break
                    except retryable_exc as exc:
                        last_error = (
                            f"attempt={idx+1}/{len(payload_candidates)}, "
                            f"retry={retry_idx+1}/{max_retries}, network_error={exc}"
                        )
                        if retry_idx + 1 >= max_retries:
                            resp = None
                            break
                        # Small exponential backoff for transient peer resets/timeouts.
                        time.sleep(0.8 * (2 ** retry_idx))
                if resp is None:
                    # Continue trying alternate payload shapes (e.g., different reference fields).
                    continue
                if resp.status_code < 400:
                    break
                body = ""
                try:
                    body = resp.text
                except Exception:
                    body = "<unavailable>"
                last_error = f"attempt={idx+1}/{len(payload_candidates)}, status={resp.status_code}, body={body[:800]}"
                # If request did not fail because of unsupported image parameter, stop retrying.
                lower_body = body.lower()
                has_ref = any(k in payload for k in ("image", "images", "image_urls"))
                invalid_param = ("invalidparameter" in lower_body) or ("parameter" in lower_body)
                if (not has_ref) or (not invalid_param):
                    break
            if resp is None or resp.status_code >= 400:
                hint = ""
                if resp is not None and resp.status_code == 404:
                    hint = (
                        " Hint: 404 on Ark image API is usually caused by one of: "
                        "(1) wrong ARK_SEEDREAM_MODEL/endpoint id, "
                        "(2) endpoint not in cn-beijing region, "
                        "(3) account has no permission for this image model."
                    )
                raise RuntimeError(
                    f"SeedDream4 API request failed: url={endpoint}, model={self.model_name}. "
                    f"Last error: {last_error}{hint}"
                )

            parsed = resp.json()
            item = self._extract_image_payload(parsed)

            image_url = item.get("url")
            if isinstance(image_url, str) and image_url.strip():
                img_resp = client.get(image_url)
                img_resp.raise_for_status()
                return Image.open(BytesIO(img_resp.content)).convert("RGB")

            b64 = item.get("b64_json")
            if isinstance(b64, str) and b64.strip():
                raw = base64.b64decode(b64)
                return Image.open(BytesIO(raw)).convert("RGB")

        raise RuntimeError(
            "SeedDream4 API response does not contain image data. "
            "Expected `data[0].url` or `data[0].b64_json`."
        )

    def edit_image(
        self,
        image_path: Path,
        instruction: str,
        refs: Optional[list[Path]] = None,
        lora_path: Optional[str] = None,
        lora_paths: Optional[list[str]] = None,
        prompt_2: Optional[str] = None,
        max_sequence_length: int = 256,
        size: tuple[int, int] = (576, 832),
        num_inference_steps: int = 8,
        strength: float = 0.65,
    ) -> Image.Image:
        """API-mode edit: image-conditioned generation using source image + optional refs."""
        del lora_path, lora_paths, max_sequence_length, num_inference_steps, strength
        pack = {
            "prompt": instruction or "polish details while preserving identity and anime style",
            "prompt_2": prompt_2 or "",
        }
        api_refs = self._refs_from_paths(image_path=image_path, refs=refs, mask_path=None)
        return self.generate(pack, refs=api_refs, size=size)

    def fill_edit(
        self,
        image_path: Path,
        mask_path: Optional[Path],
        instruction: str,
    ) -> Image.Image:
        """API-mode fill-edit: best-effort masked edit via image+mask references."""
        pack = {
            "prompt": instruction or "local detail correction while preserving anime identity",
            "prompt_2": "",
        }
        api_refs = self._refs_from_paths(image_path=image_path, refs=None, mask_path=mask_path)
        return self.generate(pack, refs=api_refs, size=(576, 832))
