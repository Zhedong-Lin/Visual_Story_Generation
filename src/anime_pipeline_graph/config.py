"""Runtime config."""

from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class ModelConfig(BaseModel):
    """Model names used by providers."""

    flux_base_model: str = "black-forest-labs/FLUX.1-dev"
    flux_kontext_model: str = "black-forest-labs/FLUX.1-Kontext-dev"
    flux_fill_model: str = "black-forest-labs/FLUX.1-Fill-dev"


class AppConfig(BaseSettings):
    """Application settings from env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    qwen_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1", alias="QWEN_BASE_URL"
    )
    qwen_model_name: str = Field(default="qwen-vl-max-latest", alias="QWEN_MODEL_NAME")
    qwen_timeout_seconds: int = Field(default=180, alias="QWEN_TIMEOUT_SECONDS")
    qwen_max_retries: int = Field(default=1, alias="QWEN_MAX_RETRIES")
    hf_token: str = Field(default="", alias="HF_TOKEN")
    default_output_dir: str = Field(default="runs", alias="DEFAULT_OUTPUT_DIR")
    image_backend: str = Field(default="local", alias="IMAGE_BACKEND")
    flux_base_model: str = Field(default="black-forest-labs/FLUX.1-dev", alias="FLUX_BASE_MODEL")
    flux_kontext_model: str = Field(
        default="black-forest-labs/FLUX.1-Kontext-dev", alias="FLUX_KONTEXT_MODEL"
    )
    flux_fill_model: str = Field(default="black-forest-labs/FLUX.1-Fill-dev", alias="FLUX_FILL_MODEL")
    ark_api_key: str = Field(default="", alias="ARK_API_KEY")
    ark_base_url: str = Field(default="https://ark.cn-beijing.volces.com/api/v3", alias="ARK_BASE_URL")
    ark_images_endpoint: str = Field(default="/images/generations", alias="ARK_IMAGES_ENDPOINT")
    ark_seedream_model: str = Field(default="ep-xxxxxx", alias="ARK_SEEDREAM_MODEL")
    ark_timeout_seconds: int = Field(default=120, alias="ARK_TIMEOUT_SECONDS")
    ark_response_format: str = Field(default="url", alias="ARK_RESPONSE_FORMAT")
    ark_default_size: str = Field(default="1024x1024", alias="ARK_DEFAULT_SIZE")
    ark_min_pixels: int = Field(default=921600, alias="ARK_MIN_PIXELS")

    @property
    def output_dir(self) -> Path:
        """Return run output directory."""
        return Path(self.default_output_dir)


def default_lora_map() -> Dict[str, str]:
    """Default empty LoRA map."""
    return {}
