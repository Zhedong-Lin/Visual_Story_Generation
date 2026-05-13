"""Enums for pipeline domain."""

from enum import Enum


class TaskType(str, Enum):
    """Supported task type."""

    SINGLE_IMAGE = "single_image"
    STORYBOARD = "storyboard"


class BackendType(str, Enum):
    """Skill backend type."""

    HF_LOCAL = "hf_local"
    QWEN_API = "qwen_api"
    BUILDER = "builder"
    LOCAL_PREPROCESS = "local_preprocess"


class CostLevel(str, Enum):
    """Cost level tags."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LatencyLevel(str, Enum):
    """Latency level tags."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
