"""
iVIT Model Zoo - Pre-optimized models for edge deployment.

Provides easy access to pre-trained and optimized models.

Examples:
    >>> import ivit
    >>>
    >>> # List available models
    >>> ivit.zoo.list()
    >>> ivit.zoo.search("yolo")
    >>>
    >>> # Load model from zoo
    >>> model = ivit.zoo.load("yolov8n")
    >>> results = model("image.jpg")
    >>>
    >>> # Load with specific device
    >>> model = ivit.zoo.load("yolov8n", device="npu")
"""

from .registry import (
    list_models,
    search,
    get_model_info,
    download,
    load,
    ModelInfo,
)

__all__ = [
    "list_models",
    "search",
    "get_model_info",
    "download",
    "load",
    "ModelInfo",
]
