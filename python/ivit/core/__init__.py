"""
Core module for iVIT-SDK.
"""

from .types import (
    LoadConfig,
    InferConfig,
    DeviceInfo,
    BBox,
    Detection,
    ClassificationResult,
    Keypoint,
    Pose,
    TensorInfo,
)

from .result import Results
from .model import Model, load_model
from .device import list_devices, get_best_device, get_device

# Callback system
from .callbacks import (
    CallbackManager,
    CallbackContext,
    CallbackEvent,
    callback,
    LatencyLogger,
    FPSCounter,
    DetectionFilter,
)

# Runtime configuration
from .runtime_config import (
    OpenVINOConfig,
    TensorRTConfig,
    ONNXRuntimeConfig,
    SNPEConfig,
)

# Processors
from .processors import (
    BasePreProcessor,
    BasePostProcessor,
    LetterboxPreProcessor,
    CenterCropPreProcessor,
    YOLOPostProcessor,
    ClassificationPostProcessor,
    get_preprocessor,
    get_postprocessor,
    register_preprocessor,
    register_postprocessor,
)

# Exceptions
from .exceptions import (
    IVITError,
    ModelLoadError,
    DeviceNotFoundError,
    BackendNotAvailableError,
    InferenceError,
    InvalidInputError,
    ModelNotLoadedError,
    ConfigurationError,
    ModelConversionError,
    ResourceExhaustedError,
    UnsupportedOperationError,
    wrap_error,
)

__all__ = [
    "LoadConfig",
    "InferConfig",
    "DeviceInfo",
    "BBox",
    "Detection",
    "ClassificationResult",
    "Keypoint",
    "Pose",
    "TensorInfo",
    "Results",
    "Model",
    "load_model",
    "list_devices",
    "get_best_device",
    "get_device",
    # Callbacks
    "CallbackManager",
    "CallbackContext",
    "CallbackEvent",
    "callback",
    "LatencyLogger",
    "FPSCounter",
    "DetectionFilter",
    # Runtime configs
    "OpenVINOConfig",
    "TensorRTConfig",
    "ONNXRuntimeConfig",
    "SNPEConfig",
    # Exceptions
    "IVITError",
    "ModelLoadError",
    "DeviceNotFoundError",
    "BackendNotAvailableError",
    "InferenceError",
    "InvalidInputError",
    "ModelNotLoadedError",
    "ConfigurationError",
    "ModelConversionError",
    "ResourceExhaustedError",
    "UnsupportedOperationError",
    "wrap_error",
]
