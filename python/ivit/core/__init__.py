"""
Core module for iVIT-SDK.

Re-exports C++ binding types and Python-only utilities.
"""

# Core types from C++ bindings
from .._ivit_core import (
    LoadConfig,
    InferConfig,
    DeviceInfo,
    BBox,
    Detection,
    ClassificationResult,
    Keypoint,
    Pose,
    TensorInfo,
    Results,
    Model,
    load_model,
    CallbackEvent,
    CallbackContext,
    OpenVINOConfig,
    TensorRTConfig,
    ONNXRuntimeConfig,
    QNNConfig,
)

# Device functions
from .device import (
    list_devices,
    get_best_device,
    get_device,
)

# Python-only exceptions (supplement C++ exceptions)
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
    # Types
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
    # Model
    "Model",
    "load_model",
    # Device
    "list_devices",
    "get_best_device",
    "get_device",
    # Callbacks
    "CallbackEvent",
    "CallbackContext",
    # Runtime configs
    "OpenVINOConfig",
    "TensorRTConfig",
    "ONNXRuntimeConfig",
    "QNNConfig",
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
