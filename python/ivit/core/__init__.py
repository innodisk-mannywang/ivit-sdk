"""
Core module for iVIT-SDK.

Re-exports C++ binding types and Python-only utilities.
Uses lazy loading to allow training module to work independently.
"""

# Python-only exceptions (always available, no C++ dependency)
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

# Lazy loading for C++ binding types
_cpp_loaded = False


def _load_cpp_bindings():
    """Lazily load C++ bindings."""
    global _cpp_loaded
    if _cpp_loaded:
        return

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
        QNNConfig,
    )

    globals().update({
        'LoadConfig': LoadConfig,
        'InferConfig': InferConfig,
        'DeviceInfo': DeviceInfo,
        'BBox': BBox,
        'Detection': Detection,
        'ClassificationResult': ClassificationResult,
        'Keypoint': Keypoint,
        'Pose': Pose,
        'TensorInfo': TensorInfo,
        'Results': Results,
        'Model': Model,
        'load_model': load_model,
        'CallbackEvent': CallbackEvent,
        'CallbackContext': CallbackContext,
        'OpenVINOConfig': OpenVINOConfig,
        'TensorRTConfig': TensorRTConfig,
        'QNNConfig': QNNConfig,
    })

    from .device import (
        list_devices,
        get_best_device,
        get_device,
    )

    globals().update({
        'list_devices': list_devices,
        'get_best_device': get_best_device,
        'get_device': get_device,
    })

    _cpp_loaded = True


def __getattr__(name):
    """Lazy load C++ bindings when accessed."""
    _cpp_attrs = {
        'LoadConfig', 'InferConfig', 'DeviceInfo', 'BBox', 'Detection',
        'ClassificationResult', 'Keypoint', 'Pose', 'TensorInfo', 'Results',
        'Model', 'load_model', 'CallbackEvent', 'CallbackContext',
        'OpenVINOConfig', 'TensorRTConfig', 'QNNConfig',
        'list_devices', 'get_best_device', 'get_device',
    }

    if name in _cpp_attrs:
        _load_cpp_bindings()
        if name in globals():
            return globals()[name]

    raise AttributeError(f"module 'ivit.core' has no attribute '{name}'")

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
