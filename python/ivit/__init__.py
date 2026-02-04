"""
iVIT-SDK: Innodisk Vision Intelligence Toolkit

Unified Computer Vision SDK with extensible hardware support.
Currently supports Intel (OpenVINO) and NVIDIA (TensorRT) platforms.

Note: C++ bindings are required for inference. Install with: pip install ivit-sdk
      Training module can be used independently with just PyTorch.

Example:
    >>> import ivit
    >>>
    >>> # One-liner inference (auto device selection)
    >>> model = ivit.load("yolov8n.onnx")
    >>> results = model("image.jpg")
    >>> results.show()
    >>>
    >>> # Task-specific classes
    >>> detector = ivit.Detector("yolov8n.onnx", device="cuda:0")
    >>> results = detector.predict("image.jpg")
    >>>
    >>> # Training (works independently without inference backend)
    >>> from ivit.train import Trainer, ImageFolderDataset
"""

__version__ = "1.0.0"
__author__ = "Innodisk AI Team"

import os as _os
import logging as _logging

_logger = _logging.getLogger(__name__)
_package_dir = _os.path.dirname(_os.path.abspath(__file__))

# ============================================================================
# Lazy loading for inference module (C++ bindings)
# This allows ivit.train to be used independently without OpenVINO/TensorRT
# ============================================================================

_inference_loaded = False
_inference_error = None


def _load_inference_module():
    """Lazily load inference C++ bindings."""
    global _inference_loaded, _inference_error

    if _inference_loaded:
        return True
    if _inference_error is not None:
        raise _inference_error

    import ctypes as _ctypes

    # Pre-load libivit.so if it exists in the package directory
    _libivit_path = _os.path.join(_package_dir, "libivit.so")
    if _os.path.exists(_libivit_path):
        try:
            _ctypes.CDLL(_libivit_path, mode=_ctypes.RTLD_GLOBAL)
        except OSError as e:
            _logger.debug(f"Failed to preload libivit.so: {e}")

    try:
        from ._ivit_core import (
            # Enums
            DataType,
            Precision,
            Layout,
            # Configuration
            LoadConfig,
            InferConfig,
            # Data structures
            TensorInfo,
            DeviceInfo,
            BBox,
            Detection,
            ClassificationResult,
            Keypoint,
            Pose,
            # Results
            Results,
            # Vision models
            Classifier,
            Detector,
            Segmentor,
            # Model
            Model,
            load_model,
            # Callback system
            CallbackEvent,
            CallbackContext,
            # Runtime config
            OpenVINOConfig,
            TensorRTConfig,
            QNNConfig,
            # Stream
            StreamResult,
            StreamIterator,
            # Functions
            version,
            list_devices,
            get_best_device,
            set_log_level,
            set_cache_dir,
            # Exceptions
            IVITError,
            ModelLoadError,
            InferenceError,
            DeviceNotFoundError,
            UnsupportedFormatError,
        )

        # Inject into module globals
        globals().update({
            'DataType': DataType,
            'Precision': Precision,
            'Layout': Layout,
            'LoadConfig': LoadConfig,
            'InferConfig': InferConfig,
            'TensorInfo': TensorInfo,
            'DeviceInfo': DeviceInfo,
            'BBox': BBox,
            'Detection': Detection,
            'ClassificationResult': ClassificationResult,
            'Keypoint': Keypoint,
            'Pose': Pose,
            'Results': Results,
            'Classifier': Classifier,
            'Detector': Detector,
            'Segmentor': Segmentor,
            'Model': Model,
            'load_model': load_model,
            'CallbackEvent': CallbackEvent,
            'CallbackContext': CallbackContext,
            'OpenVINOConfig': OpenVINOConfig,
            'TensorRTConfig': TensorRTConfig,
            'QNNConfig': QNNConfig,
            'StreamResult': StreamResult,
            'StreamIterator': StreamIterator,
            'version': version,
            'list_devices': list_devices,
            'get_best_device': get_best_device,
            'set_log_level': set_log_level,
            'set_cache_dir': set_cache_dir,
            'IVITError': IVITError,
            'ModelLoadError': ModelLoadError,
            'InferenceError': InferenceError,
            'DeviceNotFoundError': DeviceNotFoundError,
            'UnsupportedFormatError': UnsupportedFormatError,
        })

        _inference_loaded = True
        return True

    except ImportError as e:
        _inference_error = ImportError(
            "iVIT-SDK inference requires C++ bindings. "
            "Please build and install with: pip install -e . "
            f"(Original error: {e})"
        )
        _inference_error.__cause__ = e
        raise _inference_error


def __getattr__(name):
    """Lazy load inference module when accessing inference-related attributes."""
    # List of attributes that require inference module
    _inference_attrs = {
        'DataType', 'Precision', 'Layout',
        'LoadConfig', 'InferConfig',
        'TensorInfo', 'DeviceInfo', 'BBox', 'Detection',
        'ClassificationResult', 'Keypoint', 'Pose',
        'Results', 'Classifier', 'Detector', 'Segmentor',
        'Model', 'load_model',
        'CallbackEvent', 'CallbackContext',
        'OpenVINOConfig', 'TensorRTConfig', 'QNNConfig',
        'StreamResult', 'StreamIterator',
        'version', 'list_devices', 'get_best_device',
        'set_log_level', 'set_cache_dir',
        'IVITError', 'ModelLoadError', 'InferenceError',
        'DeviceNotFoundError', 'UnsupportedFormatError',
        'devices', 'Device', 'D', 'zoo',
    }

    if name in _inference_attrs:
        _load_inference_module()
        # After loading, the attribute should be in globals
        if name in globals():
            return globals()[name]
        # For devices/zoo, load them now
        if name == 'devices':
            from .devices import devices
            return devices
        if name == 'Device':
            from .devices import Device
            return Device
        if name == 'D':
            from .devices import D
            return D
        if name == 'zoo':
            from . import zoo
            return zoo

    raise AttributeError(f"module 'ivit' has no attribute '{name}'")


def is_cpp_available():
    """Check if C++ bindings are available."""
    try:
        _load_inference_module()
        return True
    except ImportError:
        return False


def is_inference_available():
    """Check if inference module is available."""
    return is_cpp_available()


def get_runtime_info():
    """Get runtime information about iVIT SDK."""
    cpp_available = is_cpp_available()
    return {
        "version": __version__,
        "cpp_bindings": cpp_available,
        "mode": "C++ accelerated" if cpp_available else "Training only",
    }


# Import extended Python exceptions (always available)
from .core.exceptions import (
    BackendNotAvailableError,
    InvalidInputError,
    ModelNotLoadedError,
    ConfigurationError,
    ModelConversionError,
    ResourceExhaustedError,
    UnsupportedOperationError,
    wrap_error,
)


def load(
    source: str,
    device: str = "auto",
    task: str = None,
    **kwargs
):
    """
    Load a model for inference.

    Args:
        source: Model path (.onnx, .xml, .engine) or Model Zoo name
        device: Target device ("auto", "cpu", "cuda:0", "npu")
        task: Task type hint ("detect", "classify", "segment", "pose")

    Returns:
        Model: Loaded model ready for inference

    Examples:
        >>> model = ivit.load("yolox-s.onnx")
        >>> results = model.predict("image.jpg")
    """
    # Ensure inference module is loaded
    _load_inference_module()

    from .devices import devices

    if hasattr(device, 'id'):
        device = device.id
    elif device == "auto":
        best = devices.best()
        device = best.id

    config = globals()['LoadConfig']()
    config.device = device
    if task:
        config.task = task
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return globals()['load_model'](source, config)


__all__ = [
    # Version
    "__version__",

    # Main API
    "load",
    "load_model",
    "devices",
    "Device",
    "D",

    # Training module
    "train",

    # Enums
    "DataType",
    "Precision",
    "Layout",

    # Configuration
    "LoadConfig",
    "InferConfig",
    "OpenVINOConfig",
    "TensorRTConfig",

    "QNNConfig",

    # Data structures
    "TensorInfo",
    "DeviceInfo",
    "BBox",
    "Detection",
    "ClassificationResult",
    "Keypoint",
    "Pose",

    # Results
    "Results",

    # Model
    "Model",

    # Vision models
    "Classifier",
    "Detector",
    "Segmentor",

    # Callback
    "CallbackEvent",
    "CallbackContext",

    # Stream
    "StreamResult",
    "StreamIterator",

    # Functions
    "version",
    "list_devices",
    "get_best_device",
    "set_log_level",
    "set_cache_dir",
    "is_cpp_available",
    "is_inference_available",
    "get_runtime_info",

    # Exceptions
    "IVITError",
    "ModelLoadError",
    "InferenceError",
    "DeviceNotFoundError",
    "BackendNotAvailableError",
    "InvalidInputError",
    "ModelNotLoadedError",
    "ConfigurationError",
    "ModelConversionError",
    "ResourceExhaustedError",
    "UnsupportedOperationError",
    "UnsupportedFormatError",
]
