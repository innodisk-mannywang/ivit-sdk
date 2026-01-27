"""
iVIT-SDK: Innodisk Vision Intelligence Toolkit

Unified Computer Vision SDK with extensible hardware support.
Currently supports Intel (OpenVINO) and NVIDIA (TensorRT) platforms.

Key Features:
- Automatic device selection based on current platform
- Vendor-aware priority (NVIDIA dGPU > Intel iGPU > NPU > CPU)
- Multiple selection strategies (latency, efficiency, balanced)

Example:
    >>> import ivit
    >>>
    >>> # One-liner inference (auto device selection)
    >>> model = ivit.load("yolov8n.onnx")  # Auto-selects best device
    >>> results = model("image.jpg")
    >>> results.show()
    >>>
    >>> # Device selection strategies
    >>> ivit.devices.best()                    # Default: latency-optimized
    >>> ivit.devices.best(strategy="efficiency")  # Power-efficient (NPU preferred)
    >>>
    >>> # Use specific device
    >>> model = ivit.load("model.onnx", device="cuda:0")
    >>>
    >>> # Task-specific classes
    >>> detector = ivit.Detector("yolov8n.onnx", device="cuda:0")
    >>> results = detector.predict("image.jpg")
    >>> print(f"Found {len(results)} objects")
"""

__version__ = "1.0.0"
__author__ = "Innodisk AI Team"

# Try to import C++ bindings
_HAS_CPP_BINDING = False

# Pre-load libivit.so if it exists in the package directory
# This is needed because the binding module depends on libivit.so
import os as _os
import ctypes as _ctypes

_package_dir = _os.path.dirname(_os.path.abspath(__file__))
_libivit_path = _os.path.join(_package_dir, "libivit.so")

if _os.path.exists(_libivit_path):
    try:
        _ctypes.CDLL(_libivit_path, mode=_ctypes.RTLD_GLOBAL)
    except OSError:
        pass  # Failed to load, will fall back to pure Python

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
    _HAS_CPP_BINDING = True
except ImportError:
    # C++ bindings not available - this is expected in pure Python installations
    # The SDK works fully with pure Python, C++ bindings are optional for better performance
    pass

    # Import pure Python fallback
    from .core.types import (
        LoadConfig,
        InferConfig,
        TensorInfo,
        DeviceInfo,
        SelectionStrategy,
        BBox,
        Detection,
        ClassificationResult,
        Keypoint,
        Pose,
    )

    # Import Python exceptions as fallback
    from .core.exceptions import (
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
    )

    # Alias for compatibility with C++ binding
    UnsupportedFormatError = UnsupportedOperationError

    # Import device functions from Python core
    from .core.device import (
        list_devices,
        get_best_device,
        get_device,
    )

    def version():
        return __version__


def is_cpp_available():
    """Check if C++ bindings are available."""
    return _HAS_CPP_BINDING


def get_runtime_info():
    """
    Get runtime information about iVIT SDK.

    Returns:
        dict: Runtime information including version and backend mode
    """
    return {
        "version": __version__,
        "cpp_bindings": _HAS_CPP_BINDING,
        "mode": "C++ accelerated" if _HAS_CPP_BINDING else "Pure Python",
    }


# Always import Python exceptions for extended exception types
# These may extend or supplement the C++ exceptions
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

# If C++ bindings are not available, we already imported all exceptions above
# If C++ bindings are available, we still want to use the Python base exceptions
# for cases not covered by C++
if _HAS_CPP_BINDING:
    # Keep C++ exceptions for core types, but use Python for extended types
    pass
else:
    # Already imported in fallback section above
    pass

# Import devices module for device discovery
from .devices import devices, Device, D

# Import load function from core
from .core.model import load_model as _load_model

# Import Model Zoo
from . import zoo

# Import Training module (lazy load to avoid torch dependency)
try:
    from . import train
except ImportError:
    train = None


def load(
    source: str,
    device: str = "auto",
    task: str = None,
    **kwargs
):
    """
    Load a model for inference.

    This is the main entry point for iVIT SDK, providing a simple
    one-liner API similar to Ultralytics.

    Args:
        source: Model path (.onnx, .xml, .engine) or Model Zoo name
        device: Target device
            - "auto": Auto-select best available device
            - "cpu": CPU inference
            - "cuda:0": NVIDIA GPU (index 0)
            - "npu": Intel NPU
            - Device object from ivit.devices
        task: Task type hint (auto-detected if None)
            - "detect": Object detection
            - "classify": Image classification
            - "segment": Semantic segmentation
            - "pose": Pose estimation

    Returns:
        Model: Loaded model ready for inference

    Examples:
        >>> import ivit
        >>>
        >>> # Simple load and inference
        >>> model = ivit.load("yolov8n.onnx")
        >>> results = model("image.jpg")
        >>> results.show()
        >>>
        >>> # With device selection
        >>> model = ivit.load("model.onnx", device=ivit.devices.cuda())
        >>> model = ivit.load("model.onnx", device=ivit.devices.best())
        >>>
        >>> # With explicit task
        >>> model = ivit.load("resnet50.onnx", task="classify")
    """
    # Handle Device object
    if hasattr(device, 'id'):
        device = device.id
    elif device == "auto":
        # Use best available device
        best = devices.best()
        device = best.id

    return _load_model(source, device=device, task=task, **kwargs)


__all__ = [
    # Version
    "__version__",

    # Main API (Ultralytics-style)
    "load",
    "devices",
    "Device",
    "D",

    # Training module
    "train",

    # Enums (if available)
    "DataType",
    "Precision",
    "Layout",

    # Configuration
    "LoadConfig",
    "InferConfig",

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

    # Vision models
    "Classifier",
    "Detector",
    "Segmentor",

    # Functions
    "version",
    "list_devices",
    "get_best_device",
    "set_log_level",
    "set_cache_dir",
    "is_cpp_available",
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
