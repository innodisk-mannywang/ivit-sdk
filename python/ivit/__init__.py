"""
iVIT-SDK: Innodisk Vision Intelligence Toolkit

Unified Computer Vision SDK for Intel/NVIDIA/Qualcomm platforms.

Example:
    >>> import ivit
    >>>
    >>> # List available devices
    >>> for d in ivit.list_devices():
    ...     print(f"{d.id}: {d.name}")
    >>>
    >>> # Load and run detector
    >>> detector = ivit.Detector("yolov8n.onnx", device="cuda:0")
    >>> results = detector.predict("image.jpg")
    >>> print(f"Found {results.num_detections()} objects")
    >>>
    >>> # Visualize results
    >>> vis = results.visualize(image)
"""

__version__ = "1.0.0"
__author__ = "Innodisk AI Team"

# Try to import C++ bindings
_HAS_CPP_BINDING = False
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
except ImportError as e:
    import warnings
    warnings.warn(f"C++ bindings not available: {e}. Using pure Python fallback.")

    # Import pure Python fallback
    from .core.types import (
        LoadConfig,
        InferConfig,
        TensorInfo,
        DeviceInfo,
        BBox,
        Detection,
        ClassificationResult,
        Keypoint,
        Pose,
    )

    # Placeholder functions
    def version():
        return __version__

    def list_devices():
        return []

    def get_best_device(task="", priority="performance"):
        return DeviceInfo(
            id="cpu",
            name="CPU (Fallback)",
            backend="onnxruntime",
            type="cpu"
        )


def is_cpp_available():
    """Check if C++ bindings are available."""
    return _HAS_CPP_BINDING


__all__ = [
    # Version
    "__version__",

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

    # Exceptions
    "IVITError",
    "ModelLoadError",
    "InferenceError",
    "DeviceNotFoundError",
    "UnsupportedFormatError",
]
