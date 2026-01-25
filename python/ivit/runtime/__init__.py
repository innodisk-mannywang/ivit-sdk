"""
Runtime backends for iVIT-SDK.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Registry of available runtimes
_runtimes = {}


def get_runtime(backend: str):
    """
    Get runtime instance for backend.

    Args:
        backend: Backend name (openvino, tensorrt, snpe, onnxruntime)

    Returns:
        Runtime instance

    Raises:
        ValueError: If backend not found
    """
    backend = backend.lower()

    if backend in _runtimes:
        return _runtimes[backend]

    # Try to create runtime
    runtime = None

    if backend == "openvino":
        runtime = _create_openvino_runtime()
    elif backend == "tensorrt":
        runtime = _create_tensorrt_runtime()
    elif backend == "snpe":
        runtime = _create_snpe_runtime()
    elif backend in ("onnxruntime", "onnx"):
        runtime = _create_onnxruntime()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if runtime is None:
        raise ValueError(f"Failed to create runtime for backend: {backend}")

    _runtimes[backend] = runtime
    return runtime


def _create_openvino_runtime():
    """Create OpenVINO runtime."""
    try:
        from .openvino_runtime import OpenVINORuntime
        return OpenVINORuntime()
    except ImportError as e:
        logger.warning(f"OpenVINO not available: {e}")
        return None


def _create_tensorrt_runtime():
    """Create TensorRT runtime."""
    try:
        from .tensorrt_runtime import TensorRTRuntime
        return TensorRTRuntime()
    except ImportError as e:
        logger.warning(f"TensorRT not available: {e}")
        return None


def _create_snpe_runtime():
    """Create SNPE runtime."""
    try:
        from .snpe_runtime import SNPERuntime
        return SNPERuntime()
    except ImportError as e:
        logger.warning(f"SNPE not available: {e}")
        return None


def _create_onnxruntime():
    """Create ONNX Runtime."""
    try:
        from .onnx_runtime import ONNXRuntime
        return ONNXRuntime()
    except ImportError as e:
        logger.warning(f"ONNX Runtime not available: {e}")
        return None


def list_available_backends():
    """List available backends."""
    available = []

    try:
        from openvino import Core
        available.append("openvino")
    except ImportError:
        pass

    try:
        import tensorrt
        available.append("tensorrt")
    except ImportError:
        pass

    try:
        import onnxruntime
        available.append("onnxruntime")
    except ImportError:
        pass

    return available
