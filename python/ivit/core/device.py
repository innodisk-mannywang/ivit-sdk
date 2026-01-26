"""
Device management for iVIT-SDK.
"""

from typing import List, Optional, Tuple
import logging

from .types import DeviceInfo, BackendType

logger = logging.getLogger(__name__)

# Cache for devices
_devices_cache: Optional[List[DeviceInfo]] = None


def list_devices(refresh: bool = False) -> List[DeviceInfo]:
    """
    List all available inference devices.

    Args:
        refresh: Force refresh device list

    Returns:
        List of available devices

    Examples:
        >>> devices = list_devices()
        >>> for dev in devices:
        ...     print(f"{dev.id}: {dev.name} ({dev.backend})")
    """
    global _devices_cache

    if _devices_cache is not None and not refresh:
        return _devices_cache

    devices = []

    # Discover OpenVINO devices
    devices.extend(_discover_openvino_devices())

    # Discover TensorRT/CUDA devices
    devices.extend(_discover_tensorrt_devices())

    # Discover SNPE devices
    devices.extend(_discover_snpe_devices())

    # Always add CPU fallback
    if not any(d.id == "cpu" for d in devices):
        devices.append(DeviceInfo(
            id="cpu",
            name="CPU (ONNX Runtime)",
            backend="onnxruntime",
            type="cpu",
            is_available=True,
        ))

    _devices_cache = devices
    return devices


def get_device(device_id: str) -> DeviceInfo:
    """
    Get device by ID.

    Args:
        device_id: Device ID (e.g., "cpu", "cuda:0", "npu")

    Returns:
        Device information

    Raises:
        ValueError: If device not found
    """
    devices = list_devices()

    for dev in devices:
        if dev.id == device_id:
            return dev

    raise ValueError(f"Device not found: {device_id}")


def get_best_device(
    task: Optional[str] = None,
    priority: str = "performance"
) -> DeviceInfo:
    """
    Get the best device for a task.

    Args:
        task: Task type (optional)
        priority: Selection priority:
            - "performance": Highest performance
            - "efficiency": Best power efficiency
            - "memory": Most available memory

    Returns:
        Best device for the task

    Examples:
        >>> best = get_best_device(task="detection")
        >>> print(f"Using: {best.name}")
    """
    devices = list_devices()

    if not devices:
        raise RuntimeError("No devices available")

    # Priority order based on device type
    if priority == "performance":
        type_order = ["gpu", "npu", "vpu", "cpu"]
    elif priority == "efficiency":
        type_order = ["npu", "vpu", "gpu", "cpu"]
    else:  # memory
        type_order = ["gpu", "cpu", "npu", "vpu"]

    # Sort devices by priority
    def device_priority(dev: DeviceInfo) -> int:
        try:
            return type_order.index(dev.type)
        except ValueError:
            return len(type_order)

    sorted_devices = sorted(devices, key=device_priority)

    # Return first available device
    for dev in sorted_devices:
        if dev.is_available:
            return dev

    return sorted_devices[0]


def get_backend_for_device(device: str) -> str:
    """
    Get appropriate backend for device.

    Args:
        device: Device string (e.g., "auto", "cpu", "cuda:0")

    Returns:
        Backend name
    """
    if device == "auto":
        best = get_best_device()
        return best.backend

    device_lower = device.lower()

    if device_lower.startswith("cuda"):
        # Try TensorRT first, then fall back to ONNX Runtime CUDA
        if _tensorrt_available():
            return "tensorrt"
        elif _onnxruntime_cuda_available():
            return "onnxruntime"

    if device_lower in ("cpu", "gpu", "npu", "vpu") or device_lower.startswith("gpu:"):
        if _openvino_available():
            return "openvino"

    if device_lower in ("hexagon", "dsp", "htp"):
        if _snpe_available():
            return "snpe"

    # Fallback to ONNX Runtime
    return "onnxruntime"


def _discover_openvino_devices() -> List[DeviceInfo]:
    """Discover OpenVINO devices."""
    devices = []

    try:
        from openvino import Core
        core = Core()

        for device_name in core.available_devices:
            full_name = core.get_property(device_name, "FULL_DEVICE_NAME")

            device_type = "cpu"
            if "GPU" in device_name:
                device_type = "gpu"
            elif "NPU" in device_name:
                device_type = "npu"
            elif "VPU" in device_name or "MYRIAD" in device_name:
                device_type = "vpu"

            device_id = device_name.lower()
            if device_id == "cpu":
                device_id = "cpu"
            elif "GPU" in device_name:
                device_id = f"gpu:{device_name.split('.')[-1]}" if "." in device_name else "gpu:0"
            elif "NPU" in device_name:
                device_id = "npu"

            devices.append(DeviceInfo(
                id=device_id,
                name=full_name,
                backend="openvino",
                type=device_type,
                supported_precisions=["fp32", "fp16", "int8"],
                is_available=True,
            ))

        logger.debug(f"Found {len(devices)} OpenVINO devices")

    except ImportError:
        logger.debug("OpenVINO not available")
    except Exception as e:
        logger.warning(f"Error discovering OpenVINO devices: {e}")

    return devices


def _discover_tensorrt_devices() -> List[DeviceInfo]:
    """Discover TensorRT/CUDA devices."""
    devices = []

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(DeviceInfo(
                    id=f"cuda:{i}",
                    name=props.name,
                    backend="tensorrt",
                    type="gpu",
                    memory_total=props.total_memory,
                    memory_available=props.total_memory - torch.cuda.memory_allocated(i),
                    supported_precisions=["fp32", "fp16", "int8"],
                    is_available=True,
                ))

        logger.debug(f"Found {len(devices)} CUDA devices")

    except ImportError:
        # Try pycuda
        try:
            import pycuda.driver as cuda
            cuda.init()

            for i in range(cuda.Device.count()):
                dev = cuda.Device(i)
                devices.append(DeviceInfo(
                    id=f"cuda:{i}",
                    name=dev.name(),
                    backend="tensorrt",
                    type="gpu",
                    memory_total=dev.total_memory(),
                    supported_precisions=["fp32", "fp16", "int8"],
                    is_available=True,
                ))

        except ImportError:
            logger.debug("CUDA not available")
        except Exception as e:
            logger.warning(f"Error discovering CUDA devices: {e}")

    return devices


def _discover_snpe_devices() -> List[DeviceInfo]:
    """Discover SNPE devices."""
    devices = []

    # SNPE discovery would require the SNPE SDK
    # This is a placeholder for actual implementation

    return devices


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        try:
            import pycuda.driver as cuda
            cuda.init()
            return cuda.Device.count() > 0
        except ImportError:
            return False


def _tensorrt_available() -> bool:
    """Check if TensorRT with pycuda is available."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        return True
    except ImportError:
        return False


def _onnxruntime_cuda_available() -> bool:
    """Check if ONNX Runtime CUDA provider is available."""
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def _openvino_available() -> bool:
    """Check if OpenVINO is available."""
    try:
        from openvino import Core
        return True
    except ImportError:
        return False


def _snpe_available() -> bool:
    """Check if SNPE is available."""
    # Placeholder - would check for SNPE SDK
    return False
