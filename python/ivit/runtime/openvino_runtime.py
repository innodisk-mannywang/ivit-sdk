"""
OpenVINO runtime for iVIT-SDK.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import OpenVINO
try:
    from openvino import Core, Type, Layout
    from openvino.preprocess import PrePostProcessor
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    logger.debug("OpenVINO not available")


class OpenVINORuntime:
    """OpenVINO runtime for Intel hardware."""

    def __init__(self):
        if not HAS_OPENVINO:
            raise ImportError("OpenVINO is not available")

        self.core = Core()
        self._models = {}

    @property
    def name(self) -> str:
        return "OpenVINO"

    @property
    def backend_type(self) -> str:
        return "openvino"

    def is_available(self) -> bool:
        return HAS_OPENVINO

    def supported_formats(self) -> List[str]:
        return [".onnx", ".xml", ".pdmodel"]

    def get_devices(self) -> List[Dict[str, Any]]:
        """Get available OpenVINO devices."""
        devices = []
        available = self.core.available_devices

        for device_name in available:
            try:
                full_name = self.core.get_property(device_name, "FULL_DEVICE_NAME")
            except Exception:
                full_name = device_name

            device_type = "unknown"
            device_id = device_name.lower()

            if "CPU" in device_name:
                device_type = "cpu"
                device_id = "cpu"
            elif "GPU" in device_name:
                device_type = "gpu"
                if "." in device_name:
                    idx = device_name.split(".")[-1]
                    device_id = f"gpu:{idx}"
                else:
                    device_id = "gpu:0"
            elif "NPU" in device_name:
                device_type = "npu"
                device_id = "npu"
            elif "MYRIAD" in device_name or "VPU" in device_name:
                device_type = "vpu"
                device_id = "vpu"

            devices.append({
                "id": device_id,
                "name": full_name,
                "backend": "openvino",
                "type": device_type,
                "is_available": True,
            })

        return devices

    def load_model(
        self,
        path: str,
        device: str = "CPU",
        precision: str = "fp32",
        **kwargs
    ) -> "OVModel":
        """
        Load model for OpenVINO inference.

        Args:
            path: Model path (.onnx or .xml)
            device: Device name (CPU, GPU, NPU, etc.)
            precision: Precision mode (fp32, fp16, int8)

        Returns:
            OVModel instance
        """
        # Map device string to OpenVINO device name
        ov_device = self._map_device(device)

        # Read model
        model = self.core.read_model(path)

        # Configure properties
        config = {}

        if precision == "fp16":
            config["INFERENCE_PRECISION_HINT"] = Type.f16
        elif precision == "int8":
            config["INFERENCE_PRECISION_HINT"] = Type.i8

        # Set performance hint
        config["PERFORMANCE_HINT"] = "LATENCY"

        # Compile model
        compiled_model = self.core.compile_model(model, ov_device, config)

        return OVModel(compiled_model, model, device)

    def _map_device(self, device: str) -> str:
        """Map iVIT device string to OpenVINO device name."""
        device_lower = device.lower()

        if device_lower in ("auto", ""):
            return "AUTO"
        if device_lower == "cpu":
            return "CPU"
        if device_lower.startswith("gpu"):
            if ":" in device_lower:
                idx = device_lower.split(":")[1]
                return f"GPU.{idx}"
            return "GPU"
        if device_lower == "npu":
            return "NPU"
        if device_lower in ("vpu", "myriad"):
            return "MYRIAD"

        return device.upper()


class OVModel:
    """OpenVINO model wrapper."""

    def __init__(
        self,
        compiled_model,
        model,
        device: str
    ):
        self.compiled_model = compiled_model
        self.model = model
        self.device = device

        # Create inference request
        self.infer_request = compiled_model.create_infer_request()

        # Get input/output info
        self._input_info = []
        self._output_info = []

        for inp in model.inputs:
            self._input_info.append({
                "name": inp.any_name,
                "shape": list(inp.partial_shape.get_min_shape()),
                "dtype": str(inp.element_type)
            })

        for out in model.outputs:
            self._output_info.append({
                "name": out.any_name,
                "shape": list(out.partial_shape.get_min_shape()),
                "dtype": str(out.element_type)
            })

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference.

        Args:
            inputs: Dictionary of input name to numpy array

        Returns:
            Dictionary of output name to numpy array
        """
        # Set inputs
        for name, data in inputs.items():
            self.infer_request.set_tensor(name, data)

        # Run inference
        self.infer_request.infer()

        # Get outputs
        outputs = {}
        for info in self._output_info:
            tensor = self.infer_request.get_tensor(info["name"])
            outputs[info["name"]] = tensor.data.copy()

        return outputs

    def infer_async(
        self,
        inputs: Dict[str, np.ndarray],
        callback=None
    ):
        """
        Run asynchronous inference.

        Args:
            inputs: Dictionary of input name to numpy array
            callback: Callback function called when inference is done
        """
        for name, data in inputs.items():
            self.infer_request.set_tensor(name, data)

        if callback:
            self.infer_request.set_callback(callback)

        self.infer_request.start_async()

    def wait(self):
        """Wait for async inference to complete."""
        self.infer_request.wait()

    def get_input_info(self) -> List[Dict[str, Any]]:
        """Get input tensor information."""
        return self._input_info

    def get_output_info(self) -> List[Dict[str, Any]]:
        """Get output tensor information."""
        return self._output_info
