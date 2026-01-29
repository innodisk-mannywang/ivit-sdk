"""
ONNX Runtime backend.

.. deprecated:: 1.0.0
    Pure Python ONNX Runtime is deprecated. All inference now uses C++ bindings.
"""

import warnings

warnings.warn(
    "ivit.runtime.onnx_runtime is deprecated. Use C++ bindings instead.",
    DeprecationWarning,
    stacklevel=2,
)

from typing import Dict, List, Any, Optional
import numpy as np
import logging

from .base import BaseRuntime, BaseModel

logger = logging.getLogger(__name__)


class ONNXRuntime(BaseRuntime):
    """
    ONNX Runtime backend.

    This serves as the fallback backend when specialized backends
    (OpenVINO, TensorRT, SNPE) are not available.
    """

    def __init__(self):
        try:
            import onnxruntime as ort
            self._ort = ort

            # Get available providers
            self._providers = ort.get_available_providers()
            logger.info(f"ONNX Runtime providers: {self._providers}")

        except ImportError:
            raise ImportError("ONNX Runtime not installed. Install with: pip install onnxruntime")

    @property
    def name(self) -> str:
        return "onnxruntime"

    @property
    def is_available(self) -> bool:
        return self._ort is not None

    def get_version(self) -> str:
        """Get ONNX Runtime version."""
        return self._ort.__version__ if self._ort else "unknown"

    def get_supported_devices(self) -> List[str]:
        """Get list of supported devices based on available providers."""
        devices = ["cpu"]
        if "CUDAExecutionProvider" in self._providers:
            devices.append("cuda")
        if "TensorrtExecutionProvider" in self._providers:
            devices.append("tensorrt")
        if "OpenVINOExecutionProvider" in self._providers:
            devices.append("openvino")
        if "DmlExecutionProvider" in self._providers:
            devices.append("directml")
        return devices

    def load_model(
        self,
        path: str,
        device: str = "cpu",
        precision: str = "fp32",
        **kwargs
    ) -> "ORTModel":
        """
        Load ONNX model.

        Args:
            path: Model path
            device: Target device
            precision: Precision mode (fp32, fp16)

        Returns:
            ORTModel instance
        """
        # Select providers based on device
        providers = self._select_providers(device)

        # Create session options
        sess_options = self._ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True

        # Suppress Memcpy warning (level 3 = ERROR, suppresses WARNING)
        # This avoids the "Memcpy nodes are added to the graph" warning
        sess_options.log_severity_level = 3

        # Set number of threads
        if hasattr(sess_options, 'intra_op_num_threads'):
            import os
            num_threads = os.cpu_count() or 4
            sess_options.intra_op_num_threads = num_threads

        # Create session
        logger.info(f"Loading ONNX model: {path}")
        logger.info(f"Providers: {providers}")

        session = self._ort.InferenceSession(
            path,
            sess_options=sess_options,
            providers=providers
        )

        return ORTModel(session, device)

    def _select_providers(self, device: str) -> List[str]:
        """Select execution providers based on device."""
        device = device.lower()

        if device.startswith("cuda") or device == "gpu":
            if "CUDAExecutionProvider" in self._providers:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        if device == "tensorrt":
            if "TensorrtExecutionProvider" in self._providers:
                return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

        if device in ("dml", "directml"):
            if "DmlExecutionProvider" in self._providers:
                return ["DmlExecutionProvider", "CPUExecutionProvider"]

        if device == "openvino":
            if "OpenVINOExecutionProvider" in self._providers:
                return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

        # Default to CPU
        return ["CPUExecutionProvider"]

    @staticmethod
    def _map_dtype(onnx_type: str) -> str:
        """Map ONNX type string to dtype."""
        type_map = {
            "tensor(float)": "float32",
            "tensor(float16)": "float16",
            "tensor(int8)": "int8",
            "tensor(uint8)": "uint8",
            "tensor(int32)": "int32",
            "tensor(int64)": "int64",
            "tensor(bool)": "bool",
        }
        return type_map.get(onnx_type, "float32")


class ORTModel(BaseModel):
    """ONNX Runtime model wrapper."""

    def __init__(self, session, device: str):
        self.session = session
        self.device = device

        # Cache input/output info
        self._input_info = []
        self._output_info = []

        for inp in session.get_inputs():
            shape = list(inp.shape) if inp.shape else []
            # Handle dynamic dimensions
            shape = [s if isinstance(s, int) else 1 for s in shape]
            self._input_info.append({
                "name": inp.name,
                "shape": shape,
                "dtype": ONNXRuntime._map_dtype(inp.type)
            })

        for out in session.get_outputs():
            shape = list(out.shape) if out.shape else []
            shape = [s if isinstance(s, int) else 1 for s in shape]
            self._output_info.append({
                "name": out.name,
                "shape": shape,
                "dtype": ONNXRuntime._map_dtype(out.type)
            })

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference.

        Args:
            inputs: Dictionary of input name to numpy array

        Returns:
            Dictionary of output name to numpy array
        """
        # Get output names
        output_names = [o["name"] for o in self._output_info]

        # Get input names
        input_names = [i["name"] for i in self._input_info]

        # Map inputs to correct names
        feed_dict = {}
        if len(input_names) == 1 and len(inputs) == 1:
            # Single input - use correct name
            feed_dict[input_names[0]] = list(inputs.values())[0]
        else:
            feed_dict = inputs

        # Run inference
        outputs = self.session.run(output_names, feed_dict)

        # Create output dict
        return {name: output for name, output in zip(output_names, outputs)}

    def get_input_info(self) -> List[Dict[str, Any]]:
        """Get input tensor information."""
        return self._input_info

    def get_output_info(self) -> List[Dict[str, Any]]:
        """Get output tensor information."""
        return self._output_info
