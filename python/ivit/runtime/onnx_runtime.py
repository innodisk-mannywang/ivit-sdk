"""
ONNX Runtime backend.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import logging

from ..core.types import TensorInfo, LoadConfig

logger = logging.getLogger(__name__)


class ONNXRuntime:
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

    def load(self, path: str, config: LoadConfig) -> Any:
        """
        Load ONNX model.

        Args:
            path: Model path
            config: Load configuration

        Returns:
            ONNX session
        """
        # Select providers based on device
        providers = self._select_providers(config.device)

        # Create session options
        sess_options = self._ort.SessionOptions()

        if config.use_cache:
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True

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

        return session

    def get_input_info(self, session: Any) -> List[TensorInfo]:
        """Get input tensor information."""
        inputs = []
        for inp in session.get_inputs():
            dtype = self._map_dtype(inp.type)
            inputs.append(TensorInfo(
                name=inp.name,
                shape=tuple(inp.shape) if inp.shape else (),
                dtype=dtype,
                layout="NCHW",
            ))
        return inputs

    def get_output_info(self, session: Any) -> List[TensorInfo]:
        """Get output tensor information."""
        outputs = []
        for out in session.get_outputs():
            dtype = self._map_dtype(out.type)
            outputs.append(TensorInfo(
                name=out.name,
                shape=tuple(out.shape) if out.shape else (),
                dtype=dtype,
                layout="NCHW",
            ))
        return outputs

    def infer(
        self,
        session: Any,
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Run inference.

        Args:
            session: ONNX session
            inputs: Input tensors

        Returns:
            Output tensors
        """
        # Get output names
        output_names = [o.name for o in session.get_outputs()]

        # Get input name (handle single input case)
        input_names = [i.name for i in session.get_inputs()]

        # Map inputs to correct names
        feed_dict = {}
        if len(input_names) == 1 and len(inputs) == 1:
            # Single input - use correct name
            feed_dict[input_names[0]] = list(inputs.values())[0]
        else:
            feed_dict = inputs

        # Run inference
        outputs = session.run(output_names, feed_dict)

        # Create output dict
        return {name: output for name, output in zip(output_names, outputs)}

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
