"""
TensorRT runtime for iVIT-SDK.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    logger.debug("TensorRT not available")


class TensorRTRuntime:
    """TensorRT runtime for NVIDIA GPUs."""

    def __init__(self):
        if not HAS_TENSORRT:
            raise ImportError("TensorRT is not available")

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self._engines = {}

    @property
    def name(self) -> str:
        return "TensorRT"

    @property
    def backend_type(self) -> str:
        return "tensorrt"

    def is_available(self) -> bool:
        return HAS_TENSORRT

    def supported_formats(self) -> List[str]:
        return [".onnx", ".engine", ".trt", ".plan"]

    def get_devices(self) -> List[Dict[str, Any]]:
        """Get available CUDA devices."""
        devices = []
        try:
            device_count = cuda.Device.count()
            for i in range(device_count):
                device = cuda.Device(i)
                devices.append({
                    "id": f"cuda:{i}",
                    "name": device.name(),
                    "backend": "tensorrt",
                    "type": "gpu",
                    "memory_total": device.total_memory(),
                    "is_available": True,
                })
        except Exception as e:
            logger.warning(f"Failed to enumerate CUDA devices: {e}")
        return devices

    def load_model(
        self,
        path: str,
        device: str = "cuda:0",
        precision: str = "fp32",
        **kwargs
    ) -> "TRTModel":
        """
        Load model for TensorRT inference.

        Args:
            path: Model path (.onnx or .engine)
            device: Device ID
            precision: Precision mode (fp32, fp16, int8)

        Returns:
            TRTModel instance
        """
        import os

        ext = os.path.splitext(path)[1].lower()

        if ext == ".onnx":
            engine = self._build_engine_from_onnx(path, precision)
        elif ext in (".engine", ".trt", ".plan"):
            engine = self._load_engine(path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        return TRTModel(engine, self.runtime, device)

    def _build_engine_from_onnx(
        self,
        onnx_path: str,
        precision: str = "fp32"
    ) -> trt.ICudaEngine:
        """Build TensorRT engine from ONNX model."""
        builder = trt.Builder(self.logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX model
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX Parser Error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")

        # Build config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # Set precision
        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        return self.runtime.deserialize_cuda_engine(serialized_engine)

    def _load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """Load serialized TensorRT engine."""
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = self.runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        return engine


class TRTModel:
    """TensorRT model wrapper."""

    def __init__(
        self,
        engine: "trt.ICudaEngine",
        runtime: "trt.Runtime",
        device: str
    ):
        self.engine = engine
        self.runtime = runtime
        self.device = device

        # Create execution context
        self.context = engine.create_execution_context()

        # Create CUDA stream
        self.stream = cuda.Stream()

        # Allocate buffers
        self._allocate_buffers()

    def _allocate_buffers(self):
        """Allocate host and device buffers."""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.input_names = []
        self.output_names = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)

            # Handle dynamic shapes
            if -1 in shape:
                shape = tuple(max(1, s) for s in shape)

            size = int(np.prod(shape))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.inputs.append({
                    "name": name,
                    "host": host_mem,
                    "device": device_mem,
                    "shape": shape,
                    "dtype": dtype
                })
                self.input_names.append(name)
            else:
                self.outputs.append({
                    "name": name,
                    "host": host_mem,
                    "device": device_mem,
                    "shape": shape,
                    "dtype": dtype
                })
                self.output_names.append(name)

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference.

        Args:
            inputs: Dictionary of input name to numpy array

        Returns:
            Dictionary of output name to numpy array
        """
        # Copy inputs to device
        for inp in self.inputs:
            data = inputs.get(inp["name"])
            if data is None:
                raise ValueError(f"Missing input: {inp['name']}")

            np.copyto(inp["host"], data.ravel())
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Copy outputs to host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        # Synchronize
        self.stream.synchronize()

        # Return outputs
        outputs = {}
        for out in self.outputs:
            outputs[out["name"]] = out["host"].reshape(out["shape"]).copy()

        return outputs

    def get_input_info(self) -> List[Dict[str, Any]]:
        """Get input tensor information."""
        return [
            {
                "name": inp["name"],
                "shape": list(inp["shape"]),
                "dtype": str(inp["dtype"])
            }
            for inp in self.inputs
        ]

    def get_output_info(self) -> List[Dict[str, Any]]:
        """Get output tensor information."""
        return [
            {
                "name": out["name"],
                "shape": list(out["shape"]),
                "dtype": str(out["dtype"])
            }
            for out in self.outputs
        ]

    def __del__(self):
        """Clean up resources."""
        # Free device memory
        for inp in self.inputs:
            inp["device"].free()
        for out in self.outputs:
            out["device"].free()
