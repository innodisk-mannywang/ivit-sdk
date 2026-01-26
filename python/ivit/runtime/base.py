"""
Abstract base classes for runtime backends.

Provides a consistent interface for all runtime implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for model wrappers.

    All runtime-specific model classes should inherit from this class
    to ensure a consistent interface.
    """

    @abstractmethod
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference.

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        pass

    @abstractmethod
    def get_input_info(self) -> List[Dict[str, Any]]:
        """
        Get input tensor information.

        Returns:
            List of dicts with keys: name, shape, dtype
        """
        pass

    @abstractmethod
    def get_output_info(self) -> List[Dict[str, Any]]:
        """
        Get output tensor information.

        Returns:
            List of dicts with keys: name, shape, dtype
        """
        pass

    def apply_config(self, config: Dict[str, Any]) -> None:
        """
        Apply runtime-specific configuration.

        Args:
            config: Configuration dictionary

        Note:
            Default implementation does nothing. Override in subclass
            if configuration is supported.
        """
        pass

    def warmup(self, iterations: int = 3) -> None:
        """
        Warmup model by running dummy inferences.

        Args:
            iterations: Number of warmup iterations
        """
        input_info = self.get_input_info()
        if not input_info:
            return

        # Create dummy input
        inputs = {}
        for info in input_info:
            shape = info["shape"]
            # Handle dynamic dimensions
            shape = [s if s > 0 else 1 for s in shape]
            dtype = info.get("dtype", "float32")
            inputs[info["name"]] = np.random.randn(*shape).astype(dtype)

        # Run warmup
        for _ in range(iterations):
            self.infer(inputs)


class BaseRuntime(ABC):
    """
    Abstract base class for runtime backends.

    All runtime implementations (OpenVINO, TensorRT, SNPE, ONNXRuntime)
    should inherit from this class to ensure a consistent interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get runtime name.

        Returns:
            Runtime name (lowercase): "onnxruntime", "openvino", "tensorrt", "snpe"
        """
        pass

    @property
    def is_available(self) -> bool:
        """
        Check if runtime is available.

        Returns:
            True if runtime dependencies are installed and usable
        """
        return True

    @abstractmethod
    def load_model(
        self,
        path: str,
        device: str = "cpu",
        precision: str = "fp32",
        **kwargs
    ) -> BaseModel:
        """
        Load a model.

        Args:
            path: Path to model file
            device: Target device (e.g., "cpu", "cuda:0", "npu")
            precision: Inference precision ("fp32", "fp16", "int8")
            **kwargs: Additional runtime-specific options

        Returns:
            BaseModel instance

        Raises:
            FileNotFoundError: If model file not found
            ValueError: If model format not supported
        """
        pass

    def get_supported_devices(self) -> List[str]:
        """
        Get list of supported devices.

        Returns:
            List of device identifiers
        """
        return ["cpu"]

    def get_version(self) -> str:
        """
        Get runtime version.

        Returns:
            Version string
        """
        return "unknown"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.get_version()}')"
