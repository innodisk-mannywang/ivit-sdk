"""
Semantic segmentation module.

Provides a high-level API for semantic segmentation tasks.
Automatically uses C++ implementation when available.
"""

from typing import Union, List, Optional, Dict, Tuple
import numpy as np
from pathlib import Path

# Check if C++ bindings are available
try:
    from .._ivit_core import (
        Segmentor as _CppSegmentor,
        LoadConfig as _CppLoadConfig,
    )
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
    _CppSegmentor = None

from ..core.types import LoadConfig
from ..core.result import Results


class Segmentor:
    """
    Semantic segmentor.

    High-level API for semantic segmentation tasks.
    Automatically uses C++ bindings when available for optimal performance.

    Examples:
        >>> segmentor = Segmentor("deeplabv3.onnx", device="cuda:0")
        >>> results = segmentor.predict("scene.jpg")
        >>> colored_mask = results.colorize_mask()
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        **kwargs
    ):
        """
        Create segmentor.

        Args:
            model: Model path (.onnx, .engine, .xml)
            device: Target device ("auto", "cpu", "cuda:0", etc.)
            **kwargs: Additional configuration options
        """
        self._model_path = model
        self._device = device
        self._colormap: Dict[int, Tuple[int, int, int]] = {}
        self._init_default_colormap()

        if _HAS_CPP and _CppSegmentor is not None:
            # Use C++ implementation
            config = _CppLoadConfig()
            config.device = device
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            self._cpp_segmentor = _CppSegmentor(model, device, config)
            self._use_cpp = True
        else:
            # Fall back to pure Python
            from ..core.model import load_model
            config = LoadConfig(device=device, task="segmentation", **kwargs)
            self._model = load_model(model, **config.__dict__)
            self._labels = self._model.labels
            self._use_cpp = False

    def predict(self, image: Union[str, np.ndarray]) -> Results:
        """
        Segment image.

        Args:
            image: Input image (file path or numpy array)

        Returns:
            Results object containing segmentation mask
        """
        if self._use_cpp:
            # Use C++ implementation
            if isinstance(image, str):
                cpp_results = self._cpp_segmentor.predict(image)
            else:
                cpp_results = self._cpp_segmentor.predict(image)

            return self._convert_cpp_results(cpp_results, image)
        else:
            return self._predict_python(image)

    def __call__(self, image: Union[str, np.ndarray]) -> Results:
        """Shorthand for predict()."""
        return self.predict(image)

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]]
    ) -> List[Results]:
        """
        Batch segmentation on multiple images.

        Args:
            images: List of images (paths or arrays)

        Returns:
            List of Results objects
        """
        return [self.predict(img) for img in images]

    @property
    def classes(self) -> List[str]:
        """Get class labels."""
        if self._use_cpp:
            return list(self._cpp_segmentor.classes)
        return self._labels

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        if self._use_cpp:
            return self._cpp_segmentor.num_classes
        return len(self._labels)

    @property
    def input_size(self) -> Tuple[int, int]:
        """Get input size (width, height)."""
        if self._use_cpp:
            return self._cpp_segmentor.input_size
        input_info = self._model.input_info[0]
        shape = input_info.get('shape', input_info.get('dims', [1, 3, 512, 512]))
        if len(shape) >= 4:
            return (int(shape[3]), int(shape[2]))
        return (512, 512)

    @property
    def device(self) -> str:
        """Get device being used."""
        return self._device

    @property
    def backend(self) -> str:
        """Get backend being used."""
        if self._use_cpp:
            return "C++"
        return "Python"

    @property
    def colormap(self) -> Dict[int, Tuple[int, int, int]]:
        """Get colormap for visualization."""
        return self._colormap

    def set_colormap(self, colormap: Dict[int, Tuple[int, int, int]]) -> None:
        """Set custom colormap for visualization."""
        self._colormap = colormap

    def _convert_cpp_results(self, cpp_results, original_image) -> Results:
        """Convert C++ Results to Python Results."""
        results = Results()

        # Copy segmentation mask
        results.segmentation_mask = np.array(cpp_results.segmentation_mask)

        # Copy metadata
        results.inference_time_ms = cpp_results.inference_time_ms
        results.device_used = cpp_results.device_used

        # Set image size
        if isinstance(original_image, str):
            import cv2
            img = cv2.imread(original_image)
            if img is not None:
                results.image_size = (img.shape[1], img.shape[0])
        else:
            results.image_size = (original_image.shape[1], original_image.shape[0])

        # Store colormap for visualization
        results._colormap = self._colormap

        return results

    def _predict_python(self, image: Union[str, np.ndarray]) -> Results:
        """Pure Python prediction implementation."""
        import cv2
        import time

        # Load image if path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
        else:
            img = image

        orig_h, orig_w = img.shape[:2]

        # Preprocess
        input_tensor = self._preprocess(img)

        # Inference
        start = time.perf_counter()
        outputs = self._model._runtime.infer(
            self._model._handle,
            {"input": input_tensor}
        )
        inference_time = (time.perf_counter() - start) * 1000

        # Postprocess
        results = self._postprocess(outputs, (orig_h, orig_w))
        results.inference_time_ms = inference_time
        results.device_used = self._model.device
        results.image_size = (orig_w, orig_h)
        results._colormap = self._colormap

        return results

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for segmentation."""
        import cv2

        w, h = self.input_size

        # Resize
        resized = cv2.resize(image, (w, h))

        # BGR to RGB
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - mean) / std

        # HWC -> CHW
        tensor = tensor.transpose(2, 0, 1)

        # Add batch dimension
        tensor = np.expand_dims(tensor, 0)

        return tensor.astype(np.float32)

    def _postprocess(
        self,
        outputs: dict,
        orig_size: Tuple[int, int]
    ) -> Results:
        """Postprocess segmentation outputs."""
        import cv2

        results = Results()

        # Get output
        output = list(outputs.values())[0]

        # Handle different output formats
        if len(output.shape) == 4:
            # (N, C, H, W) - take argmax over channels
            output = output[0]  # Remove batch
            if output.shape[0] > 1:  # Multiple channels
                mask = np.argmax(output, axis=0)
            else:
                mask = output[0]
        elif len(output.shape) == 3:
            if output.shape[0] > 1:
                mask = np.argmax(output, axis=0)
            else:
                mask = output[0]
        else:
            mask = output

        # Resize to original size
        mask = cv2.resize(
            mask.astype(np.float32),
            (orig_size[1], orig_size[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)

        results.segmentation_mask = mask

        return results

    def _init_default_colormap(self) -> None:
        """Initialize default colormap."""
        import colorsys

        # Generate distinct colors for up to 256 classes
        for i in range(256):
            hue = i / 256.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
            self._colormap[i] = tuple(int(c * 255) for c in rgb)

        # Override with standard colors for common classes
        # (using Pascal VOC colormap style)
        self._colormap[0] = (0, 0, 0)  # Background
        self._colormap[1] = (128, 0, 0)  # Class 1
        self._colormap[2] = (0, 128, 0)  # Class 2
        self._colormap[3] = (128, 128, 0)  # Class 3
        self._colormap[4] = (0, 0, 128)  # Class 4
        self._colormap[5] = (128, 0, 128)  # Class 5
