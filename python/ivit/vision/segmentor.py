"""
Semantic segmentation module.
"""

from typing import Union, List, Optional, Dict, Tuple
import numpy as np
from pathlib import Path

from ..core.types import LoadConfig
from ..core.result import Results
from ..core.model import Model, load_model


class Segmentor:
    """
    Semantic segmentor.

    High-level API for semantic segmentation tasks.

    Examples:
        >>> segmentor = Segmentor("deeplabv3_resnet50")
        >>> results = segmentor.predict("scene.jpg")
        >>> colored_mask = results.colorize_mask()
    """

    def __init__(
        self,
        model: Union[str, Model],
        device: str = "auto",
        **kwargs
    ):
        """
        Create segmentor.

        Args:
            model: Model name or path
            device: Target device
        """
        if isinstance(model, str):
            config = LoadConfig(device=device, task="segmentation", **kwargs)
            self._model = load_model(model, **config.__dict__)
        else:
            self._model = model

        self._labels: List[str] = self._model.labels
        self._colormap: Dict[int, Tuple[int, int, int]] = {}
        self._init_default_colormap()

    def predict(self, image: Union[str, np.ndarray]) -> Results:
        """
        Segment image.

        Args:
            image: Input image (path or array)

        Returns:
            Segmentation results with mask
        """
        import cv2

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
        import time
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
        results.image_size = (orig_h, orig_w)
        results._original_image = img

        return results

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]]
    ) -> List[Results]:
        """Batch segmentation."""
        return [self.predict(img) for img in images]

    @property
    def classes(self) -> List[str]:
        """Get class labels."""
        return self._labels

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self._labels)

    @property
    def input_size(self) -> tuple:
        """Get input size."""
        input_info = self._model.input_info[0]
        if len(input_info.shape) >= 4:
            return (input_info.shape[2], input_info.shape[3])
        return (512, 512)

    @property
    def colormap(self) -> Dict[int, Tuple[int, int, int]]:
        """Get colormap."""
        return self._colormap

    def set_colormap(self, colormap: Dict[int, Tuple[int, int, int]]) -> None:
        """Set custom colormap."""
        self._colormap = colormap

    @property
    def model(self) -> Model:
        """Get underlying model."""
        return self._model

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for segmentation."""
        import cv2

        h, w = self.input_size

        # Resize
        resized = cv2.resize(image, (w, h))

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
        results.raw_outputs = outputs

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
