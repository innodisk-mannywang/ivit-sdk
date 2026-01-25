"""
Image classification module.
"""

from typing import Union, List, Optional
import numpy as np
from pathlib import Path

from ..core.types import LoadConfig, ClassificationResult
from ..core.result import Results
from ..core.model import Model, load_model


class Classifier:
    """
    Image classifier.

    High-level API for image classification tasks.

    Examples:
        >>> classifier = Classifier("efficientnet_b0")
        >>> results = classifier.predict("cat.jpg")
        >>> print(f"Top-1: {results.top1.label} ({results.top1.score:.2%})")
    """

    def __init__(
        self,
        model: Union[str, Model],
        device: str = "auto",
        **kwargs
    ):
        """
        Create classifier.

        Args:
            model: Model name (from Model Zoo) or path to model file
            device: Target device
        """
        if isinstance(model, str):
            config = LoadConfig(device=device, task="classification", **kwargs)
            self._model = load_model(model, **config.__dict__)
        else:
            self._model = model

        self._labels: List[str] = self._model.labels

    def predict(
        self,
        image: Union[str, np.ndarray],
        top_k: int = 5
    ) -> Results:
        """
        Classify image.

        Args:
            image: Input image (path or array)
            top_k: Return top K results

        Returns:
            Classification results
        """
        import cv2

        # Load image if path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
        else:
            img = image

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
        results = self._postprocess(outputs, top_k)
        results.inference_time_ms = inference_time
        results.device_used = self._model.device
        results.image_size = img.shape[:2]
        results._original_image = img

        return results

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        top_k: int = 5
    ) -> List[Results]:
        """Batch classification."""
        return [self.predict(img, top_k) for img in images]

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
        return (224, 224)

    @property
    def model(self) -> Model:
        """Get underlying model."""
        return self._model

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for classification."""
        import cv2

        h, w = self.input_size

        # Resize
        resized = cv2.resize(image, (w, h))

        # Normalize (ImageNet mean/std)
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
        top_k: int
    ) -> Results:
        """Postprocess classification outputs."""
        results = Results()

        # Get output
        output = list(outputs.values())[0]

        # Softmax if needed
        if output.max() > 1.0 or output.min() < 0.0:
            output = self._softmax(output)

        # Get top-k
        output = output.flatten()
        top_indices = np.argsort(output)[::-1][:top_k]

        for idx in top_indices:
            label = self._labels[idx] if idx < len(self._labels) else str(idx)
            results.classifications.append(ClassificationResult(
                class_id=int(idx),
                label=label,
                score=float(output[idx]),
            ))

        results.raw_outputs = outputs
        return results

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax function."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
