"""
Image classification module.

Provides a high-level API for image classification tasks.
Automatically uses C++ implementation when available.
"""

from typing import Union, List, Optional, Tuple
import numpy as np
from pathlib import Path

# Check if C++ bindings are available
try:
    from .._ivit_core import (
        Classifier as _CppClassifier,
        LoadConfig as _CppLoadConfig,
    )
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
    _CppClassifier = None

from ..core.types import LoadConfig, ClassificationResult
from ..core.result import Results


class Classifier:
    """
    Image classifier.

    High-level API for image classification tasks.
    Automatically uses C++ bindings when available for optimal performance.

    Examples:
        >>> classifier = Classifier("resnet50.onnx", device="cuda:0")
        >>> results = classifier.predict("cat.jpg")
        >>> print(f"Top-1: {results.top1.label} ({results.top1.score:.2%})")
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        **kwargs
    ):
        """
        Create classifier.

        Args:
            model: Model path (.onnx, .engine, .xml)
            device: Target device ("auto", "cpu", "cuda:0", etc.)
            **kwargs: Additional configuration options
        """
        self._model_path = model
        self._device = device

        if _HAS_CPP and _CppClassifier is not None:
            # Use C++ implementation
            config = _CppLoadConfig()
            config.device = device
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            self._cpp_classifier = _CppClassifier(model, device, config)
            self._use_cpp = True
        else:
            # Fall back to pure Python
            from ..core.model import load_model
            config = LoadConfig(device=device, task="classification", **kwargs)
            self._model = load_model(model, **config.__dict__)
            self._labels = self._model.labels
            self._use_cpp = False

    def predict(
        self,
        image: Union[str, np.ndarray],
        top_k: int = 5
    ) -> Results:
        """
        Classify image.

        Args:
            image: Input image (file path or numpy array)
            top_k: Number of top results to return

        Returns:
            Results object containing classifications
        """
        if self._use_cpp:
            # Use C++ implementation
            if isinstance(image, str):
                # C++ can handle file path directly
                cpp_results = self._cpp_classifier.predict(image, top_k)
            else:
                cpp_results = self._cpp_classifier.predict(image, top_k)

            return self._convert_cpp_results(cpp_results, image)
        else:
            # Pure Python implementation
            return self._predict_python(image, top_k)

    def __call__(
        self,
        image: Union[str, np.ndarray],
        top_k: int = 5
    ) -> Results:
        """Shorthand for predict()."""
        return self.predict(image, top_k)

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        top_k: int = 5
    ) -> List[Results]:
        """
        Batch classification on multiple images.

        Args:
            images: List of images (paths or arrays)
            top_k: Number of top results per image

        Returns:
            List of Results objects
        """
        if self._use_cpp:
            # Load all images as numpy arrays
            import cv2
            np_images = []
            for img in images:
                if isinstance(img, str):
                    np_img = cv2.imread(img)
                    if np_img is None:
                        raise ValueError(f"Failed to load image: {img}")
                    np_images.append(np_img)
                else:
                    np_images.append(img)

            # Call C++ batch predict
            cpp_results_list = self._cpp_classifier.predict_batch(np_images, top_k)

            # Convert results
            results_list = []
            for i, cpp_results in enumerate(cpp_results_list):
                results_list.append(self._convert_cpp_results(cpp_results, images[i]))
            return results_list
        else:
            return [self.predict(img, top_k) for img in images]

    @property
    def classes(self) -> List[str]:
        """Get class labels."""
        if self._use_cpp:
            return list(self._cpp_classifier.classes)
        return self._labels

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        if self._use_cpp:
            return self._cpp_classifier.num_classes
        return len(self._labels)

    @property
    def input_size(self) -> Tuple[int, int]:
        """Get input size (width, height)."""
        if self._use_cpp:
            return self._cpp_classifier.input_size
        input_info = self._model.input_info[0]
        shape = input_info.get('shape', input_info.get('dims', [1, 3, 224, 224]))
        if len(shape) >= 4:
            return (int(shape[3]), int(shape[2]))
        return (224, 224)

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

    def _convert_cpp_results(self, cpp_results, original_image) -> Results:
        """Convert C++ Results to Python Results."""
        results = Results()

        # Copy classifications
        results.classifications = []
        for cls in cpp_results.classifications:
            py_cls = ClassificationResult(
                class_id=cls.class_id,
                label=cls.label,
                score=cls.score
            )
            results.classifications.append(py_cls)

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

        return results

    def _predict_python(self, image: Union[str, np.ndarray], top_k: int) -> Results:
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
        results = self._postprocess(outputs, top_k)
        results.inference_time_ms = inference_time
        results.device_used = self._model.device
        results.image_size = (img.shape[1], img.shape[0])

        return results

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for classification."""
        import cv2

        w, h = self.input_size

        # Resize
        resized = cv2.resize(image, (w, h))

        # BGR to RGB
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

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

    def _postprocess(self, outputs: dict, top_k: int) -> Results:
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

        return results

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax function."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
