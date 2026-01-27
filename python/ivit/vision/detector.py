"""
Object detection module.

Provides a high-level API for object detection tasks.
Automatically uses C++ implementation when available.
"""

from typing import Union, List, Optional, Callable, Tuple
import numpy as np
from pathlib import Path

# Check if C++ bindings are available
try:
    from .._ivit_core import (
        Detector as _CppDetector,
        LoadConfig as _CppLoadConfig,
        InferConfig as _CppInferConfig,
    )
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
    _CppDetector = None

from ..core.types import LoadConfig, InferConfig, Detection, BBox
from ..core.result import Results


class Detector:
    """
    Object detector.

    High-level API for object detection tasks.
    Supports YOLO, SSD, and other detection models.

    This class automatically uses C++ bindings when available
    for optimal performance.

    Examples:
        >>> detector = Detector("yolov8n.onnx", device="cuda:0")
        >>> results = detector.predict("image.jpg")
        >>> for det in results.detections:
        ...     print(f"{det.label}: {det.confidence:.2%}")
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        **kwargs
    ):
        """
        Create detector.

        Args:
            model: Model path (.onnx, .engine, .xml)
            device: Target device ("auto", "cpu", "cuda:0", etc.)
            **kwargs: Additional configuration options
        """
        self._model_path = model
        self._device = device

        if _HAS_CPP and _CppDetector is not None:
            # Use C++ implementation
            config = _CppLoadConfig()
            config.device = device
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            self._cpp_detector = _CppDetector(model, device, config)
            self._use_cpp = True
        else:
            # Fall back to pure Python
            from ..core.model import load_model
            config = LoadConfig(device=device, task="detection", **kwargs)
            self._model = load_model(model, **config.__dict__)
            self._use_cpp = False

    def predict(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
        max_detections: int = 300,
    ) -> Results:
        """
        Detect objects in image.

        Args:
            image: Input image (file path or numpy array)
            conf_threshold: Confidence threshold (0.0 - 1.0)
            iou_threshold: NMS IoU threshold (0.0 - 1.0)
            classes: Filter by class IDs (None = all classes)
            max_detections: Maximum number of detections to return

        Returns:
            Results object containing detections
        """
        if self._use_cpp:
            # Use C++ implementation
            if isinstance(image, str):
                # Load image
                import cv2
                img = cv2.imread(image)
                if img is None:
                    raise ValueError(f"Failed to load image: {image}")
            else:
                img = image

            # Call C++ detector
            cpp_results = self._cpp_detector.predict(
                img,
                conf_threshold,
                iou_threshold
            )

            # Convert C++ results to Python Results
            return self._convert_cpp_results(cpp_results, image)
        else:
            # Pure Python implementation
            config = InferConfig(
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                classes=classes,
                max_detections=max_detections,
            )
            return self._model.predict(image, **config.__dict__)

    def __call__(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        **kwargs
    ) -> Results:
        """Shorthand for predict()."""
        return self.predict(
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            **kwargs
        )

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        **kwargs
    ) -> List[Results]:
        """
        Batch detection on multiple images.

        Args:
            images: List of images (paths or arrays)
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            **kwargs: Additional options

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

            # Create config
            config = _CppInferConfig()
            config.conf_threshold = conf_threshold
            config.iou_threshold = iou_threshold

            # Call C++ batch predict
            cpp_results_list = self._cpp_detector.predict_batch(np_images, config)

            # Convert results
            results_list = []
            for i, cpp_results in enumerate(cpp_results_list):
                results_list.append(self._convert_cpp_results(cpp_results, images[i]))
            return results_list
        else:
            return [
                self.predict(img, conf_threshold, iou_threshold, **kwargs)
                for img in images
            ]

    def predict_video(
        self,
        source: Union[str, int],
        callback: Callable[[Results, np.ndarray], None],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        **kwargs
    ) -> None:
        """
        Detect objects in video stream.

        Args:
            source: Video file path or camera ID (0 for default camera)
            callback: Function called for each frame: callback(results, frame)
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
        """
        import cv2

        cap = cv2.VideoCapture(source)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.predict(
                    frame,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    **kwargs
                )

                callback(results, frame)

        finally:
            cap.release()

    @property
    def classes(self) -> List[str]:
        """Get class labels."""
        if self._use_cpp:
            return list(self._cpp_detector.classes)
        return self._model.labels

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        if self._use_cpp:
            return self._cpp_detector.num_classes
        return len(self._model.labels)

    @property
    def input_size(self) -> Tuple[int, int]:
        """Get input size (width, height)."""
        if self._use_cpp:
            return self._cpp_detector.input_size
        input_info = self._model.input_info[0]
        shape = input_info.get('shape', input_info.get('dims', [1, 3, 640, 640]))
        if len(shape) >= 4:
            return (int(shape[3]), int(shape[2]))
        return (640, 640)

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
        # The C++ Results object has similar structure
        # We need to create a Python Results and copy data

        results = Results()

        # Copy detections
        results.detections = []
        for det in cpp_results.detections:
            py_det = Detection(
                bbox=BBox(det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2),
                class_id=det.class_id,
                label=det.label,
                confidence=det.confidence
            )
            results.detections.append(py_det)

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
