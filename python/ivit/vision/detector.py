"""
Object detection module.
"""

from typing import Union, List, Optional, Callable
import numpy as np
from pathlib import Path

from ..core.types import LoadConfig, InferConfig, Detection, BBox
from ..core.result import Results
from ..core.model import Model, load_model


class Detector:
    """
    Object detector.

    High-level API for object detection tasks.
    Supports YOLO, SSD, and other detection models.

    Examples:
        >>> detector = Detector("yolov8n", device="cuda:0")
        >>> results = detector.predict("street.jpg")
        >>> for det in results.detections:
        ...     print(f"{det.label}: {det.confidence:.2%}")
    """

    def __init__(
        self,
        model: Union[str, Model],
        device: str = "auto",
        **kwargs
    ):
        """
        Create detector.

        Args:
            model: Model name or path
            device: Target device
        """
        if isinstance(model, str):
            config = LoadConfig(device=device, task="detection", **kwargs)
            self._model = load_model(model, **config.__dict__)
        else:
            self._model = model

        self._labels: List[str] = self._model.labels
        self._model_type = self._detect_model_type(model if isinstance(model, str) else model.name)

    def predict(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
        max_detections: int = 100,
    ) -> Results:
        """
        Detect objects in image.

        Args:
            image: Input image (path or array)
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            classes: Filter by class IDs (None = all)
            max_detections: Maximum detections

        Returns:
            Detection results
        """
        config = InferConfig(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            classes=classes,
            max_detections=max_detections,
        )

        return self._model.predict(image, **config.__dict__)

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        **kwargs
    ) -> List[Results]:
        """Batch detection."""
        return [self.predict(img, **kwargs) for img in images]

    def predict_video(
        self,
        source: Union[str, int],
        callback: Callable[[Results, np.ndarray], None],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        **kwargs
    ) -> None:
        """
        Detect objects in video stream.

        Args:
            source: Video path or camera ID
            callback: Callback function(results, frame)
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
        return (640, 640)

    @property
    def model(self) -> Model:
        """Get underlying model."""
        return self._model

    @staticmethod
    def _detect_model_type(model_name: str) -> str:
        """Detect model type from name."""
        name_lower = model_name.lower()

        if "yolov8" in name_lower or "yolov9" in name_lower or "yolov10" in name_lower:
            return "yolov8"
        elif "yolov5" in name_lower or "yolov7" in name_lower:
            return "yolov5"
        elif "ssd" in name_lower:
            return "ssd"
        elif "rcnn" in name_lower:
            return "rcnn"
        elif "detr" in name_lower:
            return "detr"
        else:
            return "yolov8"  # Default
