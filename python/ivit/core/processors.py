"""
Pre-processing and post-processing classes for iVIT-SDK.

Extracted from Model class to improve modularity and testability.
"""

from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
import logging

from .types import Detection, BBox, ClassificationResult, InferConfig
from .result import Results

logger = logging.getLogger(__name__)


class BasePreProcessor(ABC):
    """Abstract base class for pre-processors."""

    @abstractmethod
    def process(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Pre-process image for inference.

        Args:
            image: Input image (BGR, HWC format)
            target_size: Target size (height, width)
            **kwargs: Additional options

        Returns:
            Tuple of (processed_tensor, preprocess_info)
        """
        pass

    def __call__(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (640, 640),
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Callable interface for pre-processing.

        Args:
            image: Input image (BGR, HWC format)
            target_size: Target size (height, width), default (640, 640)
            **kwargs: Additional options

        Returns:
            Tuple of (processed_tensor, preprocess_info)
        """
        return self.process(image, target_size, **kwargs)


class BasePostProcessor(ABC):
    """Abstract base class for post-processors."""

    @abstractmethod
    def process(
        self,
        outputs: Dict[str, np.ndarray],
        orig_size: Tuple[int, int],
        preprocess_info: Dict[str, Any],
        config: InferConfig,
        labels: List[str] = None,
    ) -> Results:
        """
        Post-process model outputs.

        Args:
            outputs: Raw model outputs
            orig_size: Original image size (height, width)
            preprocess_info: Info from preprocessing
            config: Inference configuration
            labels: Class labels

        Returns:
            Results object
        """
        pass

    def __call__(
        self,
        outputs: Dict[str, np.ndarray],
        orig_size: Tuple[int, int],
        preprocess_info: Dict[str, Any] = None,
        config: InferConfig = None,
        labels: List[str] = None,
    ) -> Results:
        """
        Callable interface for post-processing.

        Args:
            outputs: Raw model outputs
            orig_size: Original image size (height, width)
            preprocess_info: Info from preprocessing (default: empty dict)
            config: Inference configuration (default: InferConfig())
            labels: Class labels

        Returns:
            Results object
        """
        if preprocess_info is None:
            preprocess_info = {}
        if config is None:
            config = InferConfig()
        return self.process(outputs, orig_size, preprocess_info, config, labels)


class LetterboxPreProcessor(BasePreProcessor):
    """
    Letterbox pre-processor for object detection models.

    Resizes image while maintaining aspect ratio, padding with gray.
    """

    def __init__(self, pad_value: int = 114):
        """
        Args:
            pad_value: Padding value (default: 114, gray)
        """
        self.pad_value = pad_value

    def process(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply letterbox preprocessing.

        Args:
            image: Input image (BGR, HWC)
            target_size: Target size (height, width)

        Returns:
            (tensor, preprocess_info)
        """
        import cv2

        h, w = target_size
        orig_h, orig_w = image.shape[:2]

        # Calculate scale
        scale = min(w / orig_w, h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_w, pad_h = (w - new_w) // 2, (h - new_h) // 2

        # Resize
        resized = cv2.resize(image, (new_w, new_h))

        # Create padded image
        padded = np.full((h, w, 3), self.pad_value, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Convert to tensor (NCHW, normalized)
        tensor = padded.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
        tensor = np.expand_dims(tensor, 0)  # Add batch dimension

        preprocess_info = {
            "scale": scale,
            "pad_w": pad_w,
            "pad_h": pad_h,
            "orig_size": (orig_h, orig_w),
            "new_size": (new_h, new_w),
        }

        return tensor, preprocess_info


class CenterCropPreProcessor(BasePreProcessor):
    """
    Center crop pre-processor for classification models.

    Resizes and center-crops image.
    """

    def __init__(
        self,
        crop_ratio: float = 0.875,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            crop_ratio: Ratio of crop to resize (default: 0.875 = 224/256)
            mean: Normalization mean (ImageNet default)
            std: Normalization std (ImageNet default)
        """
        self.crop_ratio = crop_ratio
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def process(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply center crop preprocessing.

        Args:
            image: Input image (BGR, HWC)
            target_size: Target size (height, width)

        Returns:
            (tensor, preprocess_info)
        """
        import cv2

        h, w = target_size
        orig_h, orig_w = image.shape[:2]

        # Resize to larger size first
        resize_size = int(h / self.crop_ratio)
        scale = resize_size / min(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # Center crop
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        cropped = resized[start_h:start_h + h, start_w:start_w + w]

        # Convert BGR to RGB
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # Normalize
        tensor = rgb.astype(np.float32) / 255.0
        tensor = (tensor - self.mean) / self.std

        # Convert to NCHW
        tensor = tensor.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, 0)

        preprocess_info = {
            "scale": scale,
            "crop_start": (start_h, start_w),
            "orig_size": (orig_h, orig_w),
        }

        return tensor, preprocess_info


class YOLOPostProcessor(BasePostProcessor):
    """
    Post-processor for YOLO detection models.

    Handles YOLOv5/v8 output format.
    """

    def process(
        self,
        outputs: Dict[str, np.ndarray],
        orig_size: Tuple[int, int],
        preprocess_info: Dict[str, Any],
        config: InferConfig,
        labels: List[str] = None,
    ) -> Results:
        """
        Post-process YOLO outputs.

        Args:
            outputs: Model outputs
            orig_size: Original image size (h, w)
            preprocess_info: Preprocessing info
            config: Inference config
            labels: Class labels

        Returns:
            Results with detections
        """
        results = Results()
        labels = labels or []

        # Get first output
        output = list(outputs.values())[0]

        if len(output.shape) == 3:
            # Shape: (1, num_detections, 4+num_classes) or (1, 4+num_classes, num_detections)
            if output.shape[1] < output.shape[2]:
                output = output.transpose(0, 2, 1)

            output = output[0]  # Remove batch dimension

            scale = preprocess_info["scale"]
            pad_w = preprocess_info["pad_w"]
            pad_h = preprocess_info["pad_h"]

            detections = []
            for det in output:
                # YOLO format: [cx, cy, w, h, class_scores...]
                cx, cy, w, h = det[:4]
                scores = det[4:]

                if len(scores) == 1:
                    # Object confidence only
                    class_id = 0
                    confidence = float(scores[0])
                else:
                    # Multiple classes
                    class_id = int(np.argmax(scores))
                    confidence = float(scores[class_id])

                if confidence < config.conf_threshold:
                    continue

                # Filter by classes
                if config.classes is not None and class_id not in config.classes:
                    continue

                # Convert to original coordinates
                x1 = (cx - w / 2 - pad_w) / scale
                y1 = (cy - h / 2 - pad_h) / scale
                x2 = (cx + w / 2 - pad_w) / scale
                y2 = (cy + h / 2 - pad_h) / scale

                # Clip to image bounds
                x1 = max(0, min(x1, orig_size[1]))
                y1 = max(0, min(y1, orig_size[0]))
                x2 = max(0, min(x2, orig_size[1]))
                y2 = max(0, min(y2, orig_size[0]))

                label = labels[class_id] if class_id < len(labels) else str(class_id)

                detections.append(Detection(
                    bbox=BBox(x1, y1, x2, y2),
                    class_id=class_id,
                    label=label,
                    confidence=confidence,
                ))

            # Apply NMS
            detections = self._nms(detections, config.iou_threshold)
            results.detections = detections[:config.max_detections]

        results.raw_outputs = outputs
        return results

    def _nms(
        self,
        detections: List[Detection],
        iou_threshold: float
    ) -> List[Detection]:
        """Non-maximum suppression."""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            detections = [
                d for d in detections
                if d.class_id != best.class_id or best.bbox.iou(d.bbox) < iou_threshold
            ]

        return keep


class ClassificationPostProcessor(BasePostProcessor):
    """
    Post-processor for classification models.

    Handles softmax outputs.
    """

    def __init__(self, apply_softmax: bool = True, top_k: int = 5):
        """
        Args:
            apply_softmax: Apply softmax to logits
            top_k: Number of top predictions to return
        """
        self.apply_softmax = apply_softmax
        self.top_k = top_k

    def process(
        self,
        outputs: Dict[str, np.ndarray],
        orig_size: Tuple[int, int],
        preprocess_info: Dict[str, Any],
        config: InferConfig,
        labels: List[str] = None,
    ) -> Results:
        """
        Post-process classification outputs.

        Args:
            outputs: Model outputs (logits or probabilities)
            orig_size: Original image size
            preprocess_info: Preprocessing info
            config: Inference config
            labels: Class labels

        Returns:
            Results with classifications
        """
        results = Results()
        labels = labels or []

        # Get first output
        output = list(outputs.values())[0]

        # Remove batch dimension if present
        if len(output.shape) > 1:
            output = output.squeeze()

        # Apply softmax if needed
        if self.apply_softmax:
            exp_output = np.exp(output - np.max(output))
            probs = exp_output / exp_output.sum()
        else:
            probs = output

        # Get top-k predictions
        top_indices = np.argsort(probs)[::-1][:self.top_k]

        classifications = []
        for idx in top_indices:
            label = labels[idx] if idx < len(labels) else str(idx)
            conf = float(probs[idx])

            if conf < config.conf_threshold:
                continue

            classifications.append(ClassificationResult(
                class_id=int(idx),
                label=label,
                confidence=conf,
            ))

        results.classifications = classifications
        results.raw_outputs = outputs
        return results


# Processor registry
_PREPROCESSORS = {
    "letterbox": LetterboxPreProcessor,
    "center_crop": CenterCropPreProcessor,
}

_POSTPROCESSORS = {
    "yolo": YOLOPostProcessor,
    "classification": ClassificationPostProcessor,
}


def get_preprocessor(name: str, **kwargs) -> BasePreProcessor:
    """
    Get preprocessor by name.

    Args:
        name: Preprocessor name ("letterbox", "center_crop")
        **kwargs: Preprocessor arguments

    Returns:
        PreProcessor instance
    """
    if name not in _PREPROCESSORS:
        raise ValueError(f"Unknown preprocessor: {name}. Available: {list(_PREPROCESSORS.keys())}")
    return _PREPROCESSORS[name](**kwargs)


def get_postprocessor(name: str, **kwargs) -> BasePostProcessor:
    """
    Get postprocessor by name.

    Args:
        name: Postprocessor name ("yolo", "classification")
        **kwargs: Postprocessor arguments

    Returns:
        PostProcessor instance
    """
    if name not in _POSTPROCESSORS:
        raise ValueError(f"Unknown postprocessor: {name}. Available: {list(_POSTPROCESSORS.keys())}")
    return _POSTPROCESSORS[name](**kwargs)


def register_preprocessor(name: str, cls: type) -> None:
    """Register a custom preprocessor."""
    _PREPROCESSORS[name] = cls


def register_postprocessor(name: str, cls: type) -> None:
    """Register a custom postprocessor."""
    _POSTPROCESSORS[name] = cls
