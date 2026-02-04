"""
Pre-processing and post-processing classes for iVIT-SDK.

Extracted from Model class to improve modularity and testability.
"""

from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
import logging

from .types import Detection, BBox, ClassificationResult, InferConfig, Pose, Keypoint
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
        """
        Optimized Non-Maximum Suppression using vectorized operations.

        Uses numpy for batch IoU computation, significantly faster than
        the naive O(n²) approach for large numbers of detections.
        """
        if not detections:
            return []

        if len(detections) == 1:
            return detections

        # Convert to numpy arrays for vectorized operations
        n = len(detections)
        boxes = np.zeros((n, 4), dtype=np.float32)
        scores = np.zeros(n, dtype=np.float32)
        class_ids = np.zeros(n, dtype=np.int32)

        for i, det in enumerate(detections):
            boxes[i] = [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
            scores[i] = det.confidence
            class_ids[i] = det.class_id

        # Sort by confidence (descending)
        order = np.argsort(-scores)

        # Pre-compute areas
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        keep_indices = []

        while len(order) > 0:
            # Take the detection with highest confidence
            idx = order[0]
            keep_indices.append(idx)

            if len(order) == 1:
                break

            # Compute IoU with remaining boxes
            remaining = order[1:]

            # Intersection coordinates
            xx1 = np.maximum(x1[idx], x1[remaining])
            yy1 = np.maximum(y1[idx], y1[remaining])
            xx2 = np.minimum(x2[idx], x2[remaining])
            yy2 = np.minimum(y2[idx], y2[remaining])

            # Intersection area
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            # IoU
            union = areas[idx] + areas[remaining] - intersection
            iou = np.where(union > 0, intersection / union, 0)

            # Keep boxes with low IoU OR different class
            same_class = class_ids[remaining] == class_ids[idx]
            suppress = (iou >= iou_threshold) & same_class

            # Update order to keep non-suppressed boxes
            order = remaining[~suppress]

        # Return kept detections in order
        return [detections[i] for i in keep_indices]


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
                score=conf,
            ))

        results.classifications = classifications
        results.raw_outputs = outputs
        return results


class YOLOXPostProcessor(BasePostProcessor):
    """
    Post-processor for YOLOX detection models.

    YOLOX outputs decoded bounding boxes directly, format:
    [batch, num_anchors, 85] where 85 = 4 (bbox) + 1 (obj_conf) + 80 (class_scores)
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
        Post-process YOLOX outputs.

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
            output = output[0]  # Remove batch dimension

        scale = preprocess_info.get("scale", 1.0)
        pad_w = preprocess_info.get("pad_w", 0)
        pad_h = preprocess_info.get("pad_h", 0)

        detections = []

        for det in output:
            # YOLOX format: [cx, cy, w, h, obj_conf, class_scores...]
            cx, cy, w, h, obj_conf = det[:5]
            class_scores = det[5:]

            # Object confidence threshold
            if obj_conf < config.conf_threshold:
                continue

            # Get class with highest score
            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])

            # Final confidence = obj_conf * class_conf
            confidence = float(obj_conf * class_conf)

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

        # Apply NMS (reuse from YOLOPostProcessor)
        detections = self._nms(detections, config.iou_threshold)
        results.detections = detections[:config.max_detections]

        results.raw_outputs = outputs
        return results

    def _nms(
        self,
        detections: List[Detection],
        iou_threshold: float
    ) -> List[Detection]:
        """Non-Maximum Suppression using vectorized operations."""
        if not detections:
            return []

        if len(detections) == 1:
            return detections

        n = len(detections)
        boxes = np.zeros((n, 4), dtype=np.float32)
        scores = np.zeros(n, dtype=np.float32)
        class_ids = np.zeros(n, dtype=np.int32)

        for i, det in enumerate(detections):
            boxes[i] = [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
            scores[i] = det.confidence
            class_ids[i] = det.class_id

        order = np.argsort(-scores)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        keep_indices = []

        while len(order) > 0:
            idx = order[0]
            keep_indices.append(idx)

            if len(order) == 1:
                break

            remaining = order[1:]
            xx1 = np.maximum(x1[idx], x1[remaining])
            yy1 = np.maximum(y1[idx], y1[remaining])
            xx2 = np.minimum(x2[idx], x2[remaining])
            yy2 = np.minimum(y2[idx], y2[remaining])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            union = areas[idx] + areas[remaining] - intersection
            iou = np.where(union > 0, intersection / union, 0)

            same_class = class_ids[remaining] == class_ids[idx]
            suppress = (iou >= iou_threshold) & same_class

            order = remaining[~suppress]

        return [detections[i] for i in keep_indices]


class RTDETRPostProcessor(BasePostProcessor):
    """
    Post-processor for RT-DETR detection models.

    RT-DETR outputs:
    - [batch, num_queries, 4+num_classes] where:
      - 4 = normalized [cx, cy, w, h]
      - num_classes = class logits (no softmax)
    - No NMS needed (Transformer already performs deduplication)
    """

    def __init__(self, num_queries: int = 300):
        """
        Args:
            num_queries: Number of object queries (default: 300)
        """
        self.num_queries = num_queries

    def process(
        self,
        outputs: Dict[str, np.ndarray],
        orig_size: Tuple[int, int],
        preprocess_info: Dict[str, Any],
        config: InferConfig,
        labels: List[str] = None,
    ) -> Results:
        """
        Post-process RT-DETR outputs.

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

        # RT-DETR typically has two outputs: boxes and scores
        # or a single combined output
        output_list = list(outputs.values())

        if len(output_list) == 2:
            # Separate boxes and logits
            boxes_output = output_list[0]
            logits_output = output_list[1]

            # Check shapes to determine which is which
            if boxes_output.shape[-1] > 4:
                boxes_output, logits_output = logits_output, boxes_output
        else:
            # Combined output [batch, num_queries, 4+num_classes]
            output = output_list[0]
            if len(output.shape) == 3:
                output = output[0]  # Remove batch dimension

            boxes_output = output[:, :4]
            logits_output = output[:, 4:]

        if len(boxes_output.shape) == 3:
            boxes_output = boxes_output[0]
        if len(logits_output.shape) == 3:
            logits_output = logits_output[0]

        orig_h, orig_w = orig_size

        # Apply sigmoid to logits to get probabilities
        probs = 1 / (1 + np.exp(-logits_output))

        detections = []

        for i in range(len(boxes_output)):
            box = boxes_output[i]
            scores = probs[i]

            # Get best class
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence < config.conf_threshold:
                continue

            # Filter by classes
            if config.classes is not None and class_id not in config.classes:
                continue

            # RT-DETR outputs normalized [cx, cy, w, h]
            cx, cy, w, h = box[:4]

            # Convert to absolute coordinates
            x1 = (cx - w / 2) * orig_w
            y1 = (cy - h / 2) * orig_h
            x2 = (cx + w / 2) * orig_w
            y2 = (cy + h / 2) * orig_h

            # Clip to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            label = labels[class_id] if class_id < len(labels) else str(class_id)

            detections.append(Detection(
                bbox=BBox(x1, y1, x2, y2),
                class_id=class_id,
                label=label,
                confidence=confidence,
            ))

        # Sort by confidence and limit
        detections.sort(key=lambda x: x.confidence, reverse=True)
        results.detections = detections[:config.max_detections]

        results.raw_outputs = outputs
        return results


# COCO keypoint names
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


class RTMPosePostProcessor(BasePostProcessor):
    """
    Post-processor for RTMPose models.

    RTMPose uses SimCC (Simple Coordinate Classification) approach:
    - Outputs x and y coordinate distributions separately
    - Shape: [batch, num_keypoints, width] and [batch, num_keypoints, height]

    Alternative regression output:
    - Shape: [batch, num_keypoints, 2] for (x, y) coordinates
    - Plus optional heatmap confidence
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        keypoint_names: List[str] = None,
    ):
        """
        Args:
            num_keypoints: Number of keypoints (default: 17 for COCO)
            keypoint_names: Names for each keypoint
        """
        self.num_keypoints = num_keypoints
        self.keypoint_names = keypoint_names or COCO_KEYPOINT_NAMES[:num_keypoints]

    def process(
        self,
        outputs: Dict[str, np.ndarray],
        orig_size: Tuple[int, int],
        preprocess_info: Dict[str, Any],
        config: InferConfig,
        labels: List[str] = None,
    ) -> Results:
        """
        Post-process RTMPose outputs.

        Args:
            outputs: Model outputs
            orig_size: Original image size (h, w)
            preprocess_info: Preprocessing info
            config: Inference config
            labels: Unused for pose

        Returns:
            Results with poses
        """
        results = Results()

        output_list = list(outputs.values())
        output_names = list(outputs.keys())

        orig_h, orig_w = orig_size
        input_h = preprocess_info.get("new_size", (256, 192))[0]
        input_w = preprocess_info.get("new_size", (256, 192))[1]

        # Handle different output formats
        if len(output_list) == 2:
            # SimCC format: separate x and y distributions
            simcc_x = output_list[0]
            simcc_y = output_list[1]

            # Check which is x and which is y based on shape
            if simcc_x.shape[-1] < simcc_y.shape[-1]:
                simcc_x, simcc_y = simcc_y, simcc_x

            keypoints = self._decode_simcc(simcc_x, simcc_y, orig_size, preprocess_info)

        else:
            # Single output - could be coordinates or heatmaps
            output = output_list[0]

            if len(output.shape) == 4:
                # Heatmap format: [batch, num_keypoints, h, w]
                keypoints = self._decode_heatmaps(output, orig_size, preprocess_info)
            else:
                # Coordinate format: [batch, num_keypoints, 2 or 3]
                keypoints = self._decode_coordinates(output, orig_size, preprocess_info)

        # Create pose result
        if keypoints:
            pose = Pose(
                keypoints=keypoints,
                confidence=np.mean([kp.confidence for kp in keypoints]),
            )
            results.poses = [pose]

        results.raw_outputs = outputs
        return results

    def _decode_simcc(
        self,
        simcc_x: np.ndarray,
        simcc_y: np.ndarray,
        orig_size: Tuple[int, int],
        preprocess_info: Dict[str, Any],
    ) -> List[Keypoint]:
        """Decode SimCC outputs to keypoints."""
        if len(simcc_x.shape) == 3:
            simcc_x = simcc_x[0]  # Remove batch
            simcc_y = simcc_y[0]

        orig_h, orig_w = orig_size
        keypoints = []

        for i in range(min(len(simcc_x), self.num_keypoints)):
            # Get peak position
            x_idx = np.argmax(simcc_x[i])
            y_idx = np.argmax(simcc_y[i])

            # Get confidence as softmax peak
            x_conf = np.max(simcc_x[i])
            y_conf = np.max(simcc_y[i])
            conf = (x_conf + y_conf) / 2

            # Convert to original image coordinates
            # SimCC uses expanded coordinate space
            x = x_idx / simcc_x.shape[-1] * orig_w
            y = y_idx / simcc_y.shape[-1] * orig_h

            name = self.keypoint_names[i] if i < len(self.keypoint_names) else f"kp_{i}"

            keypoints.append(Keypoint(
                x=float(x),
                y=float(y),
                confidence=float(conf),
                name=name,
            ))

        return keypoints

    def _decode_heatmaps(
        self,
        heatmaps: np.ndarray,
        orig_size: Tuple[int, int],
        preprocess_info: Dict[str, Any],
    ) -> List[Keypoint]:
        """Decode heatmap outputs to keypoints."""
        if len(heatmaps.shape) == 4:
            heatmaps = heatmaps[0]  # Remove batch

        orig_h, orig_w = orig_size
        num_kp, hm_h, hm_w = heatmaps.shape

        keypoints = []

        for i in range(min(num_kp, self.num_keypoints)):
            hm = heatmaps[i]

            # Find peak
            idx = np.argmax(hm)
            y_idx, x_idx = np.unravel_index(idx, hm.shape)
            conf = hm[y_idx, x_idx]

            # Convert to original coordinates
            x = x_idx / hm_w * orig_w
            y = y_idx / hm_h * orig_h

            name = self.keypoint_names[i] if i < len(self.keypoint_names) else f"kp_{i}"

            keypoints.append(Keypoint(
                x=float(x),
                y=float(y),
                confidence=float(conf),
                name=name,
            ))

        return keypoints

    def _decode_coordinates(
        self,
        coords: np.ndarray,
        orig_size: Tuple[int, int],
        preprocess_info: Dict[str, Any],
    ) -> List[Keypoint]:
        """Decode coordinate outputs to keypoints."""
        if len(coords.shape) == 3:
            coords = coords[0]  # Remove batch

        orig_h, orig_w = orig_size
        input_size = preprocess_info.get("new_size", (256, 192))

        keypoints = []

        for i in range(min(len(coords), self.num_keypoints)):
            kp = coords[i]

            if len(kp) >= 3:
                x, y, conf = kp[:3]
            else:
                x, y = kp[:2]
                conf = 1.0

            # Coordinates may be normalized or in input size
            if x <= 1.0 and y <= 1.0:
                # Normalized
                x = x * orig_w
                y = y * orig_h
            else:
                # In input size - scale to original
                x = x / input_size[1] * orig_w
                y = y / input_size[0] * orig_h

            name = self.keypoint_names[i] if i < len(self.keypoint_names) else f"kp_{i}"

            keypoints.append(Keypoint(
                x=float(x),
                y=float(y),
                confidence=float(conf),
                name=name,
            ))

        return keypoints


class TorchVisionDetectionPostProcessor(BasePostProcessor):
    """
    Post-processor for torchvision detection models.

    Handles outputs from Faster R-CNN, RetinaNet, SSD, FCOS exported to ONNX.
    Output format:
    - boxes: [num_detections, 4] - absolute coordinates [x1, y1, x2, y2]
    - scores: [num_detections] - confidence scores
    - labels: [num_detections] - class indices (as floats)

    Note: torchvision detection models apply NMS internally, so no NMS is needed.
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
        Post-process torchvision detection outputs.

        Args:
            outputs: Model outputs with 'boxes', 'scores', 'labels' keys
            orig_size: Original image size (h, w)
            preprocess_info: Preprocessing info (may contain scale)
            config: Inference config
            labels: Class labels (COCO 91 classes by default)

        Returns:
            Results with detections
        """
        results = Results()
        labels_list = labels or []

        # Get outputs by name or position
        output_names = list(outputs.keys())

        if 'boxes' in outputs:
            boxes = outputs['boxes']
            scores = outputs['scores']
            class_ids = outputs['labels']
        elif len(outputs) == 3:
            # Assume order: boxes, scores, labels
            boxes = outputs[output_names[0]]
            scores = outputs[output_names[1]]
            class_ids = outputs[output_names[2]]
        else:
            logger.warning(f"Unexpected output format: {output_names}")
            results.raw_outputs = outputs
            return results

        # Handle batch dimension
        if len(boxes.shape) == 3:
            boxes = boxes[0]
        if len(scores.shape) == 2:
            scores = scores[0]
        if len(class_ids.shape) == 2:
            class_ids = class_ids[0]

        # Convert labels to int (may be float from ONNX)
        class_ids = class_ids.astype(np.int32)

        orig_h, orig_w = orig_size

        # Get scale factors if preprocessed
        scale_x = preprocess_info.get('scale_x', 1.0)
        scale_y = preprocess_info.get('scale_y', 1.0)

        # If letterbox was used, get scale
        if 'scale' in preprocess_info:
            scale = preprocess_info['scale']
            scale_x = scale_y = scale

        detections = []

        for i in range(len(boxes)):
            score = float(scores[i])

            if score < config.conf_threshold:
                continue

            class_id = int(class_ids[i])

            # Filter by classes
            if config.classes is not None and class_id not in config.classes:
                continue

            # Get box coordinates (already in input image space)
            x1, y1, x2, y2 = boxes[i]

            # Scale back to original image if needed
            if scale_x != 1.0 or scale_y != 1.0:
                x1 = x1 / scale_x
                x2 = x2 / scale_x
                y1 = y1 / scale_y
                y2 = y2 / scale_y

            # Clip to image bounds
            x1 = max(0, min(float(x1), orig_w))
            y1 = max(0, min(float(y1), orig_h))
            x2 = max(0, min(float(x2), orig_w))
            y2 = max(0, min(float(y2), orig_h))

            # Get label
            label = labels_list[class_id] if class_id < len(labels_list) else str(class_id)

            detections.append(Detection(
                bbox=BBox(x1, y1, x2, y2),
                class_id=class_id,
                label=label,
                confidence=score,
            ))

        # Torchvision already applies NMS, so just sort and limit
        detections.sort(key=lambda x: x.confidence, reverse=True)
        results.detections = detections[:config.max_detections]

        results.raw_outputs = outputs
        return results


# Processor registry
_PREPROCESSORS = {
    "letterbox": LetterboxPreProcessor,
    "center_crop": CenterCropPreProcessor,
}

_POSTPROCESSORS = {
    "yolo": YOLOPostProcessor,
    "yolox": YOLOXPostProcessor,
    "rtdetr": RTDETRPostProcessor,
    "rtmpose": RTMPosePostProcessor,
    "classification": ClassificationPostProcessor,
    "torchvision_detection": TorchVisionDetectionPostProcessor,
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
