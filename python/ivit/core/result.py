"""
Results container for iVIT-SDK.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import json
from pathlib import Path

from .types import (
    Detection,
    ClassificationResult,
    Keypoint,
    Pose,
    BBox,
)


class Results:
    """
    Unified results container for all task types.

    Attributes:
        classifications: List of classification results
        detections: List of detection results
        segmentation_mask: Segmentation mask array
        poses: List of pose results
        raw_outputs: Raw model outputs
        inference_time_ms: Inference time in milliseconds
        device_used: Device used for inference
        image_size: Original image size (H, W)
    """

    def __init__(self):
        # Classification
        self.classifications: List[ClassificationResult] = []

        # Detection
        self.detections: List[Detection] = []

        # Segmentation
        self.segmentation_mask: Optional[np.ndarray] = None

        # Pose
        self.poses: List[Pose] = []

        # Raw outputs
        self.raw_outputs: Dict[str, np.ndarray] = {}

        # Metadata
        self.inference_time_ms: float = 0.0
        self.device_used: str = ""
        self.image_size: Tuple[int, int] = (0, 0)
        self._original_image: Optional[np.ndarray] = None

    # =========================================================================
    # Classification
    # =========================================================================

    @property
    def top1(self) -> ClassificationResult:
        """Get top-1 classification result."""
        if not self.classifications:
            raise ValueError("No classification results")
        return self.classifications[0]

    @property
    def top5(self) -> List[ClassificationResult]:
        """Get top-5 classification results."""
        return self.classifications[:5]

    def topk(self, k: int) -> List[ClassificationResult]:
        """Get top-K classification results."""
        return self.classifications[:k]

    # =========================================================================
    # Detection
    # =========================================================================

    def filter_by_class(
        self,
        classes: List[str]
    ) -> List[Detection]:
        """Filter detections by class labels."""
        return [d for d in self.detections if d.label in classes]

    def filter_by_confidence(
        self,
        min_conf: float
    ) -> List[Detection]:
        """Filter detections by minimum confidence."""
        return [d for d in self.detections if d.confidence >= min_conf]

    def filter_by_area(
        self,
        min_area: float,
        max_area: Optional[float] = None
    ) -> List[Detection]:
        """Filter detections by bounding box area."""
        result = [d for d in self.detections if d.bbox.area >= min_area]
        if max_area is not None:
            result = [d for d in result if d.bbox.area <= max_area]
        return result

    # =========================================================================
    # Segmentation
    # =========================================================================

    def colorize_mask(
        self,
        colormap: Optional[Dict[int, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Colorize segmentation mask.

        Args:
            colormap: Custom color mapping {class_id: (R, G, B)}

        Returns:
            RGB colorized mask
        """
        if self.segmentation_mask is None:
            raise ValueError("No segmentation mask")

        mask = self.segmentation_mask
        h, w = mask.shape[:2]
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        # Generate default colormap if not provided
        if colormap is None:
            colormap = self._generate_colormap(int(mask.max()) + 1)

        for class_id, color in colormap.items():
            colored[mask == class_id] = color

        return colored

    def overlay_mask(
        self,
        image: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay segmentation mask on image.

        Args:
            image: Original image (BGR)
            alpha: Transparency (0-1)

        Returns:
            Image with overlaid mask
        """
        colored = self.colorize_mask()
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            overlay = image.copy()
            mask_rgb = colored[:, :, ::-1]  # RGB to BGR
            cv2_available = True
            try:
                import cv2
                overlay = cv2.addWeighted(overlay, 1 - alpha, mask_rgb, alpha, 0)
            except ImportError:
                overlay = (overlay * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
            return overlay
        return image

    def get_contours(
        self,
        class_id: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract contours from segmentation mask.

        Args:
            class_id: Specific class (None for all)

        Returns:
            List of contour arrays
        """
        if self.segmentation_mask is None:
            raise ValueError("No segmentation mask")

        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required for contour extraction")

        contours = []
        mask = self.segmentation_mask

        if class_id is not None:
            binary = (mask == class_id).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(cnts)
        else:
            for cid in np.unique(mask):
                if cid == 0:  # Skip background
                    continue
                binary = (mask == cid).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours.extend(cnts)

        return contours

    # =========================================================================
    # Visualization
    # =========================================================================

    def visualize(
        self,
        image: Optional[np.ndarray] = None,
        show_labels: bool = True,
        show_confidence: bool = True,
        show_boxes: bool = True,
        show_masks: bool = True,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize results on image.

        Args:
            image: Input image (uses cached if None)
            show_labels: Show class labels
            show_confidence: Show confidence scores
            show_boxes: Show bounding boxes
            show_masks: Show segmentation masks

        Returns:
            Visualized image
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required for visualization")

        if image is None:
            image = self._original_image
        if image is None:
            raise ValueError("No image provided")

        vis = image.copy()

        # Draw segmentation mask
        if show_masks and self.segmentation_mask is not None:
            vis = self.overlay_mask(vis, alpha=0.5)

        # Draw detections
        if show_boxes and self.detections:
            for det in self.detections:
                color = self._get_color(det.class_id)
                x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
                x2, y2 = int(det.bbox.x2), int(det.bbox.y2)

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                if show_labels or show_confidence:
                    label_parts = []
                    if show_labels:
                        label_parts.append(det.label)
                    if show_confidence:
                        label_parts.append(f"{det.confidence:.2f}")
                    label = " ".join(label_parts)

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                    cv2.putText(vis, label, (x1, y1 - 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw poses
        if self.poses:
            vis = self._draw_poses(vis, self.poses)

        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, vis)

        return vis

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "inference_time_ms": self.inference_time_ms,
            "device_used": self.device_used,
            "image_size": list(self.image_size),
        }

        if self.classifications:
            result["classifications"] = [
                {
                    "class_id": c.class_id,
                    "label": c.label,
                    "score": c.score,
                }
                for c in self.classifications
            ]

        if self.detections:
            result["detections"] = [
                {
                    "class_id": d.class_id,
                    "label": d.label,
                    "confidence": d.confidence,
                    "bbox": {
                        "x1": d.bbox.x1,
                        "y1": d.bbox.y1,
                        "x2": d.bbox.x2,
                        "y2": d.bbox.y2,
                    }
                }
                for d in self.detections
            ]

        if self.poses:
            result["poses"] = [
                {
                    "keypoints": [
                        {"x": kp.x, "y": kp.y, "confidence": kp.confidence, "name": kp.name}
                        for kp in p.keypoints
                    ],
                    "confidence": p.confidence,
                }
                for p in self.poses
            ]

        return result

    def save(
        self,
        path: str,
        format: str = "auto"
    ) -> None:
        """
        Save results to file.

        Args:
            path: Output path
            format: Format (auto, json, csv, txt)
        """
        path = Path(path)

        if format == "auto":
            format = path.suffix.lstrip(".")

        if format == "json":
            with open(path, "w") as f:
                f.write(self.to_json())
        elif format == "txt":
            # YOLO format for detections
            with open(path, "w") as f:
                for det in self.detections:
                    cx, cy, w, h = det.bbox.to_cxcywh()
                    # Normalize by image size
                    ih, iw = self.image_size
                    f.write(f"{det.class_id} {cx/iw:.6f} {cy/ih:.6f} {w/iw:.6f} {h/ih:.6f}\n")
        else:
            raise ValueError(f"Unsupported format: {format}")

    # =========================================================================
    # Iteration
    # =========================================================================

    def __len__(self) -> int:
        """Get total number of results."""
        return max(
            len(self.classifications),
            len(self.detections),
            len(self.poses),
        )

    def __iter__(self):
        """Iterate over detections (most common use case)."""
        return iter(self.detections)

    def __getitem__(self, idx: int) -> Detection:
        """Get detection by index."""
        return self.detections[idx]

    # =========================================================================
    # Private methods
    # =========================================================================

    @staticmethod
    def _get_color(class_id: int) -> Tuple[int, int, int]:
        """Get color for class ID."""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]
        return colors[class_id % len(colors)]

    @staticmethod
    def _generate_colormap(num_classes: int) -> Dict[int, Tuple[int, int, int]]:
        """Generate default colormap."""
        import colorsys
        colormap = {}
        for i in range(num_classes):
            hue = i / num_classes
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
            colormap[i] = tuple(int(c * 255) for c in rgb)
        return colormap

    @staticmethod
    def _draw_poses(
        image: np.ndarray,
        poses: List[Pose]
    ) -> np.ndarray:
        """Draw poses on image."""
        try:
            import cv2
        except ImportError:
            return image

        # COCO skeleton
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
        ]

        for pose in poses:
            # Draw keypoints
            for kp in pose.keypoints:
                if kp.confidence > 0.5:
                    cv2.circle(image, (int(kp.x), int(kp.y)), 3, (0, 255, 0), -1)

            # Draw skeleton
            for i, j in skeleton:
                if i < len(pose.keypoints) and j < len(pose.keypoints):
                    kp1, kp2 = pose.keypoints[i], pose.keypoints[j]
                    if kp1.confidence > 0.5 and kp2.confidence > 0.5:
                        cv2.line(
                            image,
                            (int(kp1.x), int(kp1.y)),
                            (int(kp2.x), int(kp2.y)),
                            (255, 0, 0), 2
                        )

        return image
