"""
Visualization utilities.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

from ..core.types import Detection, ClassificationResult, Pose


class Visualizer:
    """
    Visualization utility class.

    Provides static methods for drawing inference results on images.
    """

    # Default colors (BGR format)
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (255, 0, 128), (128, 255, 0),
        (0, 255, 128), (128, 0, 255), (0, 128, 255),
    ]

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: List[Detection],
        labels: Optional[List[str]] = None,
        show_labels: bool = True,
        show_confidence: bool = True,
        thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Draw detection boxes on image.

        Args:
            image: Input image (will be copied)
            detections: List of detections
            labels: Class labels
            show_labels: Show class names
            show_confidence: Show confidence scores
            thickness: Box line thickness
            font_scale: Font scale

        Returns:
            Image with drawn detections
        """
        import cv2

        vis = image.copy()

        for det in detections:
            color = Visualizer.get_color(det.class_id)
            x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
            x2, y2 = int(det.bbox.x2), int(det.bbox.y2)

            # Draw box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            if show_labels or show_confidence:
                label_parts = []
                if show_labels:
                    if labels and det.class_id < len(labels):
                        label_parts.append(labels[det.class_id])
                    else:
                        label_parts.append(det.label)
                if show_confidence:
                    label_parts.append(f"{det.confidence:.2f}")

                label = " ".join(label_parts)

                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(
                    vis, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1
                )

            # Draw mask if available
            if det.mask is not None:
                mask_color = np.array(color, dtype=np.float32)
                mask_overlay = (det.mask[:, :, np.newaxis] * mask_color).astype(np.uint8)
                vis = cv2.addWeighted(vis, 1, mask_overlay, 0.5, 0)

        return vis

    @staticmethod
    def draw_segmentation(
        image: np.ndarray,
        mask: np.ndarray,
        colormap: Optional[Dict[int, Tuple[int, int, int]]] = None,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Draw segmentation mask overlay.

        Args:
            image: Input image
            mask: Segmentation mask (H, W)
            colormap: Color mapping {class_id: (B, G, R)}
            alpha: Overlay transparency

        Returns:
            Image with mask overlay
        """
        import cv2

        vis = image.copy()
        h, w = mask.shape[:2]

        # Generate colormap if not provided
        if colormap is None:
            colormap = Visualizer.generate_colormap(int(mask.max()) + 1)

        # Create colored mask
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in colormap.items():
            colored[mask == class_id] = color

        # Blend
        vis = cv2.addWeighted(vis, 1 - alpha, colored, alpha, 0)

        return vis

    @staticmethod
    def draw_poses(
        image: np.ndarray,
        poses: List[Pose],
        skeleton: Optional[List[Tuple[int, int]]] = None,
        keypoint_radius: int = 3,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw pose keypoints and skeleton.

        Args:
            image: Input image
            poses: List of poses
            skeleton: Skeleton connections [(i, j), ...]
            keypoint_radius: Keypoint circle radius
            thickness: Line thickness

        Returns:
            Image with drawn poses
        """
        import cv2

        vis = image.copy()

        # Default COCO skeleton
        if skeleton is None:
            skeleton = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 11), (6, 12), (11, 12),  # Torso
                (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            ]

        for pose in poses:
            # Draw keypoints
            for i, kp in enumerate(pose.keypoints):
                if kp.confidence > 0.5:
                    color = Visualizer.get_color(i)
                    cv2.circle(
                        vis,
                        (int(kp.x), int(kp.y)),
                        keypoint_radius,
                        color,
                        -1
                    )

            # Draw skeleton
            for i, j in skeleton:
                if i < len(pose.keypoints) and j < len(pose.keypoints):
                    kp1, kp2 = pose.keypoints[i], pose.keypoints[j]
                    if kp1.confidence > 0.5 and kp2.confidence > 0.5:
                        cv2.line(
                            vis,
                            (int(kp1.x), int(kp1.y)),
                            (int(kp2.x), int(kp2.y)),
                            Visualizer.get_color(i),
                            thickness
                        )

        return vis

    @staticmethod
    def draw_classification(
        image: np.ndarray,
        results: List[ClassificationResult],
        top_k: int = 5,
        font_scale: float = 0.6,
    ) -> np.ndarray:
        """
        Draw classification results as text.

        Args:
            image: Input image
            results: Classification results
            top_k: Number of results to show
            font_scale: Font scale

        Returns:
            Image with classification text
        """
        import cv2

        vis = image.copy()

        for i, result in enumerate(results[:top_k]):
            text = f"{result.label}: {result.score:.2%}"
            y = 30 + i * 25

            cv2.putText(
                vis, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), 3  # Black outline
            )
            cv2.putText(
                vis, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 1  # White text
            )

        return vis

    @staticmethod
    def create_comparison(
        images: List[np.ndarray],
        titles: Optional[List[str]] = None,
        cols: int = 2,
    ) -> np.ndarray:
        """
        Create side-by-side comparison image.

        Args:
            images: List of images
            titles: Titles for each image
            cols: Number of columns

        Returns:
            Combined image
        """
        import cv2

        n = len(images)
        rows = (n + cols - 1) // cols

        # Get max dimensions
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)

        # Create canvas
        canvas = np.zeros((rows * (max_h + 30), cols * max_w, 3), dtype=np.uint8)

        for i, img in enumerate(images):
            row, col = i // cols, i % cols
            y, x = row * (max_h + 30) + 30, col * max_w

            # Resize if needed
            if img.shape[:2] != (max_h, max_w):
                img = cv2.resize(img, (max_w, max_h))

            canvas[y:y+max_h, x:x+max_w] = img

            # Add title
            if titles and i < len(titles):
                cv2.putText(
                    canvas, titles[i], (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                )

        return canvas

    @staticmethod
    def get_color(class_id: int) -> Tuple[int, int, int]:
        """Get color for class ID."""
        return Visualizer.COLORS[class_id % len(Visualizer.COLORS)]

    @staticmethod
    def generate_colormap(num_classes: int) -> Dict[int, Tuple[int, int, int]]:
        """Generate colormap for given number of classes."""
        import colorsys

        colormap = {}
        for i in range(num_classes):
            hue = i / num_classes
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
            colormap[i] = tuple(int(c * 255) for c in rgb)

        # Background is black
        colormap[0] = (0, 0, 0)

        return colormap
