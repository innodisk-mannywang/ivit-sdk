"""
Type definitions for iVIT-SDK.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np


class DataType(Enum):
    """Data type enumeration."""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8 = "int8"
    UINT8 = "uint8"
    INT32 = "int32"
    INT64 = "int64"


class BackendType(Enum):
    """Backend type enumeration."""
    OPENVINO = "openvino"
    TENSORRT = "tensorrt"
    SNPE = "snpe"
    ONNXRUNTIME = "onnxruntime"
    AUTO = "auto"


class TaskType(Enum):
    """Task type enumeration."""
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    POSE_ESTIMATION = "pose_estimation"
    FACE_DETECTION = "face_detection"
    FACE_RECOGNITION = "face_recognition"
    OCR = "ocr"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class LoadConfig:
    """Model loading configuration."""
    device: str = "auto"
    backend: str = "auto"
    task: Optional[str] = None
    batch_size: int = 1
    precision: Optional[str] = None
    cache_dir: Optional[str] = None
    use_cache: bool = True


@dataclass
class InferConfig:
    """Inference configuration."""
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    classes: Optional[List[int]] = None
    enable_profiling: bool = False


@dataclass
class TensorInfo:
    """Tensor information."""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    layout: str = "NCHW"

    @property
    def numel(self) -> int:
        """Get total number of elements."""
        result = 1
        for d in self.shape:
            result *= d
        return result


@dataclass
class DeviceInfo:
    """Device information."""
    id: str
    name: str
    backend: str
    type: str  # "cpu", "gpu", "npu"
    memory_total: int = 0
    memory_available: int = 0
    supported_precisions: List[str] = field(default_factory=list)
    is_available: bool = True


@dataclass
class BBox:
    """Bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def iou(self, other: 'BBox') -> float:
        """Calculate IoU with another box."""
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Convert to [x, y, w, h] format."""
        return (self.x1, self.y1, self.width, self.height)

    def to_cxcywh(self) -> Tuple[float, float, float, float]:
        """Convert to [cx, cy, w, h] format."""
        cx, cy = self.center
        return (cx, cy, self.width, self.height)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> 'BBox':
        """Create from [x, y, w, h] format."""
        return cls(x, y, x + w, y + h)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> 'BBox':
        """Create from [cx, cy, w, h] format."""
        return cls(cx - w/2, cy - h/2, cx + w/2, cy + h/2)


@dataclass
class Detection:
    """Detection result."""
    bbox: BBox
    class_id: int
    label: str
    confidence: float
    mask: Optional[np.ndarray] = None


@dataclass
class ClassificationResult:
    """Classification result."""
    class_id: int
    label: str
    score: float


@dataclass
class Keypoint:
    """Keypoint for pose estimation."""
    x: float
    y: float
    confidence: float
    name: str = ""


@dataclass
class Pose:
    """Pose estimation result."""
    keypoints: List[Keypoint]
    bbox: Optional[BBox] = None
    confidence: float = 0.0

    def get_keypoint(self, name: str) -> Optional[Keypoint]:
        """Get keypoint by name."""
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None
