"""
iVIT 2.0 SDK - Inference Module
提供統一的推理接口，支援分類、偵測、分割任務
"""

from ..core.base_inference import BaseInference, InferenceConfig
from .classification_inference import ClassificationInference
from .detection_inference import DetectionInference
from .segmentation_inference import SegmentationInference
from .unified_inference import UnifiedInference

__all__ = [
    'BaseInference',
    'InferenceConfig', 
    'ClassificationInference',
    'DetectionInference',
    'SegmentationInference',
    'UnifiedInference'
]

__version__ = "2.0.0"
