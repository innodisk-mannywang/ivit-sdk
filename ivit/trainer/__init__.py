"""
iVIT Trainer Module
==================
Task-specific trainers for AI vision tasks.
"""

from .classification import ClassificationTrainer, ClassificationConfig
from .detection import DetectionTrainer, DetectionConfig
from .segmentation import SegmentationTrainer, SegmentationConfig

__all__ = [
    'ClassificationTrainer', 'ClassificationConfig',
    'DetectionTrainer', 'DetectionConfig', 
    'SegmentationTrainer', 'SegmentationConfig'
]
