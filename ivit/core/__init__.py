"""
iVIT Core Module
===============
Core components for AI vision training framework.
"""

from .base_trainer import BaseTrainer, TaskConfig
from .base_inference import BaseInference, InferenceConfig

__all__ = ['BaseTrainer', 'TaskConfig', 'BaseInference', 'InferenceConfig']
