"""
Vision modules for iVIT-SDK.

This module provides high-level APIs for computer vision tasks:
- Detector: Object detection
- Classifier: Image classification
- Segmentor: Semantic segmentation

The implementation automatically uses C++ bindings when available
for optimal performance, with pure Python fallback.
"""

from typing import TYPE_CHECKING

# Check if C++ bindings are available
try:
    from .._ivit_core import (
        Detector as _CppDetector,
        Classifier as _CppClassifier,
        Segmentor as _CppSegmentor,
    )
    _HAS_CPP_VISION = True
except ImportError:
    _HAS_CPP_VISION = False

if _HAS_CPP_VISION:
    # Use C++ implementations with Python wrappers for better ergonomics
    from .detector import Detector
    from .classifier import Classifier
    from .segmentor import Segmentor
else:
    # Fall back to pure Python implementations
    from .detector import Detector
    from .classifier import Classifier
    from .segmentor import Segmentor


__all__ = [
    "Detector",
    "Classifier",
    "Segmentor",
]
