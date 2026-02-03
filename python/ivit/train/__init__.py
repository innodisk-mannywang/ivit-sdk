"""
Training module for iVIT-SDK.

Provides transfer learning capabilities for classification and detection models.

This module supports two backends:
1. C++ backend (LibTorch) - Faster, unified with inference API
2. Python backend (PyTorch) - Fallback, more flexible

The C++ backend is used when available (IVIT_USE_TORCH=ON during build).
"""

import logging

logger = logging.getLogger(__name__)

# Try to import from C++ backend first
_USE_CPP_BACKEND = False

try:
    from .._ivit_train_core import (
        # Module functions
        is_training_available as _cpp_is_training_available,
        training_version as _cpp_training_version,
        # Config
        TrainerConfig as _CppTrainerConfig,
        ExportOptions as _CppExportOptions,
        TrainingMetrics as _CppTrainingMetrics,
        DetectionTarget as _CppDetectionTarget,
        # Datasets
        IDataset as _CppIDataset,
        ImageFolderDataset as _CppImageFolderDataset,
        COCODataset as _CppCOCODataset,
        YOLODataset as _CppYOLODataset,
        split_dataset as _cpp_split_dataset,
        # Transforms
        ITransform as _CppITransform,
        Compose as _CppCompose,
        Resize as _CppResize,
        RandomHorizontalFlip as _CppRandomHorizontalFlip,
        RandomVerticalFlip as _CppRandomVerticalFlip,
        RandomRotation as _CppRandomRotation,
        ColorJitter as _CppColorJitter,
        Normalize as _CppNormalize,
        ToTensor as _CppToTensor,
        CenterCrop as _CppCenterCrop,
        RandomCrop as _CppRandomCrop,
        GaussianBlur as _CppGaussianBlur,
        get_default_augmentation as _cpp_get_default_augmentation,
        get_train_augmentation as _cpp_get_train_augmentation,
        get_val_augmentation as _cpp_get_val_augmentation,
        # Callbacks
        ITrainingCallback as _CppITrainingCallback,
        EarlyStopping as _CppEarlyStopping,
        ModelCheckpoint as _CppModelCheckpoint,
        ProgressLogger as _CppProgressLogger,
        LRScheduler as _CppLRScheduler,
        TensorBoardLogger as _CppTensorBoardLogger,
        CSVLogger as _CppCSVLogger,
        # Trainer
        Trainer as _CppTrainer,
        # Exporter
        ModelExporter as _CppModelExporter,
        export_model as _cpp_export_model,
    )

    if _cpp_is_training_available():
        _USE_CPP_BACKEND = True
        logger.debug(f"Using C++ training backend (version {_cpp_training_version()})")
    else:
        logger.debug("C++ backend available but LibTorch not compiled in, using Python backend")

except ImportError as e:
    logger.debug(f"C++ training backend not available: {e}, using Python backend")

# Import Python backend classes
from .dataset import (
    BaseDataset as _PyBaseDataset,
    ImageFolderDataset as _PyImageFolderDataset,
    COCODataset as _PyCOCODataset,
    YOLODataset as _PyYOLODataset,
    split_dataset as _py_split_dataset,
)

from .trainer import Trainer as _PyTrainer

from .callbacks import (
    TrainingCallback as _PyTrainingCallback,
    EarlyStopping as _PyEarlyStopping,
    ModelCheckpoint as _PyModelCheckpoint,
    ProgressLogger as _PyProgressLogger,
    LRScheduler as _PyLRScheduler,
    TensorBoardLogger as _PyTensorBoardLogger,
)

from .augmentation import (
    Compose as _PyCompose,
    Resize as _PyResize,
    RandomHorizontalFlip as _PyRandomHorizontalFlip,
    RandomVerticalFlip as _PyRandomVerticalFlip,
    RandomRotation as _PyRandomRotation,
    ColorJitter as _PyColorJitter,
    Normalize as _PyNormalize,
    ToTensor as _PyToTensor,
    get_default_augmentation as _py_get_default_augmentation,
    get_train_augmentation as _py_get_train_augmentation,
    get_val_augmentation as _py_get_val_augmentation,
)

from .exporter import ModelExporter as _PyModelExporter


# ============================================================================
# Select Backend
# ============================================================================

def is_cpp_backend_available() -> bool:
    """Check if C++ training backend is available."""
    return _USE_CPP_BACKEND


def get_backend() -> str:
    """Get the current backend name."""
    return "cpp" if _USE_CPP_BACKEND else "python"


# Export classes based on backend
if _USE_CPP_BACKEND:
    # Use C++ implementations
    BaseDataset = _CppIDataset
    ImageFolderDataset = _CppImageFolderDataset
    COCODataset = _CppCOCODataset
    YOLODataset = _CppYOLODataset
    split_dataset = _cpp_split_dataset

    Trainer = _CppTrainer

    TrainingCallback = _CppITrainingCallback
    EarlyStopping = _CppEarlyStopping
    ModelCheckpoint = _CppModelCheckpoint
    ProgressLogger = _CppProgressLogger
    LRScheduler = _CppLRScheduler
    TensorBoardLogger = _CppTensorBoardLogger

    Compose = _CppCompose
    Resize = _CppResize
    RandomHorizontalFlip = _CppRandomHorizontalFlip
    RandomVerticalFlip = _CppRandomVerticalFlip
    RandomRotation = _CppRandomRotation
    ColorJitter = _CppColorJitter
    Normalize = _CppNormalize
    ToTensor = _CppToTensor

    get_default_augmentation = _cpp_get_default_augmentation
    get_train_augmentation = _cpp_get_train_augmentation
    get_val_augmentation = _cpp_get_val_augmentation

    ModelExporter = _CppModelExporter

    # Additional C++ only exports
    TrainerConfig = _CppTrainerConfig
    ExportOptions = _CppExportOptions
    TrainingMetrics = _CppTrainingMetrics
    DetectionTarget = _CppDetectionTarget
    CenterCrop = _CppCenterCrop
    RandomCrop = _CppRandomCrop
    GaussianBlur = _CppGaussianBlur
    CSVLogger = _CppCSVLogger

else:
    # Use Python implementations
    BaseDataset = _PyBaseDataset
    ImageFolderDataset = _PyImageFolderDataset
    COCODataset = _PyCOCODataset
    YOLODataset = _PyYOLODataset
    split_dataset = _py_split_dataset

    Trainer = _PyTrainer

    TrainingCallback = _PyTrainingCallback
    EarlyStopping = _PyEarlyStopping
    ModelCheckpoint = _PyModelCheckpoint
    ProgressLogger = _PyProgressLogger
    LRScheduler = _PyLRScheduler
    TensorBoardLogger = _PyTensorBoardLogger

    Compose = _PyCompose
    Resize = _PyResize
    RandomHorizontalFlip = _PyRandomHorizontalFlip
    RandomVerticalFlip = _PyRandomVerticalFlip
    RandomRotation = _PyRandomRotation
    ColorJitter = _PyColorJitter
    Normalize = _PyNormalize
    ToTensor = _PyToTensor

    get_default_augmentation = _py_get_default_augmentation
    get_train_augmentation = _py_get_train_augmentation
    get_val_augmentation = _py_get_val_augmentation

    ModelExporter = _PyModelExporter


__all__ = [
    # Backend info
    "is_cpp_backend_available",
    "get_backend",
    # Datasets
    "BaseDataset",
    "ImageFolderDataset",
    "COCODataset",
    "YOLODataset",
    "split_dataset",
    # Trainer
    "Trainer",
    # Callbacks
    "TrainingCallback",
    "EarlyStopping",
    "ModelCheckpoint",
    "ProgressLogger",
    "LRScheduler",
    "TensorBoardLogger",
    # Augmentation
    "Compose",
    "Resize",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "ColorJitter",
    "Normalize",
    "ToTensor",
    "get_default_augmentation",
    "get_train_augmentation",
    "get_val_augmentation",
    # Exporter
    "ModelExporter",
]

# Add C++ only exports to __all__ if using C++ backend
if _USE_CPP_BACKEND:
    __all__.extend([
        "TrainerConfig",
        "ExportOptions",
        "TrainingMetrics",
        "DetectionTarget",
        "CenterCrop",
        "RandomCrop",
        "GaussianBlur",
        "CSVLogger",
    ])
