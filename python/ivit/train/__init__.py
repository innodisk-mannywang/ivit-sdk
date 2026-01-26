"""
Training module for iVIT-SDK.

Provides transfer learning capabilities for classification and detection models.
"""

from .dataset import (
    BaseDataset,
    ImageFolderDataset,
    COCODataset,
    YOLODataset,
    split_dataset,
)

from .trainer import Trainer

from .callbacks import (
    TrainingCallback,
    EarlyStopping,
    ModelCheckpoint,
    ProgressLogger,
    LRScheduler,
    TensorBoardLogger,
)

from .augmentation import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    ColorJitter,
    Normalize,
    ToTensor,
    get_default_augmentation,
    get_train_augmentation,
    get_val_augmentation,
)

from .exporter import ModelExporter

__all__ = [
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
