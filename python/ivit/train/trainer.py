"""
Trainer class for transfer learning.

Provides a simple interface for fine-tuning pre-trained models.
"""

from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
import numpy as np
import logging
import time

from .dataset import BaseDataset
from .callbacks import TrainingCallback, ProgressLogger
from .augmentation import get_train_augmentation, get_val_augmentation

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for fine-tuning models with transfer learning.

    Supports classification and detection models with a simple API.

    Args:
        model: Model to train (can be a model name or initialized model)
        dataset: Training dataset
        val_dataset: Validation dataset (optional, can use train_split)
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        optimizer: Optimizer type ("adam", "sgd", "adamw")
        device: Training device ("cuda:0", "cpu")
        freeze_backbone: Freeze backbone layers for transfer learning
        num_workers: Number of data loader workers

    Examples:
        >>> from ivit.train import Trainer, ImageFolderDataset
        >>>
        >>> dataset = ImageFolderDataset("./data", train_split=0.8)
        >>> trainer = Trainer(
        ...     model="resnet50",
        ...     dataset=dataset,
        ...     epochs=20,
        ...     learning_rate=0.001,
        ...     device="cuda:0",
        ... )
        >>> trainer.fit()
        >>> metrics = trainer.evaluate()
        >>> trainer.export("model.onnx")
    """

    def __init__(
        self,
        model: Union[str, Any],
        dataset: BaseDataset,
        val_dataset: Optional[BaseDataset] = None,
        epochs: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        optimizer: str = "adam",
        device: str = "cuda:0",
        freeze_backbone: bool = True,
        num_workers: int = 4,
        **kwargs
    ):
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer_type = optimizer
        self.device_str = device
        self.freeze_backbone = freeze_backbone
        self.num_workers = num_workers
        self.config = kwargs

        # Internal state
        self._stop_training = False
        self._history: List[Dict[str, float]] = []
        self._current_epoch = 0
        self._batches_per_epoch = 0

        # Initialize PyTorch components
        self._init_torch()
        self._init_model(model)
        self._init_optimizer()
        self._init_dataloaders()

    def _init_torch(self):
        """Initialize PyTorch and check CUDA."""
        try:
            import torch
            self._torch = torch

            # Set device
            if self.device_str.startswith("cuda"):
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    self.device_str = "cpu"

            self.device = torch.device(self.device_str)
            logger.info(f"Using device: {self.device}")

        except ImportError:
            raise ImportError(
                "PyTorch is required for training. "
                "Install with: pip install torch torchvision"
            )

    def _init_model(self, model: Union[str, Any]):
        """Initialize model for training."""
        import torch
        import torch.nn as nn

        if isinstance(model, str):
            # Load pre-trained model from torchvision
            self.model = self._load_pretrained_model(model)
        else:
            self.model = model

        # Modify output layer for dataset classes
        num_classes = self.dataset.num_classes
        self._modify_classifier(num_classes)

        # Freeze backbone if requested
        if self.freeze_backbone:
            self._freeze_backbone_layers()

        self.model = self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    def _load_pretrained_model(self, model_name: str):
        """Load a pre-trained model from torchvision."""
        import torchvision.models as models

        model_name = model_name.lower()

        # Map model names to torchvision functions
        model_map = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
            "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
            "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
            "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
            "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
            "vgg19": (models.vgg19, models.VGG19_Weights.DEFAULT),
            "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
        }

        if model_name not in model_map:
            available = ", ".join(model_map.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

        model_fn, weights = model_map[model_name]
        logger.info(f"Loading pre-trained {model_name}")
        return model_fn(weights=weights)

    def _modify_classifier(self, num_classes: int):
        """Modify the classifier head for the target number of classes."""
        import torch.nn as nn

        model = self.model

        # Handle different model architectures
        if hasattr(model, 'fc'):
            # ResNet, DenseNet
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            logger.debug(f"Modified fc layer: {in_features} -> {num_classes}")

        elif hasattr(model, 'classifier'):
            classifier = model.classifier
            if isinstance(classifier, nn.Sequential):
                # VGG, EfficientNet
                last_layer = classifier[-1]
                if isinstance(last_layer, nn.Linear):
                    in_features = last_layer.in_features
                    classifier[-1] = nn.Linear(in_features, num_classes)
            elif isinstance(classifier, nn.Linear):
                # MobileNet
                in_features = classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
            logger.debug(f"Modified classifier layer for {num_classes} classes")

        elif hasattr(model, 'heads'):
            # Vision Transformer
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
            logger.debug(f"Modified ViT head: {in_features} -> {num_classes}")

        else:
            logger.warning("Could not automatically modify classifier - unknown architecture")

    def _freeze_backbone_layers(self):
        """Freeze backbone layers for transfer learning."""
        import torch.nn as nn

        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze classifier
        if hasattr(self.model, 'fc'):
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'heads'):
            for param in self.model.heads.parameters():
                param.requires_grad = True

        logger.info("Froze backbone layers, only training classifier")

    def _init_optimizer(self):
        """Initialize optimizer."""
        import torch.optim as optim

        # Get trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]

        if self.optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(params, lr=self.learning_rate)
        elif self.optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(params, lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        logger.info(f"Optimizer: {self.optimizer_type}, lr={self.learning_rate}")

    def _init_dataloaders(self):
        """Initialize data loaders."""
        from torch.utils.data import DataLoader

        # Create PyTorch dataset wrappers
        train_dataset = _TorchDatasetWrapper(
            self.dataset,
            transform=get_train_augmentation(224),
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device_str.startswith("cuda") else False,
        )

        self._batches_per_epoch = len(self.train_loader)

        # Validation loader
        if self.val_dataset is not None:
            val_dataset = _TorchDatasetWrapper(
                self.val_dataset,
                transform=get_val_augmentation(224),
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        else:
            self.val_loader = None

        logger.info(f"Train batches: {len(self.train_loader)}")

    @property
    def current_epoch(self) -> int:
        """Get current training epoch."""
        return self._current_epoch

    @property
    def batches_per_epoch(self) -> int:
        """Get number of batches per epoch."""
        return self._batches_per_epoch

    @property
    def history(self) -> List[Dict[str, float]]:
        """Get training history."""
        return self._history

    def fit(
        self,
        callbacks: Optional[List[TrainingCallback]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            callbacks: List of training callbacks

        Returns:
            Dictionary of training history

        Examples:
            >>> history = trainer.fit(callbacks=[
            ...     EarlyStopping(patience=5),
            ...     ModelCheckpoint("best.pt"),
            ... ])
        """
        import torch
        import torch.nn as nn

        callbacks = callbacks or [ProgressLogger()]
        criterion = nn.CrossEntropyLoss()

        # Trigger on_train_start
        for cb in callbacks:
            cb.on_train_start(self)

        logger.info(f"Starting training for {self.epochs} epochs")
        start_time = time.time()

        for epoch in range(self.epochs):
            if self._stop_training:
                logger.info("Training stopped early")
                break

            self._current_epoch = epoch

            # Trigger on_epoch_start
            for cb in callbacks:
                cb.on_epoch_start(self, epoch)

            # Training phase
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                # Trigger on_batch_start
                for cb in callbacks:
                    cb.on_batch_start(self, batch_idx)

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Track metrics
                batch_loss = loss.item()
                epoch_loss += batch_loss

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Trigger on_batch_end
                for cb in callbacks:
                    cb.on_batch_end(self, batch_idx, batch_loss)

            # Calculate epoch metrics
            avg_loss = epoch_loss / len(self.train_loader)
            accuracy = 100. * correct / total

            metrics = {
                'loss': avg_loss,
                'accuracy': accuracy,
            }

            # Validation phase
            if self.val_loader is not None:
                val_metrics = self._validate(criterion)
                metrics.update(val_metrics)

            self._history.append(metrics)

            # Trigger on_epoch_end
            for cb in callbacks:
                cb.on_epoch_end(self, epoch, metrics)

        # Trigger on_train_end
        for cb in callbacks:
            cb.on_train_end(self)

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.1f}s")

        return self._history

    def _validate(self, criterion) -> Dict[str, float]:
        """Run validation."""
        import torch

        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return {
            'val_loss': val_loss / len(self.val_loader),
            'val_accuracy': 100. * correct / total,
        }

    def evaluate(
        self,
        dataset: Optional[BaseDataset] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            dataset: Dataset to evaluate on (default: validation set)

        Returns:
            Dictionary of metrics

        Examples:
            >>> metrics = trainer.evaluate()
            >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        if dataset is not None:
            eval_dataset = _TorchDatasetWrapper(
                dataset,
                transform=get_val_augmentation(224),
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )
        elif self.val_loader is not None:
            eval_loader = self.val_loader
        else:
            raise ValueError("No validation dataset available")

        criterion = nn.CrossEntropyLoss()
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in eval_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        avg_loss = total_loss / len(eval_loader)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }

    def export(
        self,
        path: str,
        format: str = "onnx",
        optimize_for: Optional[str] = None,
        quantize: Optional[str] = None,
        input_shape: tuple = (1, 3, 224, 224),
        **kwargs
    ) -> str:
        """
        Export trained model to deployment format.

        Args:
            path: Output path
            format: Export format ("onnx", "torchscript")
            optimize_for: Target hardware ("intel_cpu", "intel_npu", "nvidia_gpu")
            quantize: Quantization mode ("fp16", "int8")
            input_shape: Input tensor shape

        Returns:
            Path to exported model

        Examples:
            >>> trainer.export("model.onnx", format="onnx", quantize="fp16")
        """
        from .exporter import ModelExporter

        exporter = ModelExporter(self.model, self.device)
        return exporter.export(
            path=path,
            format=format,
            optimize_for=optimize_for,
            quantize=quantize,
            input_shape=input_shape,
            class_names=self.dataset.class_names,
            **kwargs
        )


class _TorchDatasetWrapper:
    """Wrapper to make iVIT dataset compatible with PyTorch DataLoader."""

    def __init__(self, dataset: BaseDataset, transform: Optional[Callable] = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        import torch

        image, label = self.dataset[idx]

        if self.transform is not None:
            image = self.transform(image)

        # Convert to torch tensor if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Remove batch dimension if present
        if image.dim() == 4:
            image = image.squeeze(0)

        return image, label
