"""
Classification Trainer Module
============================
Image classification training with popular CNN architectures.
"""

import os
from typing import Dict, Any, Optional, Tuple, List, Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, models
import torchvision.models as models_module
from PIL import Image
import numpy as np

from ..core.base_trainer import BaseTrainer, TaskConfig


class ClassificationDataset(Dataset):
    """Custom dataset for classification tasks."""

    def __init__(self, dataset_path: str, transform=None, split: str = 'train', verbose: bool = True):
        """
        Initialize classification dataset.

        Args:
            dataset_path: Path to dataset (ImageFolder structure expected)
            transform: Data transforms to apply
            split: Dataset split ('train', 'val', 'test')
            verbose: Whether to show progress bars and standard output
        """
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.split = split
        self.verbose = verbose

        # Use torchvision ImageFolder for standard directory structure
        split_path = self.dataset_path / split if (self.dataset_path / split).exists() else self.dataset_path
        self.dataset = datasets.ImageFolder(root=str(split_path), transform=transform)

        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)

        if self.verbose:
            print(f"✅ Classification dataset loaded:")
            print(f"   Path: {split_path}")
            print(f"   Classes: {self.num_classes}")
            print(f"   Samples: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class ClassificationConfig(TaskConfig):
    """Configuration for classification tasks."""

    def __init__(self, 
                 model_name: str = 'resnet18',
                 num_classes: int = None,
                 pretrained: bool = True,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize classification configuration.

        Args:
            model_name: Name of the model architecture
            num_classes: Number of classes (auto-detected if None)
            pretrained: Use pre-trained weights
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            verbose: Whether to show progress bars and standard output
        """
        super().__init__(**kwargs)

        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.verbose = verbose

        # Store for auto-detection
        self._detected_classes = None

        if self.verbose:
            print(f"✅ ClassificationConfig initialized:")
            print(f"   Model: {model_name}")
            print(f"   Pretrained: {pretrained}")
            print(f"   Learning rate: {learning_rate}")

    def get_model(self) -> nn.Module:
        """Get the classification model."""
        # Auto-detect number of classes if not provided
        if self.num_classes is None and self._detected_classes is not None:
            self.num_classes = self._detected_classes
        elif self.num_classes is None:
            self.num_classes = 1000  # Default ImageNet classes

        # Load pre-trained model
        if self.model_name == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_name == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=self.pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        elif self.model_name == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(pretrained=self.pretrained)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, self.num_classes)
        elif self.model_name == 'vit_b_16':
            model = models.vit_b_16(pretrained=self.pretrained)
            model.heads.head = nn.Linear(model.heads.head.in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        if self.verbose:
            print(f"✅ Model loaded: {self.model_name} (classes: {self.num_classes})")
        return model

    def get_loss_function(self) -> nn.Module:
        """Get the loss function for classification."""
        return nn.CrossEntropyLoss()

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Get the optimizer for classification."""
        return optim.AdamW(model.parameters(), 
                          lr=self.learning_rate, 
                          weight_decay=self.weight_decay)

    def get_scheduler(self, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get the learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    def get_dataloader(self, dataset_path: str, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Get the dataloader for classification."""
        # Data transforms
        if shuffle:  # Training transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # Validation transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Create dataset
        split = 'train' if shuffle else 'val'
        dataset = ClassificationDataset(dataset_path, transform=transform, split=split, verbose=self.verbose)

        # Auto-detect number of classes
        if self._detected_classes is None:
            self._detected_classes = dataset.num_classes
            self.num_classes = dataset.num_classes

        # Setup DistributedSampler if in distributed mode
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            effective_shuffle = False
        else:
            effective_shuffle = shuffle

        # Create dataloader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=effective_shuffle,
            sampler=sampler,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )

    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute classification metrics."""
        # Get predictions
        _, predicted = torch.max(outputs, 1)

        # Calculate accuracy
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        accuracy = correct / total

        # Calculate top-5 accuracy if we have enough classes
        metrics = {'accuracy': accuracy}

        if outputs.size(1) >= 5:
            _, top5_pred = torch.topk(outputs, 5, dim=1)
            top5_correct = 0
            for i in range(total):
                if targets[i] in top5_pred[i]:
                    top5_correct += 1
            metrics['top5_accuracy'] = top5_correct / total

        return metrics


class ClassificationTrainer(BaseTrainer):
    """Trainer specifically for image classification tasks."""

    def __init__(self, 
                 model_name: str = 'resnet18',
                 num_classes: int = None,
                 pretrained: bool = True,
                 learning_rate: float = 0.001,
                 device: str = "auto",
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize ClassificationTrainer.

        Args:
            model_name: Name of the model architecture
            num_classes: Number of classes (auto-detected if None)
            pretrained: Use pre-trained weights
            learning_rate: Learning rate for optimizer
            device: Device to use for training
            verbose: Whether to show progress bars and standard output
        """
        # Create configuration
        config = ClassificationConfig(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            learning_rate=learning_rate,
            **kwargs
        )

        # Initialize base trainer
        super().__init__(config, device=device, verbose=verbose)

        self.model_name = model_name
        if self.verbose:
            print(f"🎯 ClassificationTrainer initialized with {model_name}")

    def get_recommendations(self, dataset_path: str) -> Dict[str, Any]:
        """Get intelligent recommendations for this dataset."""
        from ..utils.smart_recommendation import SmartRecommendationEngine
        from ..utils.dataset_analyzer import DatasetAnalyzer

        # Analyze dataset
        analyzer = DatasetAnalyzer()
        stats = analyzer.extract_statistics(dataset_path)

        # Get recommendations
        recommendation_engine = SmartRecommendationEngine()
        recommendations = recommendation_engine.get_recommendations(stats)

        return recommendations

    def apply_recommendations(self, recommendations: Dict[str, Any]):
        """Apply intelligent recommendations to the trainer."""
        print("🧠 Applying intelligent recommendations...")

        # Update model if recommended
        if 'model' in recommendations:
            recommended_model = recommendations['model']
            if recommended_model != self.model_name:
                print(f"📝 Updating model: {self.model_name} → {recommended_model}")
                self.task_config.model_name = recommended_model
                self.model_name = recommended_model

        # Update learning rate if recommended
        if 'learning_rate' in recommendations:
            recommended_lr = recommendations['learning_rate']
            print(f"📝 Updating learning rate: {self.task_config.learning_rate} → {recommended_lr}")
            self.task_config.learning_rate = recommended_lr

        # Update batch size if recommended
        if 'batch_size' in recommendations:
            recommended_batch = recommendations['batch_size']
            print(f"📝 Recommended batch size: {recommended_batch}")

        print("✅ Recommendations applied!")

    def train(self, 
              dataset_path: str,
              epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.2,
              save_path: Optional[str] = None,
              callbacks: Optional[Dict[str, List[Callable[[Dict[str, Any]], None]]]] = None,
              progress_log_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop with automatic class_names.json generation.
        
        Args:
            dataset_path: Path to the dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            save_path: Path to save the trained model
            callbacks: Optional callbacks for training events
            progress_log_path: Optional path for progress logging
            
        Returns:
            Training history and final metrics
        """
        # 先載入資料集以獲取類別名稱
        print("🔍 載入資料集以獲取類別資訊...")
        temp_dataset = ClassificationDataset(dataset_path, split='train')
        class_names = temp_dataset.classes
        self.task_config._detected_classes = len(class_names)
        
        print(f"✅ 檢測到 {len(class_names)} 個類別:")
        for i, class_name in enumerate(class_names):
            print(f"   {i}: {class_name}")
        
        # 執行標準訓練流程
        results = super().train(
            dataset_path=dataset_path,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            save_path=save_path,
            callbacks=callbacks,
            progress_log_path=progress_log_path
        )
        
        # 自動產生 class_names.json
        if save_path:
            self._save_class_names(save_path, class_names)
        
        return results

    def _save_class_names(self, model_path: str, class_names: List[str]):
        """Save class names to JSON file alongside the model."""
        import json
        
        # 創建 class_names.json 檔案路徑
        model_dir = os.path.dirname(model_path)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        class_names_path = os.path.join(model_dir, f"{model_name}_class_names.json")
        
        # 保存類別名稱
        try:
            with open(class_names_path, 'w', encoding='utf-8') as f:
                json.dump(class_names, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 類別名稱已保存至: {class_names_path}")
            print(f"   包含 {len(class_names)} 個類別")
            
        except Exception as e:
            print(f"⚠️ 保存類別名稱失敗: {e}")
