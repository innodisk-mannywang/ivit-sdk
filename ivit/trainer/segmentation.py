"""
Segmentation Trainer Module
===========================
Image segmentation training with popular architectures.
"""

import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

from ..core.base_trainer import BaseTrainer, TaskConfig


class SegmentationDataset(Dataset):
    """Custom dataset for segmentation tasks."""

    def __init__(self, dataset_path: str, transform=None, target_transform=None, split: str = 'train'):
        """
        Initialize segmentation dataset.

        Args:
            dataset_path: Path to dataset
            transform: Data transforms for images
            target_transform: Data transforms for masks
            split: Dataset split ('train', 'val', 'test')
        """
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        # Expected structure: dataset_path/images/split/ and dataset_path/masks/split/
        self.images_dir = self.dataset_path / 'images' / split
        self.masks_dir = self.dataset_path / 'masks' / split

        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.image_files = []

        if self.images_dir.exists():
            for ext in image_extensions:
                self.image_files.extend(list(self.images_dir.glob(f'*{ext}')))
                self.image_files.extend(list(self.images_dir.glob(f'*{ext.upper()}')))

        self.image_files.sort()

        print(f"✅ Segmentation dataset loaded:")
        print(f"   Images path: {self.images_dir}")
        print(f"   Masks path: {self.masks_dir}")
        print(f"   Samples: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')

        # Load corresponding mask
        mask_name = image_path.stem + '.png'  # Assume masks are PNG
        mask_path = self.masks_dir / mask_name

        if mask_path.exists():
            mask = Image.open(mask_path).convert('L')  # Grayscale mask
        else:
            # Create dummy mask if not found
            mask = Image.new('L', image.size, 0)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # Default mask transform
            mask = TF.to_tensor(mask).squeeze(0).long()

        return image, mask


class SegmentationConfig(TaskConfig):
    """Configuration for segmentation tasks."""

    def __init__(self,
                 model_name: str = 'deeplabv3_resnet50',
                 num_classes: int = 21,  # Pascal VOC default
                 pretrained: bool = True,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 img_size: int = 640,
                 **kwargs):
        """
        Initialize segmentation configuration.

        Args:
            model_name: Name of the segmentation model
            num_classes: Number of segmentation classes
            pretrained: Use pre-trained weights
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        super().__init__(**kwargs)

        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.img_size = img_size

        print(f"✅ SegmentationConfig initialized:")
        print(f"   Model: {model_name}")
        print(f"   Classes: {num_classes}")
        print(f"   Pretrained: {pretrained}")
        print(f"   Learning rate: {learning_rate}")

    def get_model(self) -> nn.Module:
        """Get the segmentation model."""
        if self.model_name == 'deeplabv3_resnet50':
            from torchvision.models.segmentation import deeplabv3_resnet50
            model = deeplabv3_resnet50(pretrained=self.pretrained)

            # Replace classifier for custom number of classes
            if self.num_classes != 21:  # 21 is Pascal VOC default
                model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)
                model.aux_classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)

        elif self.model_name == 'deeplabv3_resnet101':
            from torchvision.models.segmentation import deeplabv3_resnet101
            model = deeplabv3_resnet101(pretrained=self.pretrained)

            if self.num_classes != 21:
                model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)
                model.aux_classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)

        elif self.model_name == 'fcn_resnet50':
            from torchvision.models.segmentation import fcn_resnet50
            model = fcn_resnet50(pretrained=self.pretrained)

            if self.num_classes != 21:
                model.classifier[4] = nn.Conv2d(512, self.num_classes, kernel_size=1)
                model.aux_classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)

        elif self.model_name == 'unet':
            # Simple U-Net implementation
            model = self._create_unet()

        elif self.model_name.endswith('.pt') and 'yolov8' in self.model_name.lower():
            # YOLOv8 segmentation model
            try:
                from ultralytics import YOLO
                model = YOLO(self.model_name)
                print(f"✅ YOLOv8 segmentation model loaded: {self.model_name}")
                return model
            except ImportError:
                raise ImportError("ultralytics package is required for YOLOv8 models. Install with: pip install ultralytics")
            except Exception as e:
                raise RuntimeError(f"Failed to load YOLOv8 model {self.model_name}: {e}")

        else:
            raise ValueError(f"Unsupported segmentation model: {self.model_name}")

        print(f"✅ Segmentation model loaded: {self.model_name} (classes: {self.num_classes})")
        return model

    def _create_unet(self) -> nn.Module:
        """Create a simple U-Net model."""
        class UNet(nn.Module):
            def __init__(self, num_classes):
                super(UNet, self).__init__()

                # Encoder
                self.enc1 = self._conv_block(3, 64)
                self.enc2 = self._conv_block(64, 128)
                self.enc3 = self._conv_block(128, 256)
                self.enc4 = self._conv_block(256, 512)

                # Bottleneck
                self.bottleneck = self._conv_block(512, 1024)

                # Decoder
                self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
                self.dec4 = self._conv_block(1024, 512)
                self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
                self.dec3 = self._conv_block(512, 256)
                self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.dec2 = self._conv_block(256, 128)
                self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec1 = self._conv_block(128, 64)

                # Output
                self.final = nn.Conv2d(64, num_classes, 1)
                self.pool = nn.MaxPool2d(2)

            def _conv_block(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                # Encoder
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))

                # Bottleneck
                b = self.bottleneck(self.pool(e4))

                # Decoder
                d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
                d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
                d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
                d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

                return {'out': self.final(d1)}

        return UNet(self.num_classes)

    def get_loss_function(self) -> nn.Module:
        """Get the loss function for segmentation."""
        return nn.CrossEntropyLoss(ignore_index=255)  # 255 typically ignored in segmentation

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Get the optimizer for segmentation."""
        return optim.AdamW(model.parameters(),
                          lr=self.learning_rate,
                          weight_decay=self.weight_decay)

    def get_scheduler(self, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get the learning rate scheduler."""
        return optim.lr_scheduler.PolynomialLR(optimizer, total_iters=100, power=0.9)

    def get_dataloader(self, dataset_path: str, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
        """Get the dataloader for segmentation."""
        # Data transforms
        if shuffle:  # Training transforms
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # Validation transforms
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Target transforms for masks
        target_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        # Create dataset
        split = 'train' if shuffle else 'val'
        dataset = SegmentationDataset(
            dataset_path, 
            transform=transform, 
            target_transform=target_transform, 
            split=split
        )

        # Create dataloader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )

    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute segmentation metrics."""
        # Get predictions
        if isinstance(outputs, dict):
            outputs = outputs['out']  # For torchvision models

        predictions = torch.argmax(outputs, dim=1)

        # Calculate pixel accuracy
        correct = (predictions == targets).sum().item()
        total = targets.numel()
        pixel_accuracy = correct / total

        # Calculate mean IoU
        num_classes = outputs.size(1)
        iou_per_class = []

        for class_id in range(num_classes):
            pred_mask = (predictions == class_id)
            target_mask = (targets == class_id)

            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()

            if union > 0:
                iou = intersection / union
                iou_per_class.append(iou)

        mean_iou = np.mean(iou_per_class) if iou_per_class else 0.0

        return {
            'pixel_accuracy': pixel_accuracy,
            'mean_iou': mean_iou,
            'num_classes_present': len(iou_per_class)
        }


class SegmentationTrainer(BaseTrainer):
    """Trainer specifically for image segmentation tasks."""

    def __init__(self,
                 model_name: str = 'deeplabv3_resnet50',
                 num_classes: int = 21,
                 pretrained: bool = True,
                 learning_rate: float = 0.001,
                 img_size: int = 640,
                 device: str = "auto",
                 **kwargs):
        """
        Initialize SegmentationTrainer.

        Args:
            model_name: Name of the segmentation model
            num_classes: Number of segmentation classes
            pretrained: Use pre-trained weights
            learning_rate: Learning rate for optimizer
            device: Device to use for training
        """
        # Create configuration
        config = SegmentationConfig(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            learning_rate=learning_rate,
            img_size=img_size,
            **kwargs
        )

        # Initialize base trainer
        super().__init__(config, device=device)

        self.model_name = model_name
        print(f"🎯 SegmentationTrainer initialized with {model_name}")

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with segmentation-specific handling."""
        self.model.train()

        total_loss = 0.0
        total_samples = 0
        all_outputs = []
        all_targets = []

        from tqdm import tqdm
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Handle different output formats
            if isinstance(outputs, dict):
                main_loss = self.criterion(outputs['out'], targets)
                if 'aux' in outputs:
                    aux_loss = self.criterion(outputs['aux'], targets)
                    loss = main_loss + 0.4 * aux_loss  # Standard auxiliary loss weight
                else:
                    loss = main_loss
            else:
                loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update statistics
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Store outputs and targets for metrics calculation
            with torch.no_grad():
                if isinstance(outputs, dict):
                    all_outputs.append(outputs['out'].detach().cpu())
                else:
                    all_outputs.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / total_samples:.4f}'
            })

        # Compute metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.task_config.compute_metrics(all_outputs, all_targets)
        metrics['loss'] = total_loss / total_samples

        return metrics

    def get_recommendations(self, dataset_path: str) -> Dict[str, Any]:
        """Get intelligent recommendations for segmentation dataset."""
        from ..utils.smart_recommendation import SmartRecommendationEngine
        from ..utils.dataset_analyzer import DatasetAnalyzer

        # Analyze dataset
        analyzer = DatasetAnalyzer()
        stats = analyzer.extract_segmentation_statistics(dataset_path)

        # Get recommendations
        recommendation_engine = SmartRecommendationEngine()
        recommendations = recommendation_engine.get_segmentation_recommendations(stats)

        return recommendations

    def apply_recommendations(self, recommendations: Dict[str, Any]):
        """Apply intelligent recommendations to the segmentation trainer."""
        print("🧠 Applying segmentation recommendations...")

        # Update model if recommended
        if 'model' in recommendations:
            recommended_model = recommendations['model']
            if recommended_model != self.model_name:
                print(f"📝 Updating model: {self.model_name} → {recommended_model}")
                self.task_config.model_name = recommended_model
                self.model_name = recommended_model

        # Update number of classes if recommended
        if 'num_classes' in recommendations:
            recommended_classes = recommendations['num_classes']
            print(f"📝 Updating classes: {self.task_config.num_classes} → {recommended_classes}")
            self.task_config.num_classes = recommended_classes

        # Update learning rate if recommended
        if 'learning_rate' in recommendations:
            recommended_lr = recommendations['learning_rate']
            print(f"📝 Updating learning rate: {self.task_config.learning_rate} → {recommended_lr}")
            self.task_config.learning_rate = recommended_lr

        print("✅ Segmentation recommendations applied!")

    def train(self, dataset_path: str, epochs: int = 100, batch_size: int = 32, 
              validation_split: float = 0.2, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the segmentation model.
        
        For YOLOv8 models, use YOLO's built-in training.
        For other models, use the standard PyTorch training loop.
        """
        # Check if this is a YOLOv8 model
        if self.model_name.endswith('.pt') and 'yolov8' in self.model_name.lower():
            return self._train_yolo(dataset_path, epochs, batch_size, save_path)
        else:
            # Use standard PyTorch training
            return super().train(dataset_path, epochs, batch_size, validation_split, save_path)

    def _train_yolo(self, dataset_path: str, epochs: int, batch_size: int, save_path: Optional[str]) -> Dict[str, Any]:
        """Train YOLOv8 segmentation model using ultralytics."""
        print("🚀 Starting YOLOv8 segmentation training...")
        
        # Get the YOLO model
        yolo_model = self.task_config.get_model()
        
        # Create YOLO config
        yolo_config = {
            'data': str(Path(dataset_path) / 'data.yaml'),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': self.task_config.img_size,
            'device': self.device,
            'project': 'models',
            'name': f'segmentation_{self.model_name.replace(".pt", "")}',
            'save': True,
            'save_period': 10,
            'patience': 50,
            'lr0': self.task_config.learning_rate,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 2.0,
            'mask_ratio': 4,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'verbose': True
        }
        
        print(f"📊 YOLO training config:")
        for key, value in yolo_config.items():
            print(f"   {key}: {value}")
        
        # Start training
        try:
            results = yolo_model.train(**yolo_config)
            
            # Extract metrics from results
            final_metrics = {
                'map50': results.results_dict.get('metrics/mAP50(M)', 0.0),
                'map50_95': results.results_dict.get('metrics/mAP50-95(M)', 0.0),
                'precision': results.results_dict.get('metrics/precision(M)', 0.0),
                'recall': results.results_dict.get('metrics/recall(M)', 0.0),
                'box_loss': results.results_dict.get('train/box_loss', 0.0),
                'seg_loss': results.results_dict.get('train/seg_loss', 0.0),
                'cls_loss': results.results_dict.get('train/cls_loss', 0.0),
                'dfl_loss': results.results_dict.get('train/dfl_loss', 0.0)
            }
            
            # Get model save path
            model_save_dir = results.save_dir
            best_model_path = Path(model_save_dir) / 'best.pt'
            
            print(f"\n🎉 YOLOv8 segmentation training completed!")
            print(f"📊 Final metrics:")
            for metric, value in final_metrics.items():
                print(f"   {metric}: {value:.4f}")
            
            if best_model_path.exists():
                print(f"💾 Best model saved to: {best_model_path}")
            
            return {
                'final_metrics': final_metrics,
                'model_path': str(model_save_dir),
                'best_model_path': str(best_model_path) if best_model_path.exists() else None,
                'yolo_results': results
            }
            
        except Exception as e:
            print(f"❌ YOLOv8 training failed: {str(e)}")
            raise e

    def validate_dataset_format(self, dataset_path: str) -> bool:
        """
        Validate that the dataset follows the expected YOLO format for segmentation.
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            True if dataset format is valid, False otherwise
        """
        dataset_path = Path(dataset_path)
        
        # Check if dataset directory exists
        if not dataset_path.exists():
            print(f"❌ Dataset directory does not exist: {dataset_path}")
            return False
        
        # Check for required directories
        required_dirs = ['images', 'labels']
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                print(f"❌ Missing required directory: {dir_path}")
                return False
        
        # Check for train/val splits in images and labels
        for split in ['train', 'val']:
            images_split = dataset_path / 'images' / split
            labels_split = dataset_path / 'labels' / split
            
            if not images_split.exists():
                print(f"❌ Missing images/{split} directory")
                return False
            
            if not labels_split.exists():
                print(f"❌ Missing labels/{split} directory")
                return False
            
            # Check if there are any image files
            image_files = list(images_split.glob('*'))
            if not image_files:
                print(f"❌ No image files found in {images_split}")
                return False
        
        # Check for data.yaml
        data_yaml = dataset_path / 'data.yaml'
        if not data_yaml.exists():
            print(f"❌ Missing data.yaml file")
            return False
        
        # Try to read data.yaml
        try:
            import yaml
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
            
            # Check required fields
            required_fields = ['train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in data:
                    print(f"❌ Missing required field '{field}' in data.yaml")
                    return False
                    
        except Exception as e:
            print(f"❌ Error reading data.yaml: {e}")
            return False
        
        print("✅ Dataset format validation passed")
        return True
