"""
Smart Recommendation Engine Module
=================================
Intelligent parameter recommendations based on dataset analysis.
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class SmartRecommendationEngine:
    """Generate intelligent recommendations for AI vision training."""

    def __init__(self):
        """Initialize SmartRecommendationEngine."""
        print("🧠 SmartRecommendationEngine initialized")

    def get_recommendations(self, dataset_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive recommendations for classification tasks.

        Args:
            dataset_stats: Statistics from DatasetAnalyzer

        Returns:
            Dictionary containing recommendations
        """
        print("🔍 Generating intelligent recommendations...")

        recommendations = {
            'model': self._recommend_classification_model(dataset_stats),
            'learning_rate': self._recommend_learning_rate(dataset_stats),
            'batch_size': self._recommend_batch_size(dataset_stats),
            'epochs': self._recommend_epochs(dataset_stats),
            'optimizer': self._recommend_optimizer(dataset_stats),
            'data_augmentation': self._recommend_augmentation(dataset_stats),
            'regularization': self._recommend_regularization(dataset_stats),
            'reasoning': {}
        }

        # Add reasoning for each recommendation
        recommendations['reasoning'] = self._generate_reasoning(dataset_stats, recommendations)

        print(f"✅ Recommendations generated:")
        print(f"   Model: {recommendations['model']}")
        print(f"   Learning rate: {recommendations['learning_rate']}")
        print(f"   Batch size: {recommendations['batch_size']}")
        print(f"   Epochs: {recommendations['epochs']}")

        return recommendations

    def get_detection_recommendations(self, dataset_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for object detection tasks."""
        print("🔍 Generating detection recommendations...")

        recommendations = {
            'model': self._recommend_detection_model(dataset_stats),
            'img_size': self._recommend_image_size(dataset_stats),
            'learning_rate': self._recommend_detection_lr(dataset_stats),
            'batch_size': self._recommend_detection_batch_size(dataset_stats),
            'epochs': self._recommend_detection_epochs(dataset_stats),
            'augmentation': self._recommend_detection_augmentation(dataset_stats),
            'reasoning': {}
        }

        recommendations['reasoning'] = self._generate_detection_reasoning(dataset_stats, recommendations)

        print(f"✅ Detection recommendations generated:")
        print(f"   Model: {recommendations['model']}")
        print(f"   Image size: {recommendations['img_size']}")
        print(f"   Learning rate: {recommendations['learning_rate']}")

        return recommendations

    def get_segmentation_recommendations(self, dataset_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for segmentation tasks."""
        print("🔍 Generating segmentation recommendations...")

        recommendations = {
            'model': self._recommend_segmentation_model(dataset_stats),
            'learning_rate': self._recommend_segmentation_lr(dataset_stats),
            'batch_size': self._recommend_segmentation_batch_size(dataset_stats),
            'epochs': self._recommend_segmentation_epochs(dataset_stats),
            'num_classes': dataset_stats.get('num_classes', 21),
            'augmentation': self._recommend_segmentation_augmentation(dataset_stats),
            'reasoning': {}
        }

        recommendations['reasoning'] = self._generate_segmentation_reasoning(dataset_stats, recommendations)

        print(f"✅ Segmentation recommendations generated:")
        print(f"   Model: {recommendations['model']}")
        print(f"   Classes: {recommendations['num_classes']}")

        return recommendations

    def _recommend_classification_model(self, stats: Dict[str, Any]) -> str:
        """Recommend classification model based on dataset characteristics."""
        num_classes = stats.get('num_classes', 10)
        total_samples = stats.get('total_samples', 1000)
        complexity_score = stats.get('complexity_score', 0.5)
        mean_width = stats.get('image_stats', {}).get('mean_width', 224)
        mean_height = stats.get('image_stats', {}).get('mean_height', 224)

        # Consider image resolution
        avg_resolution = (mean_width + mean_height) / 2

        # For very small datasets, use lightweight models
        if total_samples < 500:
            return 'mobilenet_v3_small'

        # For small images or limited compute, use efficient models
        if avg_resolution < 128 or complexity_score < 0.3:
            return 'mobilenet_v3_small'

        # For medium datasets, use balanced models
        if total_samples < 5000:
            if num_classes <= 10:
                return 'efficientnet_b0'
            else:
                return 'resnet18'

        # For large datasets or high complexity
        if complexity_score > 0.7 or num_classes > 100:
            if avg_resolution > 300:
                return 'efficientnet_b2'
            else:
                return 'resnet50'

        # Default balanced choice
        return 'efficientnet_b0'

    def _recommend_detection_model(self, stats: Dict[str, Any]) -> str:
        """Recommend YOLO model for detection."""
        total_images = stats.get('total_images', 1000)
        complexity_score = stats.get('complexity_score', 0.5)
        mean_boxes = stats.get('bbox_stats', {}).get('mean_boxes_per_image', 1)

        # For small datasets or simple scenes
        if total_images < 1000 or complexity_score < 0.3:
            return 'yolov8n.pt'

        # For medium datasets
        if total_images < 5000:
            if mean_boxes <= 5:
                return 'yolov8s.pt'
            else:
                return 'yolov8m.pt'  # More boxes need more capacity

        # For large datasets or complex scenes
        if complexity_score > 0.7 or mean_boxes > 10:
            return 'yolov8l.pt'

        # Default medium model
        return 'yolov8s.pt'

    def _recommend_segmentation_model(self, stats: Dict[str, Any]) -> str:
        """Recommend segmentation model."""
        num_classes = stats.get('num_classes', 21)
        total_samples = stats.get('total_samples', 1000)
        complexity_score = stats.get('complexity_score', 0.5)

        # For simple datasets
        if complexity_score < 0.3 or total_samples < 1000:
            return 'fcn_resnet50'

        # For fine-grained segmentation
        if num_classes > 50 or complexity_score > 0.7:
            return 'deeplabv3_resnet101'

        # Default balanced choice
        return 'deeplabv3_resnet50'

    def _recommend_learning_rate(self, stats: Dict[str, Any]) -> float:
        """Recommend learning rate based on dataset size and complexity."""
        total_samples = stats.get('total_samples', 1000)
        complexity_score = stats.get('complexity_score', 0.5)

        # Base learning rate
        base_lr = 0.001

        # Adjust for dataset size
        if total_samples < 500:
            base_lr *= 0.5  # Smaller LR for small datasets
        elif total_samples > 10000:
            base_lr *= 2.0  # Larger LR for big datasets

        # Adjust for complexity
        if complexity_score > 0.7:
            base_lr *= 0.7  # Lower LR for complex datasets
        elif complexity_score < 0.3:
            base_lr *= 1.5  # Higher LR for simple datasets

        # Clamp to reasonable range
        return max(1e-5, min(0.1, base_lr))

    def _recommend_detection_lr(self, stats: Dict[str, Any]) -> float:
        """Recommend learning rate for detection."""
        base_lr = 0.01  # YOLO typically uses higher LR

        total_images = stats.get('total_images', 1000)
        complexity_score = stats.get('complexity_score', 0.5)

        # Adjust for dataset size
        if total_images < 1000:
            base_lr *= 0.5
        elif total_images > 10000:
            base_lr *= 1.5

        # Adjust for complexity
        if complexity_score > 0.7:
            base_lr *= 0.7

        return max(1e-4, min(0.1, base_lr))

    def _recommend_segmentation_lr(self, stats: Dict[str, Any]) -> float:
        """Recommend learning rate for segmentation."""
        base_lr = 0.001

        total_samples = stats.get('total_samples', 1000)
        num_classes = stats.get('num_classes', 21)

        # Segmentation often needs lower LR
        if num_classes > 50:
            base_lr *= 0.5

        if total_samples < 1000:
            base_lr *= 0.7

        return max(1e-5, min(0.01, base_lr))

    def _recommend_batch_size(self, stats: Dict[str, Any]) -> int:
        """Recommend batch size based on dataset and typical GPU memory."""
        mean_width = stats.get('image_stats', {}).get('mean_width', 224)
        mean_height = stats.get('image_stats', {}).get('mean_height', 224)

        # Estimate memory usage (rough approximation)
        pixel_count = mean_width * mean_height * 3  # RGB

        # Base batch size for typical GPU (8GB)
        if pixel_count < 224 * 224 * 3:
            return 32
        elif pixel_count < 512 * 512 * 3:
            return 16
        elif pixel_count < 1024 * 1024 * 3:
            return 8
        else:
            return 4

    def _recommend_detection_batch_size(self, stats: Dict[str, Any]) -> int:
        """Recommend batch size for detection."""
        mean_width = stats.get('image_stats', {}).get('mean_width', 640)
        mean_height = stats.get('image_stats', {}).get('mean_height', 640)

        # Detection models are memory-intensive
        avg_size = (mean_width + mean_height) / 2

        if avg_size <= 416:
            return 16
        elif avg_size <= 640:
            return 8
        else:
            return 4

    def _recommend_segmentation_batch_size(self, stats: Dict[str, Any]) -> int:
        """Recommend batch size for segmentation."""
        # Segmentation models are very memory-intensive
        mean_width = stats.get('image_stats', {}).get('mean_width', 512)
        mean_height = stats.get('image_stats', {}).get('mean_height', 512)

        avg_size = (mean_width + mean_height) / 2

        if avg_size <= 256:
            return 8
        elif avg_size <= 512:
            return 4
        else:
            return 2

    def _recommend_epochs(self, stats: Dict[str, Any]) -> int:
        """Recommend number of training epochs."""
        total_samples = stats.get('total_samples', 1000)
        complexity_score = stats.get('complexity_score', 0.5)

        # Base epochs
        base_epochs = 50

        # Adjust for dataset size
        if total_samples < 500:
            base_epochs = 100  # More epochs for small datasets
        elif total_samples > 10000:
            base_epochs = 30   # Fewer epochs for large datasets

        # Adjust for complexity
        if complexity_score > 0.7:
            base_epochs = int(base_epochs * 1.5)  # More epochs for complex data
        elif complexity_score < 0.3:
            base_epochs = int(base_epochs * 0.7)  # Fewer epochs for simple data

        return max(10, min(200, base_epochs))

    def _recommend_detection_epochs(self, stats: Dict[str, Any]) -> int:
        """Recommend epochs for detection."""
        total_images = stats.get('total_images', 1000)

        if total_images < 1000:
            return 150
        elif total_images < 5000:
            return 100
        else:
            return 80

    def _recommend_segmentation_epochs(self, stats: Dict[str, Any]) -> int:
        """Recommend epochs for segmentation."""
        total_samples = stats.get('total_samples', 1000)

        if total_samples < 1000:
            return 120
        elif total_samples < 5000:
            return 80
        else:
            return 60

    def _recommend_optimizer(self, stats: Dict[str, Any]) -> str:
        """Recommend optimizer."""
        complexity_score = stats.get('complexity_score', 0.5)

        # AdamW is generally good for most cases
        if complexity_score > 0.6:
            return 'AdamW'  # Better for complex datasets
        else:
            return 'Adam'   # Simpler for easier datasets

    def _recommend_image_size(self, stats: Dict[str, Any]) -> int:
        """Recommend image size for detection."""
        mean_width = stats.get('image_stats', {}).get('mean_width', 640)
        mean_height = stats.get('image_stats', {}).get('mean_height', 640)

        avg_size = (mean_width + mean_height) / 2

        # Round to common YOLO sizes
        if avg_size <= 320:
            return 416
        elif avg_size <= 480:
            return 640
        elif avg_size <= 720:
            return 832
        else:
            return 1024

    def _recommend_augmentation(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend data augmentation strategies."""
        total_samples = stats.get('total_samples', 1000)
        class_balance = stats.get('class_balance_ratio', 1.0)

        augmentation = {
            'horizontal_flip': True,
            'rotation': 10 if total_samples < 5000 else 5,
            'color_jitter': True if total_samples < 5000 else False,
            'random_crop': True if total_samples > 1000 else False,
            'cutmix': True if total_samples > 2000 and class_balance < 0.8 else False
        }

        return augmentation

    def _recommend_detection_augmentation(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend augmentation for detection."""
        total_images = stats.get('total_images', 1000)

        return {
            'mosaic': True if total_images > 1000 else False,
            'mixup': True if total_images > 2000 else False,
            'hsv_augmentation': True,
            'flip_lr': 0.5,
            'flip_ud': 0.0,  # Usually not good for detection
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2.0
        }

    def _recommend_segmentation_augmentation(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend augmentation for segmentation."""
        return {
            'horizontal_flip': True,
            'rotation': 15,
            'elastic_transform': True,
            'color_jitter': True,
            'gaussian_blur': True,
            'random_crop_and_resize': True
        }

    def _recommend_regularization(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend regularization techniques."""
        total_samples = stats.get('total_samples', 1000)
        complexity_score = stats.get('complexity_score', 0.5)

        # More regularization for smaller datasets
        if total_samples < 1000:
            return {
                'dropout': 0.5,
                'weight_decay': 1e-3,
                'early_stopping': True,
                'patience': 10
            }
        elif total_samples < 5000:
            return {
                'dropout': 0.3,
                'weight_decay': 1e-4,
                'early_stopping': True,
                'patience': 15
            }
        else:
            return {
                'dropout': 0.1,
                'weight_decay': 1e-5,
                'early_stopping': True,
                'patience': 20
            }

    def _generate_reasoning(self, stats: Dict[str, Any], recommendations: Dict[str, Any]) -> Dict[str, str]:
        """Generate reasoning explanations for recommendations."""
        reasoning = {}

        total_samples = stats.get('total_samples', 0)
        num_classes = stats.get('num_classes', 0)
        complexity_score = stats.get('complexity_score', 0)

        # Model reasoning
        model = recommendations['model']
        if 'mobilenet' in model:
            reasoning['model'] = f"MobileNet recommended due to small dataset ({total_samples} samples) or low complexity ({complexity_score:.2f})"
        elif 'efficientnet' in model:
            reasoning['model'] = f"EfficientNet provides good balance for {total_samples} samples with {num_classes} classes"
        elif 'resnet' in model:
            reasoning['model'] = f"ResNet recommended for higher complexity ({complexity_score:.2f}) or many classes ({num_classes})"

        # Learning rate reasoning
        lr = recommendations['learning_rate']
        if lr < 0.0005:
            reasoning['learning_rate'] = "Lower learning rate for small/complex dataset to ensure stable training"
        elif lr > 0.005:
            reasoning['learning_rate'] = "Higher learning rate for large/simple dataset to speed up training"
        else:
            reasoning['learning_rate'] = "Standard learning rate suitable for dataset characteristics"

        # Batch size reasoning
        batch_size = recommendations['batch_size']
        mean_res = stats.get('image_stats', {}).get('mean_width', 224)
        reasoning['batch_size'] = f"Batch size {batch_size} optimized for image resolution ~{mean_res:.0f}px and typical GPU memory"

        # Epochs reasoning
        epochs = recommendations['epochs']
        if epochs > 80:
            reasoning['epochs'] = f"More epochs ({epochs}) recommended for small dataset to ensure convergence"
        elif epochs < 40:
            reasoning['epochs'] = f"Fewer epochs ({epochs}) sufficient for large dataset to prevent overfitting"
        else:
            reasoning['epochs'] = f"Standard epoch count ({epochs}) appropriate for dataset size"

        return reasoning

    def _generate_detection_reasoning(self, stats: Dict[str, Any], recommendations: Dict[str, Any]) -> Dict[str, str]:
        """Generate reasoning for detection recommendations."""
        reasoning = {}

        total_images = stats.get('total_images', 0)
        mean_boxes = stats.get('bbox_stats', {}).get('mean_boxes_per_image', 0)

        # Model reasoning
        model = recommendations['model']
        if 'n.pt' in model:
            reasoning['model'] = f"YOLOv8n (nano) for small dataset ({total_images} images) - fast and efficient"
        elif 'l.pt' in model:
            reasoning['model'] = f"YOLOv8l (large) for complex scenes with {mean_boxes:.1f} objects per image"
        else:
            reasoning['model'] = f"YOLOv8s/m provides good balance for {total_images} images"

        # Image size reasoning
        img_size = recommendations['img_size']
        reasoning['img_size'] = f"Image size {img_size} balances accuracy and inference speed for your dataset"

        return reasoning

    def _generate_segmentation_reasoning(self, stats: Dict[str, Any], recommendations: Dict[str, Any]) -> Dict[str, str]:
        """Generate reasoning for segmentation recommendations."""
        reasoning = {}

        num_classes = stats.get('num_classes', 0)
        total_samples = stats.get('total_samples', 0)

        # Model reasoning
        model = recommendations['model']
        if 'deeplabv3' in model:
            reasoning['model'] = f"DeepLabV3 excels at {num_classes}-class segmentation with atrous convolution"
        elif 'fcn' in model:
            reasoning['model'] = f"FCN provides simpler architecture suitable for {total_samples} samples"

        return reasoning
