"""
Detection Trainer Module
=======================
Object detection training using ultralytics YOLOv8.
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics
from ultralytics.data.utils import check_det_dataset

from ..core.base_trainer import BaseTrainer, TaskConfig


class DetectionConfig(TaskConfig):
    """Configuration for object detection tasks."""

    def __init__(self,
                 model_name: str = 'yolov8n.pt',
                 img_size: int = 640,
                 learning_rate: float = 0.01,
                 weight_decay: float = 5e-4,
                 **kwargs):
        """
        Initialize detection configuration.

        Args:
            model_name: YOLO model variant (yolov8n.pt, yolov8s.pt, etc.)
            img_size: Input image size
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        super().__init__(**kwargs)

        self.model_name = model_name
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # YOLO model instance
        self._yolo_model = None

        print(f"✅ DetectionConfig initialized:")
        print(f"   Model: {model_name}")
        print(f"   Image size: {img_size}")
        print(f"   Learning rate: {learning_rate}")

    def get_model(self) -> YOLO:
        """Get the YOLO detection model."""
        if self._yolo_model is None:
            # Load pre-trained YOLO model
            self._yolo_model = YOLO(self.model_name)
            print(f"✅ YOLO model loaded: {self.model_name}")

        return self._yolo_model

    def get_loss_function(self) -> nn.Module:
        """YOLO handles loss internally, return dummy."""
        return nn.Identity()  # YOLO manages its own loss

    def get_optimizer(self, model) -> None:
        """YOLO handles optimizer internally."""
        return None  # YOLO manages its own optimizer

    def get_scheduler(self, optimizer) -> None:
        """YOLO handles scheduler internally."""
        return None  # YOLO manages its own scheduler

    def get_dataloader(self, dataset_path: str, batch_size: int = 32, shuffle: bool = True) -> str:
        """Return dataset path for YOLO (YOLO handles dataloader internally)."""
        return dataset_path

    def compute_metrics(self, outputs: Any, targets: Any) -> Dict[str, float]:
        """Compute detection metrics (handled by YOLO)."""
        # YOLO computes metrics internally during validation
        return {}


class YOLOWrapper(nn.Module):
    """Wrapper class to make YOLO compatible with BaseTrainer."""

    def __init__(self, yolo_model: YOLO):
        super().__init__()
        self.yolo_model = yolo_model

    def forward(self, x):
        """Forward pass for YOLO model."""
        return self.yolo_model(x)

    def train(self, mode=True):
        """Set training mode."""
        self.training = mode
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
    
    def loss(self, batch, preds):
        """Loss function for YOLO model - delegate to underlying model."""
        # Handle DDP wrapper
        if hasattr(self.yolo_model, 'module'):
            # DDP wrapped model
            return self.yolo_model.module.loss(batch, preds)
        else:
            # Regular model
            return self.yolo_model.loss(batch, preds)


class DetectionTrainer(BaseTrainer):
    """Trainer specifically for object detection using YOLOv8."""

    def __init__(self,
                 model_name: str = 'yolov8n.pt',
                 img_size: int = 640,
                 learning_rate: float = 0.01,
                 device: str = "auto",
                 **kwargs):
        """
        Initialize DetectionTrainer.

        Args:
            model_name: YOLO model variant
            img_size: Input image size
            learning_rate: Learning rate for training
            device: Device to use for training (supports '0,1' for multi-GPU)
        """
        # Create configuration
        config = DetectionConfig(
            model_name=model_name,
            img_size=img_size,
            learning_rate=learning_rate,
            **kwargs
        )

        # Preserve raw device string for YOLO (e.g., '0,1')
        self._yolo_device_override: Optional[str] = None
        base_device = device
        if isinstance(device, str) and ("," in device or device.isdigit()):
            # If user passed '0,1' or '0', keep this for YOLO and use 'cuda' for torch
            self._yolo_device_override = device
            base_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize base trainer
        super().__init__(config, device=base_device)

        self.model_name = model_name
        self.img_size = img_size
        self.yolo_model = None

        print(f"🎯 DetectionTrainer initialized with {model_name}")
        if self._yolo_device_override:
            print(f"🧩 YOLO device override: {self._yolo_device_override}")

    def setup_training_components(self):
        """Initialize YOLO model."""
        print("🔧 Setting up YOLO detection components...")

        # Get YOLO model from config
        self.yolo_model = self.task_config.get_model()

        # For YOLO, we don't wrap it - let YOLO handle its own training
        # The YOLO model will be used directly in the train() method
        self.model = self.yolo_model

        print(f"✅ YOLO model initialized: {self.model_name}")

    def validate_dataset_format(self, dataset_path: str) -> bool:
        """Validate that dataset is in YOLO format."""
        dataset_path = Path(dataset_path)

        # Check for data.yaml
        yaml_path = dataset_path / 'data.yaml'
        if not yaml_path.exists():
            print(f"❌ Missing data.yaml file at {yaml_path}")
            return False

        # Accept either of the following structures:
        # A) images/{train,val|valid}, labels/{train,val|valid}
        # B) {train,val|valid,test}/{images,labels}
        splits_a = ['train', 'val', 'valid']
        ok = False
        # Structure A check
        images_dir = dataset_path / 'images'
        labels_dir = dataset_path / 'labels'
        if images_dir.exists() and labels_dir.exists():
            has_train = (images_dir / 'train').exists() and (labels_dir / 'train').exists()
            has_val = ((images_dir / 'val').exists() or (images_dir / 'valid').exists()) and \
                      ((labels_dir / 'val').exists() or (labels_dir / 'valid').exists())
            ok = has_train and has_val
        # Structure B check
        if not ok:
            for split in ['train', 'val', 'valid']:
                split_dir = dataset_path / split
                if split_dir.exists():
                    if (split_dir / 'images').exists() and (split_dir / 'labels').exists():
                        ok = True
                    else:
                        ok = False
                        break
            # 如果找到任何有效的分割，就認為格式正確
            if ok:
                # 檢查是否至少有一個訓練分割和一個驗證分割
                has_train = (dataset_path / 'train').exists() and \
                           (dataset_path / 'train' / 'images').exists() and \
                           (dataset_path / 'train' / 'labels').exists()
                has_val = ((dataset_path / 'val').exists() and \
                          (dataset_path / 'val' / 'images').exists() and \
                          (dataset_path / 'val' / 'labels').exists()) or \
                         ((dataset_path / 'valid').exists() and \
                          (dataset_path / 'valid' / 'images').exists() and \
                          (dataset_path / 'valid' / 'labels').exists())
                ok = has_train and has_val
        if not ok:
            print("❌ Dataset directory structure not recognized as YOLO format")
            return False

        print("✅ Dataset format validation passed")
        return True

    def create_yolo_config(self, dataset_path: str, epochs: int, batch_size: int) -> Dict[str, Any]:
        """Create YOLO training configuration."""
        # 確保 models 目錄存在
        os.makedirs('models', exist_ok=True)
        
        # 支援多GPU訓練 - YOLOv8 內建 DDP 支援
        device_config = self._yolo_device_override if self._yolo_device_override else str(self.device)
        if ',' in str(device_config):
            print(f"🧩 啟用多GPU訓練: {device_config}")
            # YOLOv8 支援多GPU，直接傳遞設備列表
        
        config = {
            'data': str(Path(dataset_path) / 'data.yaml'),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': self.img_size,
            'lr0': self.task_config.learning_rate,
            'weight_decay': self.task_config.weight_decay,
            'device': device_config,
            'project': 'models',  # 直接保存到 models 目錄
            'name': f'detection_{self.model_name}',  # 更簡潔的命名
            'save': True,
            'save_period': 10,
            'cache': False,
            'workers': 8,
            'verbose': True,
        }

        return config

    def _is_rank0(self) -> bool:
        """Return True if current process is rank-0 or non-distributed."""
        try:
            rank = int(os.getenv('RANK', '-1'))
        except ValueError:
            rank = -1
        return rank in (-1, 0)

    def train(self,
              dataset_path: str,
              epochs: int = 100,
              batch_size: int = 16,
              validation_split: float = 0.2,
              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop using YOLO.

        Args:
            dataset_path: Path to YOLO format dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Ignored (YOLO uses predefined splits)
            save_path: Path to save the trained model

        Returns:
            Training results from YOLO
        """
        print(f"🚀 Starting YOLO detection training for {epochs} epochs...")
        print(f"📁 Dataset path: {dataset_path}")
        print(f"🔢 Batch size: {batch_size}")
        print(f"📐 Image size: {self.img_size}")

        # Setup training components
        self.setup_training_components()

        # Validate dataset format
        if not self.validate_dataset_format(dataset_path):
            raise ValueError("Invalid dataset format. Please ensure YOLO format with data.yaml and splits containing images/ and labels/ directories.")

        # Create YOLO training configuration
        train_config = self.create_yolo_config(dataset_path, epochs, batch_size)

        try:
            # Start YOLO training
            print("🏃‍♂️ Starting YOLO training...")
            results = self.yolo_model.train(**train_config)

            # Save model if path provided (only on rank-0)
            if save_path and self._is_rank0():
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.yolo_model.save(save_path)
                print(f"💾 Model saved to: {save_path}")

            # Handle possible None results in DDP non-zero ranks
            results_dict = {}
            save_dir = None
            
            # Try multiple ways to extract results from YOLO
            if results is not None:
                if hasattr(results, 'results_dict') and isinstance(results.results_dict, dict):
                    results_dict = results.results_dict
                elif isinstance(results, dict):
                    results_dict = results
                elif hasattr(results, 'results') and isinstance(results.results, dict):
                    results_dict = results.results
                elif hasattr(results, 'metrics') and isinstance(results.metrics, dict):
                    results_dict = results.metrics
                else:
                    # Try to get metrics from the last validation results
                    if hasattr(results, 'val') and hasattr(results.val, 'results_dict'):
                        results_dict = results.val.results_dict
                    elif hasattr(results, 'val') and isinstance(results.val, dict):
                        results_dict = results.val
                
                if hasattr(results, 'save_dir'):
                    save_dir = results.save_dir
            else:
                if not self._is_rank0():
                    print("ℹ️ Non rank-0 process: skipping metrics aggregation.")
                else:
                    print("⚠️ YOLO returned no structured results; metrics will be empty.")

            # Extract metrics with multiple possible key formats
            final_metrics = {
                'map50': 0.0,
                'map50_95': 0.0,
                'precision': 0.0,
                'recall': 0.0,
            }
            
            if results_dict:
                # Try different key formats that YOLO might use
                map50_keys = ['metrics/mAP50(B)', 'mAP50', 'map50', 'val/mAP50', 'val_map50']
                map50_95_keys = ['metrics/mAP50-95(B)', 'mAP50-95', 'map50-95', 'val/mAP50-95', 'val_map50-95']
                precision_keys = ['metrics/precision(B)', 'precision', 'val/precision', 'val_precision']
                recall_keys = ['metrics/recall(B)', 'recall', 'val/recall', 'val_recall']
                
                for key in map50_keys:
                    if key in results_dict:
                        final_metrics['map50'] = float(results_dict[key])
                        break
                        
                for key in map50_95_keys:
                    if key in results_dict:
                        final_metrics['map50_95'] = float(results_dict[key])
                        break
                        
                for key in precision_keys:
                    if key in results_dict:
                        final_metrics['precision'] = float(results_dict[key])
                        break
                        
                for key in recall_keys:
                    if key in results_dict:
                        final_metrics['recall'] = float(results_dict[key])
                        break
                
                # Debug: print what we found
                if self._is_rank0():
                    print(f"🔍 Debug - Found results_dict keys: {list(results_dict.keys())}")
                    print(f"🔍 Debug - Extracted metrics: {final_metrics}")

            if self._is_rank0():
                print("\n🎉 YOLO training completed!")
                print(f"📊 Final metrics:")
                for metric, value in final_metrics.items():
                    print(f"   {metric}: {value:.4f}")

            # Fallback: if multi-GPU returned empty metrics, try reading from save_dir/results.csv
            try:
                need_fallback = self._is_rank0() and (final_metrics['map50'] == 0.0 and final_metrics['map50_95'] == 0.0 
                                  and final_metrics['precision'] == 0.0 and final_metrics['recall'] == 0.0)
                if need_fallback:
                    inferred_save_dir = None
                    # Prefer results.save_dir if available
                    if 'model_path' in locals() and save_dir:
                        inferred_save_dir = save_dir
                    elif hasattr(self.yolo_model, 'trainer') and hasattr(self.yolo_model.trainer, 'save_dir'):
                        inferred_save_dir = str(self.yolo_model.trainer.save_dir)

                    if inferred_save_dir:
                        import os
                        import pandas as pd
                        csv_path = os.path.join(inferred_save_dir, 'results.csv')
                        if os.path.exists(csv_path):
                            df = pd.read_csv(csv_path)
                            if not df.empty:
                                last = df.iloc[-1]
                                # Try multiple keys commonly used by YOLOv8
                                def get_col(series, keys, default=0.0):
                                    for k in keys:
                                        if k in series:
                                            try:
                                                return float(series[k])
                                            except Exception:
                                                pass
                                    return default

                                final_metrics['map50'] = get_col(last, ['metrics/mAP50(B)', 'mAP50', 'map50', 'val/mAP50'])
                                final_metrics['map50_95'] = get_col(last, ['metrics/mAP50-95(B)', 'mAP50-95', 'map50-95', 'val/mAP50-95'])
                                final_metrics['precision'] = get_col(last, ['metrics/precision(B)', 'precision', 'val/precision'])
                                final_metrics['recall'] = get_col(last, ['metrics/recall(B)', 'recall', 'val/recall'])
                                if self._is_rank0():
                                    print("ℹ️ Metrics were empty from trainer; populated from results.csv:")
                                    for metric, value in final_metrics.items():
                                        print(f"   {metric}: {value:.4f}")
            except Exception as e:
                if self._is_rank0():
                    print(f"⚠️ Failed to read fallback metrics from results.csv: {e}")

            return {
                'yolo_results': results,
                'final_metrics': final_metrics,
                'model_path': save_dir
            }

        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            raise

    def predict(self, image_path: str, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Make predictions on new images.

        Args:
            image_path: Path to image or directory of images
            conf_threshold: Confidence threshold for detections

        Returns:
            Detection results
        """
        if self.yolo_model is None:
            raise RuntimeError("Model not initialized. Please train or load a model first.")

        print(f"🔍 Making predictions on: {image_path}")

        # Run inference
        results = self.yolo_model.predict(
            source=image_path,
            conf=conf_threshold,
            save=True,
            project='ivit_predictions'
        )

        # Process results
        predictions = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                pred_data = {
                    'boxes': boxes.xyxy.cpu().numpy(),  # x1, y1, x2, y2
                    'confidences': boxes.conf.cpu().numpy(),
                    'classes': boxes.cls.cpu().numpy().astype(int),
                    'names': [r.names[int(cls)] for cls in boxes.cls.cpu().numpy()]
                }
                predictions.append(pred_data)

        print(f"✅ Predictions completed. Found {len(predictions)} images with detections.")

        return {
            'predictions': predictions,
            'yolo_results': results
        }

    def validate(self, dataset_path: str) -> Dict[str, float]:
        """
        Validate the model on validation dataset.

        Args:
            dataset_path: Path to YOLO format dataset

        Returns:
            Validation metrics
        """
        if self.yolo_model is None:
            raise RuntimeError("Model not initialized. Please train or load a model first.")

        print("📊 Running validation...")

        # Run YOLO validation
        val_results = self.yolo_model.val(
            data=str(Path(dataset_path) / 'data.yaml'),
            imgsz=self.img_size
        )

        # Extract metrics
        metrics = {
            'map50': val_results.results_dict.get('metrics/mAP50(B)', 0.0),
            'map50_95': val_results.results_dict.get('metrics/mAP50-95(B)', 0.0),
            'precision': val_results.results_dict.get('metrics/precision(B)', 0.0),
            'recall': val_results.results_dict.get('metrics/recall(B)', 0.0),
        }

        print("✅ Validation completed:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")

        return metrics

    def get_recommendations(self, dataset_path: str) -> Dict[str, Any]:
        """Get intelligent recommendations for detection dataset."""
        from ..utils.smart_recommendation import SmartRecommendationEngine
        from ..utils.dataset_analyzer import DatasetAnalyzer

        # Analyze dataset
        analyzer = DatasetAnalyzer()
        stats = analyzer.extract_detection_statistics(dataset_path)

        # Get recommendations
        recommendation_engine = SmartRecommendationEngine()
        recommendations = recommendation_engine.get_detection_recommendations(stats)

        return recommendations

    def apply_recommendations(self, recommendations: Dict[str, Any]):
        """Apply intelligent recommendations to the detection trainer."""
        print("🧠 Applying detection recommendations...")

        # Update model if recommended
        if 'model' in recommendations:
            recommended_model = recommendations['model']
            if recommended_model != self.model_name:
                print(f"📝 Updating model: {self.model_name} → {recommended_model}")
                self.model_name = recommended_model
                self.task_config.model_name = recommended_model

        # Update image size if recommended
        if 'img_size' in recommendations:
            recommended_size = recommendations['img_size']
            print(f"📝 Updating image size: {self.img_size} → {recommended_size}")
            self.img_size = recommended_size
            self.task_config.img_size = recommended_size

        # Update learning rate if recommended
        if 'learning_rate' in recommendations:
            recommended_lr = recommendations['learning_rate']
            print(f"📝 Updating learning rate: {self.task_config.learning_rate} → {recommended_lr}")
            self.task_config.learning_rate = recommended_lr

        print("✅ Detection recommendations applied!")

