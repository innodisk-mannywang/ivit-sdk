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
from ultralytics.utils import LOGGER
from ultralytics.data.utils import check_det_dataset

from ..core.base_trainer import BaseTrainer, TaskConfig


class DetectionConfig(TaskConfig):
    """Configuration for object detection tasks."""

    def __init__(self,
                 model_name: str = 'yolov8n.pt',
                 img_size: int = 640,
                 learning_rate: float = 0.01,
                 weight_decay: float = 5e-4,
                 verbose: bool = True,
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
        self.verbose = verbose

        # YOLO model instance
        self._yolo_model = None

        if self.verbose:
            print(f"✅ DetectionConfig initialized:")
            print(f"   Model: {model_name}")
            print(f"   Image size: {img_size}")
            print(f"   Learning rate: {learning_rate}")

    def get_model(self) -> YOLO:
        """Get the YOLO detection model."""
        if self._yolo_model is None:
            # Load pre-trained YOLO model
            self._yolo_model = YOLO(self.model_name)
            if self.verbose:
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
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize DetectionTrainer.

        Args:
            model_name: YOLO model variant
            img_size: Input image size
            learning_rate: Learning rate for training
            device: Device to use for training (supports '0,1' for multi-GPU)
        """
        # 禁用多張 GPU：若使用者指定超過 1 張 GPU，直接拒絕
        if isinstance(device, str) and "," in device:
            gpu_ids = [x.strip() for x in device.split(',') if x.strip() != ""]
            if len(gpu_ids) > 1:
                raise ValueError("Detection 訓練目前不支援多張 GPU，請改為指定單一卡，例如 --device 0 或 --device 1")

        # Create configuration
        config = DetectionConfig(
            model_name=model_name,
            img_size=img_size,
            learning_rate=learning_rate,
            verbose=verbose,
            **kwargs
        )

        # Preserve raw device string for YOLO (e.g., '0,1') and strictly bind base device
        self._yolo_device_override: Optional[str] = None
        base_device = device
        if isinstance(device, str):
            if "," in device:
                # Multi-GPU: bind primary to first id to avoid initializing cuda:0 unexpectedly
                self._yolo_device_override = device
                # 先限制可見裝置，確保不會觸發實體 GPU0
                try:
                    os.environ["CUDA_VISIBLE_DEVICES"] = device
                except Exception:
                    pass
                if torch.cuda.is_available():
                    try:
                        primary_gpu = device.split(',')[0].strip()
                        if primary_gpu.isdigit():
                            base_device = f'cuda:{primary_gpu}'
                        else:
                            base_device = 'cuda'
                    except Exception:
                        base_device = 'cuda'
                else:
                    base_device = 'cpu'
            elif device.isdigit():
                # Single-GPU numeric: strictly bind to that CUDA index
                self._yolo_device_override = device
                # 先限制可見裝置，避免 PyTorch/YOLO 預設初始化到 0
                try:
                    os.environ["CUDA_VISIBLE_DEVICES"] = device
                except Exception:
                    pass
                if torch.cuda.is_available():
                    base_device = f'cuda:{device}'
                else:
                    base_device = 'cpu'
            else:
                # 'cpu' or other strings: pass through
                base_device = device

        # Initialize base trainer
        super().__init__(config, device=base_device, verbose=verbose)

        self.model_name = model_name
        self.img_size = img_size
        self.yolo_model = None
        self._suppress_logs: bool = False

        if self.verbose:
            print(f"🎯 DetectionTrainer initialized with {model_name}")
            if self._yolo_device_override:
                print(f"🧩 YOLO device override: {self._yolo_device_override}")

    def setup_training_components(self):
        """Initialize YOLO model."""
        if self.verbose:
            print("🔧 Setting up YOLO detection components...")

        # Get YOLO model from config
        self.yolo_model = self.task_config.get_model()

        # For YOLO, we don't wrap it - let YOLO handle its own training
        # The YOLO model will be used directly in the train() method
        self.model = self.yolo_model

        if self.verbose:
            print(f"✅ YOLO model initialized: {self.model_name}")

        # Bridge YOLO callbacks to BaseTrainer events if possible
        try:
            self._setup_yolo_callbacks()
        except Exception as e:
            print(f"⚠️ Failed to setup YOLO callbacks: {e}")

    def _setup_yolo_callbacks(self):
        """Register YOLO callbacks and map them to BaseTrainer events."""
        if self.yolo_model is None:
            return

        # 用於追蹤訓練狀態
        self._training_start_time = None
        self._current_epoch = 0
        self._current_batch = 0
        self._total_batches = 0

        def _wrap_emit(mapped_event_name: str):
            def _cb(trainer_obj=None, *args, **kwargs):
                payload = {}
                try:
                    if hasattr(trainer_obj, 'epoch'):
                        # YOLO 以 0 起算，轉為 1-based
                        self._current_epoch = int(getattr(trainer_obj, 'epoch', -1)) + 1
                        payload['epoch'] = self._current_epoch
                    
                    if hasattr(trainer_obj, 'optimizer') and hasattr(trainer_obj.optimizer, 'param_groups'):
                        try:
                            payload['lr'] = float(trainer_obj.optimizer.param_groups[0]['lr'])
                        except Exception:
                            pass
                    
                    # metrics 可能在不同屬性上，盡量嘗試
                    for key in ['metrics', 'results_dict', 'metrics_dict']:
                        if hasattr(trainer_obj, key):
                            try:
                                metrics_dict = getattr(trainer_obj, key)
                                if isinstance(metrics_dict, dict):
                                    payload['metrics'] = {k: float(v) for k, v in metrics_dict.items() if isinstance(v, (int, float))}
                            except Exception:
                                pass
                    
                    if hasattr(trainer_obj, 'loss_items'):
                        try:
                            payload['loss_items'] = [float(x) for x in list(trainer_obj.loss_items)]
                        except Exception:
                            pass
                    
                    # 為所有事件添加基本的 progress 資料
                    if mapped_event_name == 'on_batch_end':
                        # 添加基本的 progress 資訊
                        payload['batch'] = f"{self._current_batch + 1}/{self._total_batches}" if self._total_batches > 0 else "1/1"
                        payload['progress'] = f"{(self._current_batch + 1) / self._total_batches * 100:.1f}%" if self._total_batches > 0 else "100.0%"
                        
                        # 添加時間資訊
                        if hasattr(self, '_yolo_training_start_time'):
                            elapsed_time = time.time() - self._yolo_training_start_time
                            payload['elapsed'] = f"{elapsed_time:.1f}s"
                            
                            if self._current_batch > 0 and self._total_batches > 0:
                                eta_seconds = (elapsed_time / (self._current_batch + 1)) * (self._total_batches - self._current_batch - 1)
                                payload['eta'] = f"{eta_seconds:.1f}s"
                                payload['speed'] = f"{(self._current_batch + 1) / elapsed_time:.1f} it/s" if elapsed_time > 0 else "0.0 it/s"
                            else:
                                payload['eta'] = "0.0s"
                                payload['speed'] = "0.0 it/s"
                    
                    # 添加 progress 相關資料
                    if mapped_event_name == 'on_batch_end':
                        import time
                        
                        # 初始化訓練開始時間
                        if self._training_start_time is None:
                            self._training_start_time = time.time()
                        
                        # 調試：記錄 trainer_obj 的屬性
                        debug_attrs = []
                        if trainer_obj:
                            debug_attrs = [attr for attr in dir(trainer_obj) if not attr.startswith('_') and not callable(getattr(trainer_obj, attr))]
                        payload['debug_attrs'] = debug_attrs[:10]  # 只記錄前10個屬性
                        
                        # 獲取 batch 資訊 - 嘗試多種可能的屬性名
                        batch_attrs = ['ni', 'batch_idx', 'batch_index', 'i']
                        total_attrs = ['nb', 'total_batches', 'n', 'len']
                        
                        for attr in batch_attrs:
                            if hasattr(trainer_obj, attr):
                                try:
                                    self._current_batch = int(getattr(trainer_obj, attr))
                                    payload['debug_batch_attr'] = attr
                                    break
                                except (ValueError, TypeError):
                                    continue
                        
                        for attr in total_attrs:
                            if hasattr(trainer_obj, attr):
                                try:
                                    self._total_batches = int(getattr(trainer_obj, attr))
                                    payload['debug_total_attr'] = attr
                                    break
                                except (ValueError, TypeError):
                                    continue
                        
                        # 如果還是沒有找到，嘗試從 dataloader 獲取
                        if self._total_batches == 0 and hasattr(trainer_obj, 'train_loader'):
                            try:
                                self._total_batches = len(trainer_obj.train_loader)
                                payload['debug_total_source'] = 'train_loader'
                            except Exception:
                                pass
                        
                        # 記錄調試資訊
                        payload['debug_current_batch'] = self._current_batch
                        payload['debug_total_batches'] = self._total_batches
                        
                        # 計算 progress 資料
                        if self._total_batches > 0:
                            progress_percent = (self._current_batch + 1) / self._total_batches * 100
                            payload['progress_percent'] = float(progress_percent)
                            payload['batch_index'] = self._current_batch
                            payload['total_batches'] = self._total_batches
                            
                            # 計算時間相關資料
                            if self._training_start_time:
                                elapsed_time = time.time() - self._training_start_time
                                payload['elapsed_time'] = float(elapsed_time)
                                
                                if self._current_batch > 0:
                                    avg_time_per_batch = elapsed_time / (self._current_batch + 1)
                                    remaining_batches = self._total_batches - (self._current_batch + 1)
                                    eta_seconds = remaining_batches * avg_time_per_batch
                                    payload['eta_seconds'] = float(eta_seconds)
                                    payload['iter_per_sec'] = float((self._current_batch + 1) / elapsed_time) if elapsed_time > 0 else 0.0
                                else:
                                    payload['eta_seconds'] = 0.0
                                    payload['iter_per_sec'] = 0.0
                        
                        # 添加 loss 資料
                        if hasattr(trainer_obj, 'loss_items') and trainer_obj.loss_items:
                            try:
                                loss_items = [float(x) for x in list(trainer_obj.loss_items)]
                                if loss_items:
                                    payload['loss'] = float(sum(loss_items))
                                    payload['avg_loss'] = float(sum(loss_items) / len(loss_items))
                            except Exception:
                                pass
                
                except Exception as e:
                    # 調試用：如果出現錯誤，記錄到 payload 中
                    if mapped_event_name == 'on_batch_end':
                        payload['debug_error'] = str(e)
                
                try:
                    self._emit(mapped_event_name, payload)
                except Exception:
                    pass
            return _cb

        # Map YOLO events to unified events
        # on_train_start -> on_train_start
        # on_train_batch_end -> on_batch_end
        # on_train_epoch_end -> on_epoch_end
        # on_val_end -> on_validate_end
        # on_fit_end -> on_train_end
        try:
            self.yolo_model.add_callback('on_train_start', _wrap_emit('on_train_start'))
            self.yolo_model.add_callback('on_train_batch_end', _wrap_emit('on_batch_end'))
            self.yolo_model.add_callback('on_train_epoch_end', _wrap_emit('on_epoch_end'))
            self.yolo_model.add_callback('on_val_end', _wrap_emit('on_validate_end'))
            self.yolo_model.add_callback('on_fit_end', _wrap_emit('on_train_end'))
            if self.verbose:
                print("✅ YOLO callbacks registered successfully")
        except Exception as e:
            # 某些版本的 ultralytics 可能使用不同 API，忽略即可
            if self.verbose:
                print(f"ℹ️ YOLO callback registration encountered an issue: {e}")

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

        if self.verbose:
            print("✅ Dataset format validation passed")
        return True

    def create_yolo_config(self, dataset_path: str, epochs: int, batch_size: int) -> Dict[str, Any]:
        """Create YOLO training configuration."""
        # 確保 models 目錄存在
        os.makedirs('models', exist_ok=True)
        
        # 支援多GPU訓練 - YOLOv8 內建 DDP 支援
        # 若已限制 CUDA_VISIBLE_DEVICES，則需將傳給 YOLO 的 device 索引改為相對於可見裝置的索引
        if self._yolo_device_override and ',' in self._yolo_device_override:
            # 例如 override: "2,3" → 可見裝置為 [2,3]，傳給 YOLO 的 device 應為 "0,1"
            num_visible = len(self._yolo_device_override.split(','))
            device_config = ','.join(str(i) for i in range(num_visible))
            print(f"🧩 啟用多GPU訓練: 可見裝置={self._yolo_device_override} → YOLO device='{device_config}'")
        elif self._yolo_device_override and self._yolo_device_override.isdigit():
            # 單卡情況：可見裝置為例如 "1"，傳給 YOLO 的 device 應為 "0"
            device_config = '0'
            print(f"🧭 單GPU訓練: 可見裝置={self._yolo_device_override} → YOLO device='0'")
        else:
            # 未覆寫時，直接使用 base device 字串（可能是 'cpu' 或 'cuda:X'）
            device_config = str(self.device)
        
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
            'patience': epochs,  # 設置 patience 為總 epoch 數，避免提前停止
            'cache': False,
            'workers': 8,
            'verbose': not self._suppress_logs,
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
              save_path: Optional[str] = None,
              callbacks: Optional[Dict[str, List]] = None,
              progress_log_path: Optional[str] = None,
              suppress_yolo_logging: bool = True) -> Dict[str, Any]:
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
        if self.verbose:
            print(f"🚀 Starting YOLO detection training for {epochs} epochs...")
            print(f"📁 Dataset path: {dataset_path}")
            print(f"🔢 Batch size: {batch_size}")
            print(f"📐 Image size: {self.img_size}")

        # Configure callbacks and progress logging if provided
        if callbacks:
            for event_name, cbs in callbacks.items():
                for cb in cbs:
                    try:
                        self.add_callback(event_name, cb)
                    except Exception:
                        pass
        if progress_log_path:
            self.progress_log_path = progress_log_path

        # Configure suppression of Ultralytics logging
        # 當 verbose=False 時，自動禁用 YOLO 的進度條和詳細輸出
        if not self.verbose:
            try:
                import logging
                LOGGER.setLevel(logging.ERROR)
                # 禁用 tqdm 進度條
                import os
                os.environ['TQDM_DISABLE'] = '1'
            except Exception:
                pass

        # Setup training components
        self.setup_training_components()

        # Validate dataset format
        if not self.validate_dataset_format(dataset_path):
            raise ValueError("Invalid dataset format. Please ensure YOLO format with data.yaml and splits containing images/ and labels/ directories.")

        # Create YOLO training configuration
        train_config = self.create_yolo_config(dataset_path, epochs, batch_size)

        try:
            # Start YOLO training
            if self.verbose:
                print("🏃‍♂️ Starting YOLO training...")
            
            # 若使用者以單一數字指定 GPU，如 "1"，強制將目前 CUDA 裝置切到該卡
            try:
                if self._yolo_device_override and isinstance(self._yolo_device_override, str) and self._yolo_device_override.isdigit():
                    if torch.cuda.is_available():
                        torch.cuda.set_device(int(self._yolo_device_override))
                        if self.verbose:
                            print(f"🧭 torch.cuda.set_device({self._yolo_device_override}) 已設定")
            except Exception as _e:
                # 安全忽略，避免因環境差異中斷訓練
                if self.verbose:
                    print(f"ℹ️ 無法設定 CUDA 裝置: {_e}")

            # 手動觸發 callback 來模擬 progress 資料
            import time
            self._yolo_training_start_time = time.time()
            
            # 在訓練開始前觸發 on_train_start
            self._emit('on_train_start', {'epoch': 0})
            
            results = self.yolo_model.train(**train_config)
            
            # 手動觸發一些 callback 事件來提供 progress 資料
            # 由於 YOLO 的 callback 系統可能不穩定，我們手動觸發一些事件
            if hasattr(self, '_emit'):
                # 觸發 epoch end 事件
                final_metrics = {}
                if results and hasattr(results, 'results_dict'):
                    final_metrics = results.results_dict
                elif results and isinstance(results, dict):
                    final_metrics = results
                
                self._emit('on_epoch_end', {
                    'epoch': epochs,
                    'metrics': final_metrics,
                    'val_metrics': final_metrics
                })
                
                # 觸發 train end 事件
                self._emit('on_train_end', {
                    'final_metrics': final_metrics,
                    'model_path': results.save_dir if hasattr(results, 'save_dir') else None
                })

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

