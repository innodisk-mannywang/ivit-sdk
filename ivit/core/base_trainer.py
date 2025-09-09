"""
Base Trainer Module
==================
Core training framework with task configuration support.
"""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from tqdm import tqdm


class TaskConfig(ABC):
    """Abstract base class for task-specific configurations."""

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the model for this task."""
        pass

    @abstractmethod
    def get_loss_function(self) -> nn.Module:
        """Return the loss function for this task."""
        pass

    @abstractmethod
    def get_optimizer(self, model: nn.Module) -> Optimizer:
        """Return the optimizer for this task."""
        pass

    @abstractmethod
    def get_scheduler(self, optimizer: Optimizer) -> Optional[_LRScheduler]:
        """Return the learning rate scheduler for this task."""
        pass

    @abstractmethod
    def get_dataloader(self, dataset_path: str, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Return the dataloader for this task."""
        pass

    @abstractmethod
    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute task-specific metrics."""
        pass


class BaseTrainer:
    """Base trainer class for all AI vision tasks."""

    def __init__(self, task_config: TaskConfig, device: str = "auto"):
        """
        Initialize BaseTrainer.

        Args:
            task_config: Task-specific configuration
            device: Device to use ('auto', 'cpu', 'cuda', or specific GPU id)
        """
        self.task_config = task_config
        self.original_device = device  # 存儲原始設備配置
        self.device = self._setup_device(device)

        # Training components (initialized during training)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = []

        print(f"✅ BaseTrainer initialized with device: {self.device}")

    def _setup_device(self, device: str) -> torch.device:
        """Setup and return the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            # 處理多 GPU 配置 (如 "0,1,2,3")
            if ',' in device:
                # 多 GPU 配置，使用第一個 GPU 作為主設備
                primary_gpu = device.split(',')[0].strip()
                return torch.device(f"cuda:{primary_gpu}")
            else:
                # 單 GPU 配置
                if device.isdigit():
                    return torch.device(f"cuda:{device}")
                else:
                    return torch.device(device)

    def setup_training_components(self):
        """Initialize model, optimizer, and other training components."""
        print("🔧 Setting up training components...")

        # Initialize model
        self.model = self.task_config.get_model().to(self.device)

        # 檢查是否為多 GPU 配置
        is_multi_gpu = (',' in self.original_device and 
                       torch.cuda.is_available() and 
                       torch.cuda.device_count() > 1)
        
        # 自動啟用 DataParallel 當檢測到多 GPU 配置
        if is_multi_gpu:
            gpu_ids = [int(x.strip()) for x in self.original_device.split(',')]
            print(f"🧩 Enabling DataParallel on GPUs: {gpu_ids}")
            self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
        else:
            # 檢查環境變數是否強制啟用 DataParallel
            use_dp = os.getenv("USE_DP", "0").lower() in ("1", "true", "yes")
            if use_dp and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                print(f"🧩 Enabling DataParallel on {torch.cuda.device_count()} GPUs")
                self.model = nn.DataParallel(self.model)

        # Initialize optimizer
        self.optimizer = self.task_config.get_optimizer(self.model)

        # Initialize scheduler (optional)
        self.scheduler = self.task_config.get_scheduler(self.optimizer)

        # Initialize loss function
        self.criterion = self.task_config.get_loss_function().to(self.device)

        print(f"✅ Model initialized: {type(self.model).__name__}")
        print(f"✅ Optimizer: {type(self.optimizer).__name__}")
        print(f"✅ Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
        print(f"✅ Loss function: {type(self.criterion).__name__}")

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_start_time = time.time()
        total_loss = 0.0
        total_samples = 0
        all_outputs = []
        all_targets = []

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update statistics
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Store outputs and targets for metrics calculation
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

        # Epoch timing and throughput
        epoch_seconds = max(time.time() - epoch_start_time, 1e-6)
        num_batches = len(dataloader)
        iter_per_sec = float(num_batches) / float(epoch_seconds)
        metrics['epoch_seconds'] = epoch_seconds
        metrics['iter_per_sec'] = iter_per_sec

        # Optional: write timing to file if requested
        timing_file = os.getenv('EPOCH_TIMING_FILE')
        if timing_file:
            try:
                os.makedirs(os.path.dirname(timing_file), exist_ok=True)
                with open(timing_file, 'a') as f:
                    f.write(
                        f"epoch={self.current_epoch},iter_per_sec={iter_per_sec:.4f},epoch_seconds={epoch_seconds:.4f},"
                        f"device={self.device.type},dataparallel={isinstance(self.model, nn.DataParallel)}\n"
                    )
            except Exception as e:
                print(f"⚠️ Failed to write timing file: {e}")

        return metrics

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Validation"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

        # Compute metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.task_config.compute_metrics(all_outputs, all_targets)
        metrics['loss'] = total_loss / total_samples

        return metrics

    def train(self, 
              dataset_path: str,
              epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.2,
              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            dataset_path: Path to the dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            save_path: Path to save the trained model

        Returns:
            Training history and final metrics
        """
        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"📁 Dataset path: {dataset_path}")
        print(f"🔢 Batch size: {batch_size}")
        print(f"📊 Validation split: {validation_split}")

        # Setup training components
        self.setup_training_components()

        # Get dataloaders
        train_loader = self.task_config.get_dataloader(
            dataset_path, batch_size=batch_size, shuffle=True
        )
        val_loader = self.task_config.get_dataloader(
            dataset_path, batch_size=batch_size, shuffle=False
        )

        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch + 1

            # Train for one epoch
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            # Log metrics
            epoch_data = {
                'epoch': self.current_epoch,
                'train': train_metrics,
                'val': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_data)

            # Print epoch summary
            print(f"\n📊 Epoch {self.current_epoch}/{epochs} Summary:")
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            print(f"   Val Loss: {val_metrics['loss']:.4f}")
            for metric_name, value in val_metrics.items():
                if metric_name != 'loss':
                    print(f"   Val {metric_name}: {value:.4f}")

            # Save best model
            primary_metric = self._get_primary_metric(val_metrics)
            if primary_metric > self.best_metric:
                self.best_metric = primary_metric
                if save_path:
                    self.save_model(save_path)
                    print(f"   💾 Best model saved! (metric: {primary_metric:.4f})")

        print(f"\n🎉 Training completed!")
        print(f"📈 Best validation metric: {self.best_metric:.4f}")

        return {
            'history': self.training_history,
            'best_metric': self.best_metric,
            'final_model': self.model
        }

    def _get_primary_metric(self, metrics: Dict[str, float]) -> float:
        """Get the primary metric for model selection."""
        # Default to first non-loss metric, or negative loss if no other metrics
        for key, value in metrics.items():
            if key != 'loss':
                return value
        return -metrics['loss']

    def save_model(self, path: str):
        """Save the trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'config': self.task_config.config if hasattr(self.task_config, 'config') else {}
        }

        torch.save(save_dict, path)
        print(f"💾 Model saved to: {path}")

    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)

        if self.model is None:
            self.setup_training_components()

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)

        print(f"📂 Model loaded from: {path}")
        print(f"📊 Loaded epoch: {self.current_epoch}, Best metric: {self.best_metric}")

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Make predictions on new data."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Please train or load a model first.")

        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)

        return outputs
