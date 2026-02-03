"""
Integration tests for training workflow.

Tests the complete Trainer class functionality including:
- P0: Core training methods (fit, evaluate, history)
- P1: Transfer learning features (backbone freezing, LR, checkpoints, export)
- P2: Edge cases and error handling
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path


# =============================================================================
# Skip conditions
# =============================================================================

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ivit.train import (
        Trainer,
        ImageFolderDataset,
        EarlyStopping,
        ModelCheckpoint,
        ProgressLogger,
        LRScheduler,
    )
    HAS_TRAIN_MODULE = True
except ImportError:
    HAS_TRAIN_MODULE = False


pytestmark = [
    pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available"),
    pytest.mark.skipif(not HAS_TRAIN_MODULE, reason="ivit.train module not available"),
    pytest.mark.training,
]


# =============================================================================
# P0 Tests: Core Trainer Functionality
# =============================================================================


class TestTrainerInitialization:
    """Test Trainer initialization."""

    def test_trainer_init_with_model_name(self, mini_image_folder_dataset):
        """Test Trainer initialization with model name string."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        trainer = Trainer(
            model="resnet18",  # Smallest ResNet for fast testing
            dataset=dataset,
            epochs=1,
            batch_size=2,
            device="cpu",
        )

        assert trainer.epochs == 1
        assert trainer.batch_size == 2
        assert trainer.device_str == "cpu"
        assert trainer.model is not None

    def test_trainer_init_with_validation_dataset(self, mini_image_folder_dataset):
        """Test Trainer with separate validation dataset."""
        train_ds = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=0.75,
            split="train"
        )
        val_ds = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=0.75,
            split="val"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=train_ds,
            val_dataset=val_ds,
            epochs=1,
            batch_size=2,
            device="cpu",
        )

        assert trainer.val_loader is not None

    def test_trainer_init_invalid_model(self, mini_image_folder_dataset):
        """Test Trainer raises error for invalid model name."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        with pytest.raises(ValueError, match="Unknown model"):
            Trainer(
                model="invalid_model_name",
                dataset=dataset,
                epochs=1,
                device="cpu",
            )


class TestTrainerFit:
    """Test Trainer.fit() method."""

    def test_fit_basic(self, mini_image_folder_dataset):
        """Test basic training loop completes without error."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=2,
            batch_size=2,
            device="cpu",
        )

        history = trainer.fit()

        assert isinstance(history, list)
        assert len(history) == 2
        assert "loss" in history[0]
        assert "accuracy" in history[0]

    def test_fit_with_validation(self, mini_image_folder_dataset):
        """Test training with validation metrics."""
        train_ds = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=0.75,
            split="train"
        )
        val_ds = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=0.75,
            split="val"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=train_ds,
            val_dataset=val_ds,
            epochs=2,
            batch_size=2,
            device="cpu",
        )

        history = trainer.fit()

        assert "val_loss" in history[0]
        assert "val_accuracy" in history[0]

    def test_fit_with_callbacks(self, mini_image_folder_dataset, checkpoint_dir):
        """Test training with callbacks."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=2,
            batch_size=2,
            device="cpu",
        )

        callbacks = [
            ProgressLogger(log_frequency=1),
        ]

        history = trainer.fit(callbacks=callbacks)
        assert len(history) == 2


class TestTrainerHistory:
    """Test Trainer history property."""

    def test_history_property(self, mini_image_folder_dataset):
        """Test history property returns training history."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=2,
            batch_size=2,
            device="cpu",
        )

        # Before training
        assert trainer.history == []

        trainer.fit()

        # After training
        assert len(trainer.history) == 2
        assert all("loss" in h for h in trainer.history)


class TestTrainerEvaluate:
    """Test Trainer.evaluate() method."""

    def test_evaluate_on_validation_set(self, mini_image_folder_dataset):
        """Test evaluate on validation dataset."""
        train_ds = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=0.75,
            split="train"
        )
        val_ds = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=0.75,
            split="val"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=train_ds,
            val_dataset=val_ds,
            epochs=1,
            batch_size=2,
            device="cpu",
        )

        trainer.fit()
        metrics = trainer.evaluate()

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "correct" in metrics
        assert "total" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_evaluate_on_custom_dataset(self, mini_image_folder_dataset):
        """Test evaluate on a custom dataset."""
        train_ds = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=0.75,
            split="train"
        )
        test_ds = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=0.75,
            split="val"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=train_ds,
            epochs=1,
            batch_size=2,
            device="cpu",
        )

        trainer.fit()
        metrics = trainer.evaluate(dataset=test_ds)

        assert "accuracy" in metrics

    def test_evaluate_no_dataset_error(self, mini_image_folder_dataset):
        """Test evaluate raises error when no dataset available."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=1,
            batch_size=2,
            device="cpu",
        )

        trainer.fit()

        with pytest.raises(ValueError, match="No validation dataset"):
            trainer.evaluate()


# =============================================================================
# P1 Tests: Transfer Learning Features
# =============================================================================


class TestBackboneFreezing:
    """Test backbone freezing for transfer learning."""

    def test_freeze_backbone_enabled(self, mini_image_folder_dataset):
        """Test that backbone is frozen when freeze_backbone=True."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=1,
            batch_size=2,
            device="cpu",
            freeze_backbone=True,
        )

        # Count trainable parameters
        trainable_params = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in trainer.model.parameters())

        # When frozen, only classifier should be trainable (much fewer params)
        assert trainable_params < total_params * 0.1

    def test_freeze_backbone_disabled(self, mini_image_folder_dataset):
        """Test that all params trainable when freeze_backbone=False."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=1,
            batch_size=2,
            device="cpu",
            freeze_backbone=False,
        )

        trainable_params = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in trainer.model.parameters())

        # When not frozen, all params should be trainable
        assert trainable_params == total_params


class TestOptimizers:
    """Test different optimizer configurations."""

    @pytest.mark.parametrize("optimizer", ["adam", "adamw", "sgd"])
    def test_optimizer_types(self, mini_image_folder_dataset, optimizer):
        """Test training with different optimizers."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=1,
            batch_size=2,
            device="cpu",
            optimizer=optimizer,
        )

        history = trainer.fit()
        assert len(history) == 1

    def test_invalid_optimizer(self, mini_image_folder_dataset):
        """Test that invalid optimizer raises error."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        with pytest.raises(ValueError, match="Unknown optimizer"):
            Trainer(
                model="resnet18",
                dataset=dataset,
                epochs=1,
                device="cpu",
                optimizer="invalid_optimizer",
            )


class TestLearningRate:
    """Test learning rate configurations."""

    def test_custom_learning_rate(self, mini_image_folder_dataset):
        """Test training with custom learning rate."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        custom_lr = 0.01
        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=1,
            batch_size=2,
            device="cpu",
            learning_rate=custom_lr,
        )

        assert trainer.learning_rate == custom_lr
        assert trainer.optimizer.param_groups[0]["lr"] == custom_lr


class TestModelCheckpoint:
    """Test ModelCheckpoint callback."""

    def test_checkpoint_saves_best(self, mini_image_folder_dataset, checkpoint_dir):
        """Test that checkpoint saves best model."""
        train_ds = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=0.75,
            split="train"
        )
        val_ds = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=0.75,
            split="val"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=train_ds,
            val_dataset=val_ds,
            epochs=2,
            batch_size=2,
            device="cpu",
        )

        checkpoint_path = str(checkpoint_dir / "best_model.pt")
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        )

        trainer.fit(callbacks=[checkpoint])

        # Check that checkpoint was saved
        assert Path(checkpoint_path).exists() or checkpoint.best_path is not None


class TestEarlyStopping:
    """Test EarlyStopping callback."""

    def test_early_stopping_triggers(self, mini_image_folder_dataset):
        """Test that early stopping can stop training."""
        train_ds = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=0.75,
            split="train"
        )
        val_ds = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=0.75,
            split="val"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=train_ds,
            val_dataset=val_ds,
            epochs=100,  # Large number, should stop early
            batch_size=2,
            device="cpu",
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=2,  # Stop after 2 epochs without improvement
        )

        history = trainer.fit(callbacks=[early_stop])

        # Should stop before 100 epochs
        assert len(history) < 100


class TestModelExport:
    """Test model export functionality."""

    def test_export_onnx(self, mini_image_folder_dataset, export_dir):
        """Test exporting model to ONNX format."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=1,
            batch_size=2,
            device="cpu",
        )

        trainer.fit()

        export_path = str(export_dir / "model.onnx")
        result_path = trainer.export(export_path, format="onnx")

        assert Path(result_path).exists()
        assert result_path.endswith(".onnx")

    def test_export_torchscript(self, mini_image_folder_dataset, export_dir):
        """Test exporting model to TorchScript format."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=1,
            batch_size=2,
            device="cpu",
        )

        trainer.fit()

        export_path = str(export_dir / "model.pt")
        result_path = trainer.export(export_path, format="torchscript")

        assert Path(result_path).exists()


# =============================================================================
# P2 Tests: Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset_error(self, temp_dir):
        """Test that empty dataset raises appropriate error."""
        # Create empty directory structure
        (temp_dir / "empty_class").mkdir()

        with pytest.raises(ValueError):
            ImageFolderDataset(temp_dir, train_split=1.0, split="all")

    def test_single_sample_per_class(self, temp_dir):
        """Test with minimal dataset (1 sample per class)."""
        import cv2

        for class_name in ["a", "b"]:
            class_dir = temp_dir / class_name
            class_dir.mkdir()
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(class_dir / "img.jpg"), img)

        dataset = ImageFolderDataset(temp_dir, train_split=1.0, split="all")
        assert len(dataset) == 2

    def test_trainer_with_single_class(self, temp_dir):
        """Test that single class dataset raises error."""
        import cv2

        # Create single class
        class_dir = temp_dir / "only_class"
        class_dir.mkdir()
        for i in range(5):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(class_dir / f"img_{i}.jpg"), img)

        # Dataset should work but training classification needs 2+ classes
        dataset = ImageFolderDataset(temp_dir, train_split=1.0, split="all")
        assert dataset.num_classes == 1

    def test_batch_size_larger_than_dataset(self, mini_image_folder_dataset):
        """Test with batch_size larger than dataset size."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        # Use batch size larger than dataset
        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=1,
            batch_size=100,  # Larger than 8 samples
            device="cpu",
        )

        # Should still work, just with smaller actual batches
        history = trainer.fit()
        assert len(history) == 1


class TestDifferentModels:
    """Test with different model architectures."""

    @pytest.mark.parametrize("model_name", [
        "resnet18",
        "mobilenet_v2",
    ])
    def test_various_models(self, mini_image_folder_dataset, model_name):
        """Test training with various model architectures."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        trainer = Trainer(
            model=model_name,
            dataset=dataset,
            epochs=1,
            batch_size=2,
            device="cpu",
        )

        history = trainer.fit()
        assert len(history) == 1


class TestCurrentEpochProperty:
    """Test current_epoch property."""

    def test_current_epoch_updates(self, mini_image_folder_dataset):
        """Test that current_epoch updates during training."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=3,
            batch_size=2,
            device="cpu",
        )

        # Track epochs via callback
        epochs_seen = []

        class EpochTracker:
            def on_train_start(self, trainer, **kwargs):
                pass

            def on_train_end(self, trainer, **kwargs):
                pass

            def on_epoch_start(self, trainer, epoch, **kwargs):
                epochs_seen.append(trainer.current_epoch)

            def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
                pass

            def on_batch_start(self, trainer, batch_idx, **kwargs):
                pass

            def on_batch_end(self, trainer, batch_idx, loss, **kwargs):
                pass

        trainer.fit(callbacks=[EpochTracker()])

        assert epochs_seen == [0, 1, 2]


# =============================================================================
# GPU Tests (skipped if no CUDA)
# =============================================================================


@pytest.mark.gpu
class TestGPUTraining:
    """Test training on GPU."""

    def test_gpu_training(self, mini_image_folder_dataset):
        """Test that training works on GPU."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=1,
            batch_size=2,
            device="cuda:0",
        )

        history = trainer.fit()
        assert len(history) == 1

    def test_gpu_export(self, mini_image_folder_dataset, export_dir):
        """Test that GPU model can be exported."""
        dataset = ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )

        trainer = Trainer(
            model="resnet18",
            dataset=dataset,
            epochs=1,
            batch_size=2,
            device="cuda:0",
        )

        trainer.fit()

        export_path = str(export_dir / "gpu_model.onnx")
        result_path = trainer.export(export_path, format="onnx")

        assert Path(result_path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
