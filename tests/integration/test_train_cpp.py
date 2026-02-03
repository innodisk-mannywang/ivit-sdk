"""
Integration tests for C++ training module.

Tests API compatibility between Python and C++ implementations.
"""

import pytest
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path

# Import from the train module (will use C++ or Python backend)
from ivit.train import (
    ImageFolderDataset,
    get_backend,
    is_cpp_backend_available,
    Compose,
    Resize,
    RandomHorizontalFlip,
    ColorJitter,
    Normalize,
    ToTensor,
    get_train_augmentation,
    get_val_augmentation,
    EarlyStopping,
    ModelCheckpoint,
    ProgressLogger,
)


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary image folder dataset."""
    temp_dir = tempfile.mkdtemp()

    # Create class directories
    cat_dir = Path(temp_dir) / "cat"
    dog_dir = Path(temp_dir) / "dog"
    cat_dir.mkdir()
    dog_dir.mkdir()

    # Create dummy images (64x64 RGB)
    dummy_img = np.ones((64, 64, 3), dtype=np.uint8) * 128

    import cv2
    cv2.imwrite(str(cat_dir / "cat1.jpg"), dummy_img)
    cv2.imwrite(str(cat_dir / "cat2.jpg"), dummy_img)
    cv2.imwrite(str(dog_dir / "dog1.jpg"), dummy_img)
    cv2.imwrite(str(dog_dir / "dog2.jpg"), dummy_img)
    cv2.imwrite(str(dog_dir / "dog3.jpg"), dummy_img)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestBackend:
    """Test backend detection and switching."""

    def test_get_backend(self):
        """Test that get_backend returns a valid string."""
        backend = get_backend()
        assert backend in ("cpp", "python")

    def test_is_cpp_backend_available(self):
        """Test that is_cpp_backend_available returns a boolean."""
        result = is_cpp_backend_available()
        assert isinstance(result, bool)


class TestImageFolderDataset:
    """Test ImageFolderDataset API compatibility."""

    def test_load_dataset(self, temp_dataset_dir):
        """Test loading a dataset."""
        dataset = ImageFolderDataset(temp_dataset_dir, train_split=1.0, split="all")

        assert len(dataset) == 5
        assert dataset.num_classes == 2 or dataset.num_classes() == 2

    def test_class_names(self, temp_dataset_dir):
        """Test getting class names."""
        dataset = ImageFolderDataset(temp_dataset_dir, train_split=1.0, split="all")

        # Handle both property and method access
        if hasattr(dataset, 'class_names') and callable(dataset.class_names):
            names = dataset.class_names()
        else:
            names = dataset.class_names

        assert len(names) == 2
        assert "cat" in names
        assert "dog" in names

    def test_train_val_split(self, temp_dataset_dir):
        """Test train/val splitting."""
        train_ds = ImageFolderDataset(temp_dataset_dir, train_split=0.6, split="train")
        val_ds = ImageFolderDataset(temp_dataset_dir, train_split=0.6, split="val")

        assert len(train_ds) == 3  # 5 * 0.6 = 3
        assert len(val_ds) == 2

    def test_get_item(self, temp_dataset_dir):
        """Test getting an item."""
        dataset = ImageFolderDataset(temp_dataset_dir, train_split=1.0, split="all")

        image, label = dataset[0]

        assert isinstance(image, np.ndarray)
        assert image.shape == (64, 64, 3)
        assert isinstance(label, int)
        assert 0 <= label < 2


class TestTransforms:
    """Test transform API compatibility."""

    def test_resize(self):
        """Test Resize transform."""
        resize = Resize(32)
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128

        result = resize(image)

        assert result.shape[:2] == (32, 32)

    def test_random_horizontal_flip(self):
        """Test RandomHorizontalFlip transform."""
        flip = RandomHorizontalFlip(p=0.5)
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128

        result = flip(image)

        assert result.shape == image.shape

    def test_color_jitter(self):
        """Test ColorJitter transform."""
        jitter = ColorJitter(brightness=0.2, contrast=0.2)
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128

        result = jitter(image)

        assert result.shape == image.shape

    def test_normalize(self):
        """Test Normalize transform."""
        normalize = Normalize()
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128

        result = normalize(image)

        assert result.dtype == np.float32
        assert result.shape == image.shape

    def test_to_tensor(self):
        """Test ToTensor transform."""
        to_tensor = ToTensor()
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128

        result = to_tensor(image)

        # Should be NCHW format
        assert result.ndim == 4
        assert result.shape[0] == 1  # N
        assert result.shape[1] == 3  # C
        assert result.shape[2] == 64  # H
        assert result.shape[3] == 64  # W

    def test_compose(self):
        """Test Compose transform."""
        transform = Compose([
            Resize(32),
            Normalize(),
            ToTensor(),
        ])

        image = np.ones((64, 64, 3), dtype=np.uint8) * 128
        result = transform(image)

        assert result.ndim == 4
        assert result.shape[2] == 32
        assert result.shape[3] == 32

    def test_get_train_augmentation(self):
        """Test get_train_augmentation convenience function."""
        transform = get_train_augmentation(size=64, flip_p=0.5, color_jitter=True)

        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = transform(image)

        assert result.ndim == 4
        assert result.shape[2] == 64
        assert result.shape[3] == 64

    def test_get_val_augmentation(self):
        """Test get_val_augmentation convenience function."""
        transform = get_val_augmentation(size=64)

        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = transform(image)

        assert result.ndim == 4
        assert result.shape[2] == 64


class TestCallbacks:
    """Test callback API compatibility."""

    def test_early_stopping_creation(self):
        """Test EarlyStopping creation."""
        callback = EarlyStopping(monitor="val_loss", patience=5)

        assert callback is not None

    def test_model_checkpoint_creation(self):
        """Test ModelCheckpoint creation."""
        callback = ModelCheckpoint(
            filepath="test_{epoch:02d}.pt",
            monitor="val_loss",
            save_best_only=True,
        )

        assert callback is not None

    def test_progress_logger_creation(self):
        """Test ProgressLogger creation."""
        callback = ProgressLogger(log_frequency=10)

        assert callback is not None


@pytest.mark.skipif(
    not is_cpp_backend_available(),
    reason="C++ backend not available"
)
class TestCppSpecific:
    """Tests specific to C++ backend."""

    def test_cpp_specific_classes(self):
        """Test C++ specific classes are available."""
        from ivit.train import TrainerConfig, ExportOptions, TrainingMetrics

        config = TrainerConfig()
        config.epochs = 10
        config.learning_rate = 0.001

        assert config.epochs == 10
        assert config.learning_rate == 0.001

    def test_csv_logger(self):
        """Test CSVLogger (C++ only)."""
        from ivit.train import CSVLogger

        callback = CSVLogger(filepath="test_log.csv", append=False)
        assert callback is not None

    def test_center_crop(self):
        """Test CenterCrop (C++ only)."""
        from ivit.train import CenterCrop

        crop = CenterCrop(32)
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128

        result = crop(image)

        assert result.shape[:2] == (32, 32)

    def test_random_crop(self):
        """Test RandomCrop (C++ only)."""
        from ivit.train import RandomCrop

        crop = RandomCrop(32)
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128

        result = crop(image)

        assert result.shape[:2] == (32, 32)

    def test_gaussian_blur(self):
        """Test GaussianBlur (C++ only)."""
        from ivit.train import GaussianBlur

        blur = GaussianBlur(kernel_size=5, sigma=1.0)
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128

        result = blur(image)

        assert result.shape == image.shape


class TestAPICompatibility:
    """Test API compatibility between Python and C++ backends."""

    def test_dataset_iteration(self, temp_dataset_dir):
        """Test that datasets can be iterated."""
        dataset = ImageFolderDataset(temp_dataset_dir, train_split=1.0, split="all")

        for i in range(min(3, len(dataset))):
            image, label = dataset[i]
            assert isinstance(image, np.ndarray)
            assert isinstance(label, int)

    def test_transform_pipeline_output_type(self):
        """Test that transform pipeline outputs correct types."""
        transform = get_val_augmentation(size=64)
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128

        result = transform(image)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
