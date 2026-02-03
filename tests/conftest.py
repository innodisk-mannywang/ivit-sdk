"""
Shared pytest fixtures for iVIT-SDK tests.

This module provides common fixtures for:
- Mini dataset generation
- Temporary directories
- Mock models
- Training utilities
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Generator


# =============================================================================
# Dataset Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that is cleaned up after the test."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mini_image_folder_dataset(temp_dir: Path) -> Path:
    """
    Create a minimal ImageFolder dataset for testing.

    Structure:
        temp_dir/
            cat/
                cat_001.jpg
                cat_002.jpg
                cat_003.jpg
            dog/
                dog_001.jpg
                dog_002.jpg
                dog_003.jpg
            bird/
                bird_001.jpg
                bird_002.jpg

    Returns:
        Path to the dataset root directory.
    """
    import cv2

    classes = {
        "cat": 3,
        "dog": 3,
        "bird": 2,
    }

    for class_name, num_images in classes.items():
        class_dir = temp_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_images):
            # Create a unique colored image for each class
            if class_name == "cat":
                color = (200, 100, 100)  # Blueish (BGR)
            elif class_name == "dog":
                color = (100, 200, 100)  # Greenish (BGR)
            else:
                color = (100, 100, 200)  # Reddish (BGR)

            # Create 64x64 image with some variation
            img = np.full((64, 64, 3), color, dtype=np.uint8)
            # Add some noise for variation
            noise = np.random.randint(-20, 20, (64, 64, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            img_path = class_dir / f"{class_name}_{i + 1:03d}.jpg"
            cv2.imwrite(str(img_path), img)

    return temp_dir


@pytest.fixture
def mini_dataset_with_validation(temp_dir: Path) -> Tuple[Path, Path]:
    """
    Create separate train and validation ImageFolder datasets.

    Returns:
        Tuple of (train_dir, val_dir).
    """
    import cv2

    train_dir = temp_dir / "train"
    val_dir = temp_dir / "val"

    for split_dir, num_per_class in [(train_dir, 4), (val_dir, 2)]:
        for class_name in ["class_a", "class_b"]:
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for i in range(num_per_class):
                img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                img_path = class_dir / f"{class_name}_{i + 1:03d}.jpg"
                cv2.imwrite(str(img_path), img)

    return train_dir, val_dir


# =============================================================================
# PyTorch Fixtures (conditionally loaded)
# =============================================================================


@pytest.fixture
def torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.fixture
def cpu_device():
    """Get CPU device for PyTorch."""
    try:
        import torch
        return torch.device("cpu")
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture
def gpu_available() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# =============================================================================
# Training Fixtures
# =============================================================================


@pytest.fixture
def simple_train_dataset(mini_image_folder_dataset: Path):
    """Create a simple ImageFolderDataset for training tests."""
    try:
        from ivit.train import ImageFolderDataset
        return ImageFolderDataset(
            mini_image_folder_dataset,
            train_split=1.0,
            split="all"
        )
    except ImportError:
        pytest.skip("ivit.train not available")


@pytest.fixture
def train_val_datasets(mini_image_folder_dataset: Path):
    """Create train and validation datasets."""
    try:
        from ivit.train import ImageFolderDataset

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
        return train_ds, val_ds
    except ImportError:
        pytest.skip("ivit.train not available")


@pytest.fixture
def checkpoint_dir(temp_dir: Path) -> Path:
    """Create a temporary directory for model checkpoints."""
    ckpt_dir = temp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


@pytest.fixture
def export_dir(temp_dir: Path) -> Path:
    """Create a temporary directory for exported models."""
    exp_dir = temp_dir / "exports"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU (skipped without CUDA)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (skipped with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", "training: mark test as training-related"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available hardware."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    skip_gpu = pytest.mark.skip(reason="CUDA GPU not available")

    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
