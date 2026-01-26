"""
Dataset classes for training.

Supports ImageFolder, COCO, and YOLO formats.
"""

from typing import List, Tuple, Dict, Any, Optional, Callable, Union
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import logging
import random
import json

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Abstract base class for datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Any]:
        """Get sample by index."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return number of classes."""
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Return class names."""
        pass


class ImageFolderDataset(BaseDataset):
    """
    Dataset that loads images from folder structure.

    Expected structure:
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
                img4.jpg

    Args:
        root: Root directory containing class folders
        transforms: Transform function to apply to images
        train_split: Fraction of data to use for training (0-1)
        split: "train", "val", or "all"
        seed: Random seed for reproducible splits

    Examples:
        >>> dataset = ImageFolderDataset("./data", train_split=0.8, split="train")
        >>> image, label = dataset[0]
        >>> print(f"Classes: {dataset.class_names}")
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    def __init__(
        self,
        root: Union[str, Path],
        transforms: Optional[Callable] = None,
        train_split: float = 0.8,
        split: str = "train",
        seed: int = 42,
    ):
        self.root = Path(root)
        self.transforms = transforms
        self.train_split = train_split
        self.split = split.lower()
        self.seed = seed

        if not self.root.exists():
            raise ValueError(f"Dataset root does not exist: {root}")

        # Discover classes and samples
        self._classes: List[str] = []
        self._samples: List[Tuple[Path, int]] = []  # (image_path, class_idx)

        self._load_dataset()
        self._apply_split()

        logger.info(f"Loaded {len(self)} samples from {self.root}")
        logger.info(f"Classes: {len(self._classes)}")

    def _load_dataset(self):
        """Load dataset from folder structure."""
        # Get class directories
        class_dirs = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

        if not class_dirs:
            raise ValueError(f"No class directories found in {self.root}")

        self._classes = [d.name for d in class_dirs]
        logger.debug(f"Found classes: {self._classes}")

        # Load samples
        all_samples = []
        for class_idx, class_dir in enumerate(class_dirs):
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    all_samples.append((img_path, class_idx))

        if not all_samples:
            raise ValueError(f"No valid images found in {self.root}")

        # Shuffle samples with fixed seed for reproducibility
        random.seed(self.seed)
        random.shuffle(all_samples)

        self._all_samples = all_samples

    def _apply_split(self):
        """Apply train/val split."""
        n_total = len(self._all_samples)
        n_train = int(n_total * self.train_split)

        if self.split == "train":
            self._samples = self._all_samples[:n_train]
        elif self.split == "val":
            self._samples = self._all_samples[n_train:]
        else:  # "all"
            self._samples = self._all_samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Get sample by index.

        Returns:
            Tuple of (image, label)
        """
        import cv2

        img_path, label = self._samples[idx]
        image = cv2.imread(str(img_path))

        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    @property
    def num_classes(self) -> int:
        return len(self._classes)

    @property
    def class_names(self) -> List[str]:
        return self._classes

    @property
    def calibration_set(self) -> List[np.ndarray]:
        """Get subset of images for calibration (quantization)."""
        n_samples = min(100, len(self))
        indices = random.sample(range(len(self)), n_samples)
        return [self[i][0] for i in indices]


class COCODataset(BaseDataset):
    """
    Dataset for COCO format annotations.

    Args:
        root: Root directory containing images
        annotation_file: Path to COCO JSON annotation file
        transforms: Transform function
        split: "train" or "val"

    Examples:
        >>> dataset = COCODataset(
        ...     root="./coco/images",
        ...     annotation_file="./coco/annotations/instances_train2017.json",
        ... )
    """

    def __init__(
        self,
        root: Union[str, Path],
        annotation_file: Union[str, Path],
        transforms: Optional[Callable] = None,
        split: str = "train",
    ):
        self.root = Path(root)
        self.annotation_file = Path(annotation_file)
        self.transforms = transforms
        self.split = split

        if not self.annotation_file.exists():
            raise ValueError(f"Annotation file not found: {annotation_file}")

        self._load_annotations()

    def _load_annotations(self):
        """Load COCO annotations."""
        with open(self.annotation_file) as f:
            coco = json.load(f)

        # Build category mapping
        self._categories = {cat['id']: cat['name'] for cat in coco['categories']}
        self._class_names = list(self._categories.values())

        # Build image info mapping
        self._images = {img['id']: img for img in coco['images']}

        # Group annotations by image
        self._annotations: Dict[int, List[Dict]] = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self._annotations:
                self._annotations[img_id] = []
            self._annotations[img_id].append(ann)

        # Create sample list
        self._samples = list(self._annotations.keys())

        logger.info(f"Loaded {len(self._samples)} images with {len(coco['annotations'])} annotations")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get sample by index.

        Returns:
            Tuple of (image, target) where target contains:
                - boxes: (N, 4) array of [x1, y1, x2, y2]
                - labels: (N,) array of class indices
        """
        import cv2

        img_id = self._samples[idx]
        img_info = self._images[img_id]
        img_path = self.root / img_info['file_name']

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract annotations
        anns = self._annotations[img_id]
        boxes = []
        labels = []

        for ann in anns:
            if 'bbox' in ann:
                # COCO bbox format: [x, y, width, height]
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])

        target = {
            'boxes': np.array(boxes, dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64),
            'image_id': img_id,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    @property
    def class_names(self) -> List[str]:
        return self._class_names


class YOLODataset(BaseDataset):
    """
    Dataset for YOLO format annotations.

    Expected structure:
        root/
            images/
                train/
                    img1.jpg
                val/
                    img2.jpg
            labels/
                train/
                    img1.txt
                val/
                    img2.txt

    Label format (per line): class_id cx cy w h (normalized 0-1)

    Args:
        root: Root directory
        split: "train" or "val"
        transforms: Transform function

    Examples:
        >>> dataset = YOLODataset("./yolo_dataset", split="train")
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transforms: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        self._class_names = class_names or []
        self._samples: List[Tuple[Path, Path]] = []  # (image_path, label_path)

        self._load_dataset()

    def _load_dataset(self):
        """Load YOLO dataset."""
        images_dir = self.root / "images" / self.split
        labels_dir = self.root / "labels" / self.split

        if not images_dir.exists():
            # Try alternative structure
            images_dir = self.root / self.split / "images"
            labels_dir = self.root / self.split / "labels"

        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")

        # Load class names if available
        names_file = self.root / "classes.txt"
        if names_file.exists() and not self._class_names:
            with open(names_file) as f:
                self._class_names = [line.strip() for line in f if line.strip()]

        # Find image-label pairs
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() in ImageFolderDataset.SUPPORTED_EXTENSIONS:
                label_path = labels_dir / (img_path.stem + ".txt")
                if label_path.exists():
                    self._samples.append((img_path, label_path))

        logger.info(f"Loaded {len(self._samples)} samples from {self.root}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get sample by index.

        Returns:
            Tuple of (image, target) where target contains:
                - boxes: (N, 4) array of [x1, y1, x2, y2]
                - labels: (N,) array of class indices
        """
        import cv2

        img_path, label_path = self._samples[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Parse YOLO labels
        boxes = []
        labels = []

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])

                    # Convert normalized YOLO to pixel coordinates
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h

                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)

        target = {
            'boxes': np.array(boxes, dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @property
    def num_classes(self) -> int:
        if self._class_names:
            return len(self._class_names)
        # Infer from labels
        max_class = 0
        for _, label_path in self._samples[:100]:  # Sample first 100
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        max_class = max(max_class, int(parts[0]))
        return max_class + 1

    @property
    def class_names(self) -> List[str]:
        return self._class_names


def split_dataset(
    dataset: BaseDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Split dataset indices into train and validation sets.

    Args:
        dataset: Dataset to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        seed: Random seed

    Returns:
        Tuple of (train_indices, val_indices)

    Examples:
        >>> train_idx, val_idx = split_dataset(dataset, train_ratio=0.8)
    """
    n = len(dataset)
    indices = list(range(n))

    random.seed(seed)
    random.shuffle(indices)

    n_train = int(n * train_ratio)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    return train_indices, val_indices
